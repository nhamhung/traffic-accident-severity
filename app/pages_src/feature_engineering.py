"""Trang Chuẩn bị dữ liệu: ba quyết định quan trọng trước khi cho máy học."""

import pandas as pd
import plotly.express as px
import streamlit as st

from . import shared
from traffic_accident_severity import config, vi


def render() -> None:
    st.title("🔧 Chuẩn bị dữ liệu")
    st.markdown("Trước khi cho máy học, cần quyết định **cho nó xem gì** và **kiểm tra nó thế nào**. "
                "Ba quyết định dưới đây quan trọng hơn cả việc chọn mô hình nào.")
    facts = shared.get_json("data_facts.json")
    leak = shared.get_result("leak_check.csv").set_index("folds")

    st.subheader("1. Không \"nhìn trộm\" kết quả")
    st.markdown(
        "Dữ liệu có hai loại thông tin:\n\n"
        "- **Trước hoặc lúc va chạm:** người lái, loại xe, con đường, thời tiết, kiểu va chạm… ✅ Mô hình được dùng.\n"
        "- **Sau va chạm:** ai bị thương, bị thương nặng đến đâu, bao nhiêu người thương vong… ❌ Mô hình **không** "
        "được dùng.\n\n"
        "Nếu cho mô hình biết \"có người bị thương nặng\" rồi hỏi \"vụ này có nghiêm trọng không\", nó sẽ "
        "trả lời rất giỏi, nhưng hoàn toàn **vô dụng**: giống như được xem đáp án trước khi làm bài thi."
    )
    excluded = ", ".join(vi.col(c) if c in vi.COLUMN else c for c in config.CASUALTY_OUTCOME_COLS)
    st.caption(f"Các cột bị loại ({len(config.CASUALTY_OUTCOME_COLS)}): {excluded}.")

    st.subheader("2. Một vụ tai nạn, nhiều dòng dữ liệu")
    st.markdown(
        f"Khi kiểm tra kỹ, dự án phát hiện **{shared.num(facts['rows_in_multi_row_accidents'])} dòng** "
        f"({shared.pct(facts['rows_in_multi_row_accidents'] / facts['n_rows'])}) thật ra thuộc về cùng một vụ tai nạn "
        "với dòng khác: cùng giờ, cùng ngày, cùng nơi, cùng thời tiết, cùng kiểu va chạm. Cảnh sát ghi **mỗi xe một "
        f"dòng**. Trong những nhóm này, mức độ nghiêm trọng giống nhau **{shared.pct(facts['multi_row_same_severity'])}** "
        f"số lần (nếu là các vụ khác nhau thì chỉ khoảng {shared.pct(facts['multi_row_same_severity_by_chance'])})."
    )
    st.markdown(
        "**Vì sao điều này quan trọng?** Để biết mô hình giỏi thật hay không, ta giấu đi một phần dữ liệu làm "
        "\"đề thi\". Nếu một dòng của vụ tai nạn nằm trong phần học, còn dòng kia nằm trong đề thi, mô hình chỉ cần "
        "**nhớ** chứ không cần **hiểu**. Vì vậy dự án luôn giữ **tất cả các dòng của một vụ ở cùng một phía**."
    )
    df = pd.DataFrame({"Cách chia đề thi": ["Chia ngẫu nhiên (bị lộ đề)", "Giữ nguyên từng vụ (công bằng)"],
                       "Điểm": [leak.loc["random", "pr_auc"], leak.loc["grouped", "pr_auc"]]})
    fig = px.bar(df, x="Điểm", y="Cách chia đề thi", orientation="h", text=[shared.num(v, 3) for v in df["Điểm"]],
                 labels={"Điểm": "Điểm PR-AUC của mô hình cũ (càng cao càng giỏi)"})
    fig.update_traces(marker_color=[shared.GREY, shared.SAFE_COLOUR])
    fig.update_layout(height=220, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Khi bị lộ đề, mô hình cũ trông giỏi hơn thực tế một chút. Mọi con số trong ứng dụng này đều "
               "dùng cách chia công bằng.")

    st.subheader("3. Giờ là một vòng tròn")
    st.markdown(
        "Với máy tính, 23 giờ và 0 giờ là hai con số cách xa nhau (23 và 0). Nhưng thật ra chúng chỉ cách nhau "
        "**một tiếng**. Mô hình của dự án làm việc với giờ dưới dạng số thập phân (ví dụ 17,5 = 17 giờ 30) và "
        "tự học được những khoảng giờ nguy hiểm, như trên mặt đồng hồ."
    )
    missing = pd.Series(facts["missing_share"]).sort_values(ascending=False)
    missing = missing[missing > 0]
    st.subheader("4. Thông tin bị thiếu")
    fig = px.bar(x=[vi.col(c) for c in missing.index], y=missing.values, text=[shared.pct(v) for v in missing.values],
                 labels={"x": "", "y": "Tỉ lệ bị thiếu"})
    fig.update_layout(yaxis_tickformat=".0%", height=320, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.markdown("Một số cột bị bỏ trống khá nhiều. Thay vì xóa những dòng đó (sẽ mất rất nhiều dữ liệu), mô hình "
                "coi \"không ghi nhận\" là một giá trị riêng, vì việc thiếu thông tin đôi khi cũng nói lên điều gì đó.")
    shared.source_note()
