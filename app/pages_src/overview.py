"""Trang Dữ liệu nói gì? Những phát hiện rút ra trực tiếp từ dữ liệu, chưa cần mô hình."""

import pandas as pd
import plotly.express as px
import streamlit as st

from . import shared
from traffic_accident_severity import vi


def _rate_chart(rates: pd.DataFrame, column: str, base_rate: float, order: list | None = None, title: str = ""):
    r = rates[rates["column"] == column].copy()
    r = r[r["n"] >= 30]
    if order:
        r["value"] = pd.Categorical(r["value"], order, ordered=True)
        r = r.sort_values("value")
    else:
        r = r.sort_values("severe_rate")
    r["label"] = [str(v) if column in ("Number_of_vehicles_involved", "hour") else vi.val(column, v) for v in r["value"]]
    r["Tỉ lệ nghiêm trọng"] = r["severe_rate"]
    horizontal = order is None
    fig = px.bar(r, x="Tỉ lệ nghiêm trọng" if horizontal else "label", y="label" if horizontal else "Tỉ lệ nghiêm trọng",
                 orientation="h" if horizontal else "v", title=title, text=[shared.pct(v) for v in r["severe_rate"]],
                 hover_data={"n": True}, labels={"label": "", "n": "Số vụ"})
    fig.update_traces(marker_color=[shared.SEVERE_COLOUR if v > base_rate * 1.25 else shared.SAFE_COLOUR
                                    for v in r["severe_rate"]])
    line = dict(line_dash="dash", line_color="gray", annotation_text=f"Trung bình {shared.pct(base_rate)}")
    (fig.add_vline if horizontal else fig.add_hline)(base_rate, **line)
    fig.update_layout(height=320 if horizontal else 340, margin=dict(t=50, b=10),
                      xaxis_tickformat=".0%" if horizontal else None,
                      yaxis_tickformat=None if horizontal else ".0%")
    return fig


def render() -> None:
    st.title("📊 Dữ liệu nói gì?")
    facts = shared.get_json("data_facts.json")
    rates = shared.get_result("rates_by.csv")
    base = facts["severe_share"]

    st.markdown(
        "Dữ liệu gồm **12.316 bản ghi tai nạn có thật** do cảnh sát thành phố Addis Ababa (thủ đô Ethiopia) "
        "ghi lại từ 2017 đến 2020. Vì sao không dùng dữ liệu Việt Nam? Vì hồ sơ tai nạn ở Việt Nam chưa được "
        "công bố công khai. Addis Ababa được chọn vì giao thông ở đó **khá giống** các thành phố Việt Nam: "
        "xe máy, xe buýt, ô tô và người đi bộ dùng chung đường."
    )
    c = st.columns(4)
    c[0].metric("Bản ghi", shared.num(facts["n_rows"]))
    c[1].metric("Vụ tai nạn (ước tính)", shared.num(facts["n_accidents"]),
                help="Nhiều vụ được ghi thành nhiều dòng, mỗi xe một dòng. Xem trang Chuẩn bị dữ liệu.")
    c[2].metric("Vụ nghiêm trọng", shared.pct(base, 1).replace(",0", ""))
    c[3].metric("Thông tin về mỗi vụ", f"{facts['n_features']} cột")

    st.subheader("1. Phần lớn tai nạn chỉ gây thương tích nhẹ")
    shares = pd.DataFrame({"Mức độ": [vi.cls(k) for k in facts["raw_shares"]], "Tỉ lệ": list(facts["raw_shares"].values())})
    fig = px.bar(shares, x="Mức độ", y="Tỉ lệ", text=[shared.pct(v, 1) for v in shares["Tỉ lệ"]],
                 color="Mức độ", color_discrete_sequence=[shared.SAFE_COLOUR, "#f4a261", shared.SEVERE_COLOUR])
    fig.update_layout(showlegend=False, yaxis_tickformat=".0%", height=300, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        f"Chỉ **{shared.pct(facts['raw_shares']['Fatal injury'], 1)}** số vụ có người tử vong, quá ít để máy học nhận ra "
        "riêng. Vì vậy dự án gộp **bị thương nặng** và **tử vong** thành một nhóm: **nghiêm trọng** "
        f"(khoảng {shared.pct(base)} số vụ). Câu hỏi của mô hình là: *vụ tai nạn này có nghiêm trọng không?*"
    )

    st.subheader("2. Trời tối mà không có đèn đường: nguy hiểm gần gấp đôi")
    st.plotly_chart(_rate_chart(rates, "Light_conditions", base), width="stretch")
    st.markdown(
        "Đây là phát hiện rõ nhất. Điều đáng chú ý: **ban đêm có đèn đường** gần an toàn như ban ngày. "
        "Vậy không phải \"ban đêm\" gây nguy hiểm, mà là **thiếu đèn đường**, một điều có thể sửa được."
    )

    st.subheader("3. Một xe, hoặc rất nhiều xe: đều nguy hiểm hơn")
    st.plotly_chart(_rate_chart(rates, "Number_of_vehicles_involved", base,
                                order=sorted(rates.loc[rates["column"] == "Number_of_vehicles_involved", "value"].unique(),
                                             key=lambda v: int(float(v))),
                                title="Tỉ lệ nghiêm trọng theo số xe liên quan"), width="stretch")
    st.markdown(
        "Biểu đồ có hình **chữ U**: tai nạn **chỉ một xe** (mất lái, tông người đi bộ) thường nặng hơn vụ hai, ba xe "
        "va quẹt nhau. Một phép tính \"tương quan\" thông thường chỉ đo được xu hướng **theo một chiều**, nên sẽ "
        "nói rằng số xe *chẳng liên quan gì*, dù thật ra nó rất quan trọng."
    )

    st.subheader("4. Người lái rất trẻ và người lái lớn tuổi")
    st.plotly_chart(_rate_chart(rates, "Age_band_of_driver", base,
                                order=["Under 18", "18-30", "31-50", "Over 51", "Unknown"],
                                title="Tỉ lệ nghiêm trọng theo tuổi người lái"), width="stretch")
    st.markdown("Lại một hình chữ U: **dưới 18 tuổi** (thiếu kinh nghiệm) và **trên 51 tuổi** (phản xạ chậm hơn, "
                "cơ thể dễ tổn thương hơn) có tỉ lệ nghiêm trọng cao hơn nhóm ở giữa.")

    st.subheader("5. Theo giờ trong ngày")
    hours = rates[rates["column"] == "hour"].copy()
    hours["hour"] = hours["value"].astype(float).astype(int)
    fig = px.line(hours.sort_values("hour"), x="hour", y="severe_rate", markers=True,
                  labels={"hour": "Giờ", "severe_rate": "Tỉ lệ nghiêm trọng"})
    fig.add_hline(base, line_dash="dash", line_color="gray", annotation_text=f"Trung bình {shared.pct(base)}")
    fig.update_layout(yaxis_tickformat=".0%", height=320, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.markdown("Tai nạn ban đêm và rạng sáng nặng hơn, khớp với phát hiện về đèn đường ở trên. Hai cách nhìn "
                "khác nhau cùng chỉ về một hướng, nên ta tin hơn rằng đây là quy luật thật, không phải ngẫu nhiên.")

    st.subheader("6. Kiểu va chạm")
    st.plotly_chart(_rate_chart(rates, "Type_of_collision", base), width="stretch")

    shared.glossary("Tỉ lệ nghiêm trọng", (
        "Trong tất cả các vụ thuộc một nhóm (ví dụ: *ban đêm, không có đèn đường*), có bao nhiêu phần trăm "
        "là vụ nghiêm trọng. Đường gạch đứt là mức trung bình của toàn bộ dữ liệu. Nhóm có ít hơn 30 vụ được "
        "bỏ qua vì quá ít để tin."
    ))
    st.info("**Lưu ý:** đây là **mối liên hệ**, chưa phải **nguyên nhân**. Ví dụ, đường không có đèn có thể "
            "cũng là đường xe chạy nhanh hơn. Muốn chứng minh nguyên nhân cần thêm nghiên cứu khác.")
    shared.source_note()
