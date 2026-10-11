"""Giải thích cách dùng dự đoán như một bài học quản lý rủi ro giao thông."""

import pandas as pd
import plotly.express as px
import streamlit as st

from . import shared
from traffic_accident_severity import vi


def _per_1000(row: pd.Series, base_rate: float) -> dict:
    """Convert precision/recall into an accessible scenario of 1,000 accidents."""
    severe = round(1000 * base_rate)
    caught = round(severe * row["recall"])
    alarms = round(caught / row["precision"]) if row["precision"] > 0 else 0
    return {
        "severe": severe,
        "caught": caught,
        "missed": severe - caught,
        "false_alarms": alarms - caught,
        "alarms": alarms,
    }


def render() -> None:
    st.title("🚦 Ứng dụng vào quản lý giao thông")
    facts = shared.get_json("data_facts.json")
    summary = shared.get_json("summary.json")
    tradeoff = shared.get_result("threshold_tradeoff.csv")
    importance = shared.get_result("feature_importance.csv")
    base = facts["severe_share"]

    st.markdown(
        "Trang này biến kết quả dự đoán thành một bài tập quản lý: **nơi nào cần khảo sát trước, khi nào nên "
        "phát cảnh báo, và con người phải kiểm tra điều gì trước khi hành động**. Mô hình chỉ là công cụ hỗ trợ "
        "xếp ưu tiên; nó không thay thế cảnh sát, kỹ sư giao thông hay dữ liệu thực địa."
    )

    st.subheader("1. Một quy trình hỗ trợ quyết định")
    st.markdown(
        "1. **Quan sát:** ghi nhận ánh sáng, thời tiết, loại đường, phương tiện và hoàn cảnh va chạm.\n"
        "2. **Sàng lọc:** mô hình ước tính nguy cơ nghiêm trọng cho từng tình huống.\n"
        "3. **Ưu tiên khảo sát:** cán bộ xem lại những trường hợp có nguy cơ cao và kiểm tra tại hiện trường.\n"
        "4. **Can thiệp:** cân nhắc chiếu sáng, tổ chức giao lộ, giới hạn tốc độ, biển báo hoặc tuyên truyền.\n"
        "5. **Đánh giá lại:** đo kết quả sau can thiệp và cập nhật hệ thống bằng dữ liệu địa phương."
    )

    st.subheader("2. Chọn mức cảnh báo là một quyết định chính sách")
    st.markdown(
        "Mô hình trả về một **xác suất**, không phải lời khẳng định chắc chắn. Kéo thanh trượt để xem sự đánh đổi "
        "trên **1.000 vụ tai nạn**: cảnh báo sớm hơn giúp bắt được nhiều vụ nghiêm trọng hơn, nhưng cũng tạo thêm "
        "báo động nhầm."
    )
    thresholds = tradeoff["threshold"].round(2).tolist()
    default = min(thresholds, key=lambda value: abs(value - summary["threshold"]))
    threshold = st.select_slider(
        "Bật cảnh báo khi khả năng nghiêm trọng từ…",
        thresholds,
        value=default,
        format_func=lambda value: shared.pct(value),
    )
    row = tradeoff.iloc[(tradeoff["threshold"] - threshold).abs().argmin()]
    counts = _per_1000(row, base)
    columns = st.columns(4)
    columns[0].metric("Vụ nghiêm trọng", shared.num(counts["severe"]))
    columns[1].metric("✅ Bắt được", shared.num(counts["caught"]))
    columns[2].metric("❌ Bỏ sót", shared.num(counts["missed"]))
    columns[3].metric("🔔 Báo động nhầm", shared.num(counts["false_alarms"]))
    st.caption(
        f"Ở mức này, hệ thống phát {shared.num(counts['alarms'])} cảnh báo; khoảng "
        f"{shared.num(counts['caught'])} cảnh báo trùng với vụ nghiêm trọng. Cơ quan quản lý phải tự quyết định "
        "chi phí của một vụ bỏ sót so với một lần kiểm tra không cần thiết."
    )
    figure = px.line(
        tradeoff,
        x="recall",
        y="precision",
        labels={"recall": "Tỉ lệ vụ nghiêm trọng bắt được", "precision": "Tỉ lệ cảnh báo là đúng"},
    )
    figure.add_scatter(
        x=[row["recall"]],
        y=[row["precision"]],
        mode="markers",
        marker=dict(size=14, color=shared.SEVERE_COLOUR),
        name="Mức bạn chọn",
    )
    figure.update_layout(
        xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=330, margin=dict(t=10), showlegend=False
    )
    st.plotly_chart(figure, width="stretch")

    st.subheader("3. Dùng yếu tố ảnh hưởng để đặt câu hỏi, không để kết luận nguyên nhân")
    importance_data = importance.sort_values("importance").tail(12)
    labels = [vi.col(feature) for feature in importance_data["feature"]]
    figure = px.bar(
        x=importance_data["share"],
        y=labels,
        orientation="h",
        text=[shared.pct(value) for value in importance_data["share"]],
        labels={"x": "Mức độ ảnh hưởng trong dự đoán", "y": ""},
    )
    figure.update_traces(marker_color=shared.SAFE_COLOUR)
    figure.update_layout(xaxis_tickformat=".0%", height=420, margin=dict(t=10))
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Các yếu tố trên cho biết mô hình thường dựa vào thông tin nào. Chúng giúp đặt câu hỏi khảo sát, nhưng "
        "không tự chứng minh rằng một yếu tố là nguyên nhân gây thương tích nặng."
    )

    st.subheader("4. Muốn áp dụng tại Việt Nam cần làm gì?")
    st.markdown(
        "- Thu thập dữ liệu chuẩn hóa tại địa phương, gồm tốc độ, mũ bảo hiểm, dây an toàn, thời gian cấp cứu và "
        "đặc điểm điểm đen.\n"
        "- Thử nghiệm nhỏ tại một quận hoặc một nhóm tuyến đường; không triển khai ngay trên toàn thành phố.\n"
        "- Đánh giá riêng theo loại phương tiện, khu vực và nhóm người tham gia giao thông để tránh bỏ sót có hệ thống.\n"
        "- Để chuyên gia duyệt cảnh báo, ghi lại quyết định và theo dõi hiệu quả của biện pháp sau can thiệp."
    )
    st.warning(
        "**Phạm vi giáo dục:** dữ liệu của dự án đến từ Addis Ababa, không phải Việt Nam. Vì vậy các con số minh "
        "họa cách tư duy và thiết kế một hệ thống hỗ trợ quản lý; chúng chưa đủ để ra quyết định thật tại Việt Nam."
    )
    shared.source_note()
