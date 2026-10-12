"""Ba phát hiện trực quan và có ý nghĩa quản lý nhất."""

import pandas as pd
import plotly.express as px
import streamlit as st

from . import shared
from traffic_accident_severity import vi


def _rate_chart(rates: pd.DataFrame, column: str, base_rate: float, order: list | None = None, title: str = ""):
    values = rates[rates["column"] == column].copy()
    values = values[values["n"] >= 30]
    if order:
        values["value"] = pd.Categorical(values["value"], order, ordered=True)
        values = values.sort_values("value")
    else:
        values = values.sort_values("severe_rate")
    values["label"] = [str(value) if column == "hour" else vi.val(column, value) for value in values["value"]]
    values["Tỉ lệ nghiêm trọng"] = values["severe_rate"]
    horizontal = order is None
    figure = px.bar(
        values,
        x="Tỉ lệ nghiêm trọng" if horizontal else "label",
        y="label" if horizontal else "Tỉ lệ nghiêm trọng",
        orientation="h" if horizontal else "v",
        title=title,
        text=[shared.pct(value) for value in values["severe_rate"]],
        labels={"label": ""},
    )
    figure.update_traces(
        marker_color=[
            shared.SEVERE_COLOUR if value > base_rate * 1.25 else shared.SAFE_COLOUR
            for value in values["severe_rate"]
        ]
    )
    average_line = dict(
        line_dash="dash", line_color="gray", annotation_text=f"Mức trung bình {shared.pct(base_rate)}"
    )
    (figure.add_vline if horizontal else figure.add_hline)(base_rate, **average_line)
    figure.update_layout(
        height=320 if horizontal else 340,
        margin=dict(t=50, b=10),
        xaxis_tickformat=".0%" if horizontal else None,
        yaxis_tickformat=None if horizontal else ".0%",
    )
    return figure


def render() -> None:
    st.title("🔎 Những phát hiện thú vị")
    st.markdown(
        "Sau khi so sánh các nhóm tai nạn, em chọn ra ba kết quả dễ giải thích nhất cho lớp. "
        "Chúng không chỉ là con số: mỗi kết quả đều gợi ra một câu hỏi thực tế mà người quản lý có thể kiểm tra."
    )
    st.info(
        "**Cách đọc biểu đồ:** nếu một nhóm có tỉ lệ 20%, có thể hiểu đơn giản rằng cứ 100 vụ có hoàn cảnh tương tự "
        "thì khoảng 20 vụ là nghiêm trọng. Đường gạch đứt là mức trung bình để chúng ta dễ so sánh."
    )
    facts = shared.get_json("data_facts.json")
    rates = shared.get_result("rates_by.csv")
    base_rate = facts["severe_share"]

    st.subheader("1. Điều làm em chú ý nhất: đường tối nhưng không có đèn")
    st.plotly_chart(_rate_chart(rates, "Light_conditions", base_rate), width="stretch")
    st.markdown(
        "Nhóm *trời tối, không có đèn đường* nằm cao hơn rõ rệt. Điều thú vị là ban đêm **có đèn** lại gần với ban ngày "
        "hơn. Cách hiểu đơn giản: vấn đề đáng kiểm tra có thể không phải chỉ là trời tối, mà là **khả năng quan sát**."
    )

    st.subheader("2. Thời gian: từ 20 giờ đến 5 giờ nổi bật hơn ban ngày")
    hours = rates[rates["column"] == "hour"].copy()
    hours["hour"] = hours["value"].astype(float).astype(int)
    figure = px.line(
        hours.sort_values("hour"),
        x="hour",
        y="severe_rate",
        markers=True,
        labels={"hour": "Giờ trong ngày", "severe_rate": "Tỉ lệ nghiêm trọng"},
    )
    figure.add_hline(
        base_rate, line_dash="dash", line_color="gray", annotation_text=f"Mức trung bình {shared.pct(base_rate)}"
    )
    figure.update_layout(yaxis_tickformat=".0%", height=320, margin=dict(t=10))
    st.plotly_chart(figure, width="stretch")

    late_hours = (hours["hour"] >= 20) | (hours["hour"] <= 5)
    daytime = hours["hour"].between(6, 17)

    def weighted_rate(mask: pd.Series) -> float:
        selected = hours[mask]
        return float((selected["n"] * selected["severe_rate"]).sum() / selected["n"].sum())

    late_rate = weighted_rate(late_hours)
    daytime_rate = weighted_rate(daytime)
    comparison = st.columns(2)
    comparison[0].metric("🌙 Từ 20h đến 5h", shared.pct(late_rate))
    comparison[1].metric("☀️ Từ 6h đến 17h", shared.pct(daytime_rate))
    st.markdown(
        f"Khi gộp thành hai khoảng dễ so sánh, tỉ lệ nghiêm trọng từ **20h đến 5h** cao khoảng "
        f"**{shared.num(late_rate / daytime_rate, 1)} lần** so với từ 6h đến 17h. Kết quả này kể cùng một câu chuyện với biểu đồ "
        "ánh sáng ở trên: thời điểm, tầm nhìn và những điều kiện đi kèm có thể cùng làm rủi ro thay đổi."
    )

    st.subheader("3. Các nhóm tuổi không có mức rủi ro giống nhau")
    st.plotly_chart(
        _rate_chart(
            rates,
            "Age_band_of_driver",
            base_rate,
            order=["Under 18", "18-30", "31-50", "Over 51", "Unknown"],
            title="Tỉ lệ nghiêm trọng theo tuổi người lái",
        ),
        width="stretch",
    )
    st.markdown(
        "Trong dữ liệu này, nhóm rất trẻ và nhóm lớn tuổi có tỉ lệ nghiêm trọng cao hơn các nhóm ở giữa. Biểu đồ chưa "
        "cho biết *vì sao*, nhưng giúp chúng ta đặt câu hỏi tiếp theo về kinh nghiệm lái xe, khả năng phản ứng và hình thức hỗ trợ phù hợp."
    )

    st.info(
        "Một điểm rất quan trọng khi trình bày: **đi cùng nhau không có nghĩa là gây ra nhau**. Ví dụ, đường tối có thể "
        "đồng thời là đường vắng, xe chạy nhanh hoặc xa nơi cấp cứu. Các biểu đồ giúp đặt giả thuyết, chưa chứng minh nguyên nhân."
    )
    shared.source_note()
