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
        "Ba kết quả dưới đây nổi bật vì **dễ hiểu, có thể quan sát ngoài thực tế và gợi ý được hành động quản lý**. "
        "Thanh đỏ cho thấy nhóm có tỉ lệ tai nạn nghiêm trọng cao hơn rõ rệt."
    )
    facts = shared.get_json("data_facts.json")
    rates = shared.get_result("rates_by.csv")
    base_rate = facts["severe_share"]

    st.subheader("1. Thiếu đèn đường là tín hiệu nguy hiểm rõ nhất")
    st.plotly_chart(_rate_chart(rates, "Light_conditions", base_rate), width="stretch")
    st.markdown(
        "Tai nạn trong bóng tối **không có đèn đường** nghiêm trọng hơn rõ rệt. Trong khi đó, ban đêm có chiếu sáng "
        "tốt gần với ban ngày hơn. Điều này gợi ý một hướng hành động rất cụ thể: khảo sát và ưu tiên các đoạn đường tối."
    )

    st.subheader("2. Ban đêm và rạng sáng cần được chú ý hơn")
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
    st.markdown(
        "Kết quả theo giờ củng cố phát hiện về ánh sáng. Đây là cơ sở để cân nhắc thời điểm tuần tra, cảnh báo và "
        "bố trí lực lượng phản ứng."
    )

    st.subheader("3. Nhóm tuổi ở hai đầu có nguy cơ cao hơn")
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
        "Người lái rất trẻ và người lái lớn tuổi có tỉ lệ tai nạn nghiêm trọng cao hơn nhóm ở giữa. Phát hiện này "
        "gợi ý các chương trình giáo dục, hỗ trợ và thiết kế an toàn phù hợp cho từng nhóm."
    )

    st.info(
        "Các biểu đồ cho thấy **mối liên hệ**, không khẳng định một yếu tố duy nhất là nguyên nhân. "
        "Chúng nên được dùng để chọn nơi cần khảo sát kỹ hơn."
    )
    shared.source_note()
