"""Trang Mô hình giỏi đến đâu? So sánh mô hình, đánh đổi giữa bỏ sót và báo nhầm, và
những yếu tố mô hình dựa vào. Every number is precomputed by scripts/evaluate.py."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from . import shared
from traffic_accident_severity import vi


def _per_1000(row: pd.Series, base_rate: float) -> dict:
    """Out of 1,000 accidents, in whole numbers that add up: severe ones caught /
    missed, and false alarms."""
    severe = round(1000 * base_rate)
    caught = round(severe * row["recall"])
    alarms = round(caught / row["precision"]) if row["precision"] > 0 else 0
    return {"severe": severe, "caught": caught, "missed": severe - caught, "false_alarms": alarms - caught,
            "alarms": alarms}


def render() -> None:
    st.title("🧠 Mô hình giỏi đến đâu?")
    facts = shared.get_json("data_facts.json")
    summary = shared.get_json("summary.json")
    comparison = shared.get_result("model_comparison.csv")
    tradeoff = shared.get_result("threshold_tradeoff.csv")
    importance = shared.get_result("feature_importance.csv")
    base = facts["severe_share"]

    st.markdown(
        "Để chấm điểm công bằng, dữ liệu được chia làm 5 phần. Mô hình học trên 4 phần rồi \"làm bài thi\" trên "
        "phần còn lại (gồm những vụ tai nạn nó **chưa từng thấy**), lặp lại cho đủ 5 phần và làm 3 lần. "
        "Các vụ tai nạn được giữ nguyên, không bị cắt đôi giữa phần học và phần thi."
    )

    st.subheader("Vì sao không dùng \"độ chính xác\"?")
    st.markdown(
        f"Một mô hình **lười biếng** luôn trả lời \"không nghiêm trọng\" sẽ đúng **{shared.pct(1 - base)}** số lần, "
        "vì phần lớn tai nạn đều nhẹ! Nhưng nó vô dụng: không bao giờ phát hiện được vụ nghiêm trọng nào. "
        "Vì vậy dự án dùng các điểm số khác:"
    )
    shared.glossary("Macro-F1", (
        "Điểm từ 0 đến 1, đo xem mô hình giỏi **cả hai việc**: nhận ra vụ nghiêm trọng *và* nhận ra vụ không "
        "nghiêm trọng, rồi lấy trung bình. Mô hình lười biếng ở trên chỉ được khoảng **0,46**."
    ))
    shared.glossary("PR-AUC", (
        f"Điểm từ 0 đến 1, đo xem mô hình có **xếp các vụ nghiêm trọng lên trên** hay không. Đoán bừa sẽ được "
        f"khoảng **{shared.num(base, 2)}** (bằng tỉ lệ vụ nghiêm trọng). Điểm càng cao hơn mức đó, mô hình càng có ích."
    ))

    st.subheader("So sánh các mô hình")
    comp = comparison.sort_values("f1_tuned")
    fig = go.Figure()
    fig.add_bar(y=comp["name_vi"], x=comp["f1_tuned"], orientation="h", name="Macro-F1",
                marker_color=[shared.SEVERE_COLOUR if r else shared.GREY for r in comp["is_final"]],
                text=[shared.num(v, 3) for v in comp["f1_tuned"]], textposition="outside")
    fig.update_layout(height=60 * len(comp) + 80, xaxis_title="Macro-F1 (càng cao càng tốt)",
                      xaxis_range=[0.4, max(comp["f1_tuned"]) + 0.05], margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    table = comparison.sort_values("f1_tuned", ascending=False)[["name_vi", "f1_tuned", "pr_auc", "roc_auc"]]
    table.columns = ["Mô hình", "Macro-F1", "PR-AUC", "ROC-AUC"]
    st.dataframe(table.style.format({c: "{:.3f}" for c in ["Macro-F1", "PR-AUC", "ROC-AUC"]}),
                 hide_index=True, width="stretch")
    old = comparison[comparison["model"] == "old_random_forest"].iloc[0]
    new = comparison[comparison["is_final"]].iloc[0]
    st.markdown(
        f"Mô hình mới (**{new['name_vi']}**) đạt Macro-F1 **{shared.num(new['f1_tuned'], 3)}**, so với "
        f"**{shared.num(old['f1_tuned'], 3)}** của mô hình cũ; PR-AUC tăng từ **{shared.num(old['pr_auc'], 3)}** lên "
        f"**{shared.num(new['pr_auc'], 3)}** (gấp khoảng {shared.num(new['pr_auc'] / base, 1)} lần đoán bừa). Tiến bộ "
        "đến từ hai điều: dùng mô hình **hiểu trực tiếp các giá trị dạng chữ** (như \"ban đêm, không đèn\") và "
        "**chọn mức cảnh báo** phù hợp thay vì mặc định 50%."
    )

    st.subheader("Bắt nhiều hơn hay báo nhầm ít hơn?")
    st.markdown("Mô hình đưa ra **khả năng** (ví dụ 25%). Ta phải tự chọn: từ mức nào thì **bật cảnh báo**? "
                "Hãy kéo thanh trượt và xem điều gì xảy ra với **1.000 vụ tai nạn**.")
    thresholds = tradeoff["threshold"].round(2).tolist()
    default = min(thresholds, key=lambda t: abs(t - summary["threshold"]))
    t = st.select_slider("Bật cảnh báo khi khả năng nghiêm trọng từ…", thresholds, value=default,
                         format_func=lambda v: shared.pct(v))
    row = tradeoff.iloc[(tradeoff["threshold"] - t).abs().argmin()]
    n = _per_1000(row, base)
    c = st.columns(4)
    c[0].metric("Vụ nghiêm trọng (trên 1.000 vụ)", shared.num(n["severe"]))
    c[1].metric("✅ Bắt được", shared.num(n["caught"]))
    c[2].metric("❌ Bỏ sót", shared.num(n["missed"]))
    c[3].metric("🔔 Báo động nhầm", shared.num(n["false_alarms"]))
    st.caption(f"Ở mức này, cứ {shared.num(n['alarms'])} lần cảnh báo thì khoảng {shared.num(n['caught'])} lần là "
               "đúng. Mức thấp hơn: bắt được nhiều hơn nhưng báo nhầm nhiều hơn. Mức cao hơn: ngược lại. "
               "Chọn mức nào là **quyết định của con người** (bỏ sót một vụ nghiêm trọng tệ đến đâu so với một lần "
               "báo nhầm?), không phải của mô hình.")
    fig = px.line(tradeoff, x="recall", y="precision", labels={
        "recall": "Tỉ lệ vụ nghiêm trọng bắt được", "precision": "Tỉ lệ cảnh báo là đúng"})
    fig.add_scatter(x=[row["recall"]], y=[row["precision"]], mode="markers", marker=dict(size=14, color=shared.SEVERE_COLOUR),
                    name="Mức bạn chọn")
    fig.add_hline(base, line_dash="dash", line_color="gray", annotation_text="Đoán bừa")
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=330, margin=dict(t=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Mô hình dựa vào điều gì nhiều nhất?")
    imp = importance.sort_values("importance").tail(12)
    labels = [vi.col(f) for f in imp["feature"]]
    fig = px.bar(x=imp["share"], y=labels, orientation="h", text=[shared.pct(v) for v in imp["share"]],
                 labels={"x": "Mức độ ảnh hưởng (phần trăm tổng)", "y": ""})
    fig.update_traces(marker_color=shared.SAFE_COLOUR)
    fig.update_layout(xaxis_tickformat=".0%", height=420, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Tính bằng phương pháp SHAP: với mỗi vụ, đo xem từng thông tin đẩy dự đoán lên hay xuống bao nhiêu, "
               "rồi lấy trung bình trên toàn bộ dữ liệu.")

    st.info("**Giới hạn:** mô hình vẫn bỏ sót và báo nhầm khá nhiều. Lý do là mức độ nghiêm trọng còn phụ thuộc vào "
            "những điều dữ liệu không ghi lại: tốc độ thật của xe, có đội mũ bảo hiểm hay thắt dây an toàn không, "
            "xe cứu thương đến nhanh hay chậm… Không mô hình nào đoán đúng những gì nó không được biết.")
    shared.source_note()
