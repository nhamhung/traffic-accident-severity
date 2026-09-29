"""Model Insights page: model family comparison, class-imbalance
handling comparison, and SHAP — all computed live (this dataset is only
12,316 rows, so a full 5-fold sweep across every model finishes in well
under a minute, unlike the larger projects in this portfolio that
transcribe or subsample for the same reason).
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from . import shared
from traffic_accident_severity import model
from traffic_accident_severity.interpretability import top_shap_features

PALETTE = ["#2c5cc5", "#5b8def", "#8fb4f2", "#e07a5f", "#81b29a"]


@st.cache_data(show_spinner="Sweeping model families (first load only, ~30s)...")
def _model_sweep():
    from traffic_accident_severity import data

    df = shared.get_accidents_df()
    X, y = data.split_features_target(df)
    rows = []
    for name, factory in model.MODEL_FACTORIES.items():
        result = model.cross_validate_pipeline(X, y, estimator=factory(), cv=5)
        rows.append({"model": name, "f1_macro": result["f1_macro_mean"], "accuracy": result["accuracy_mean"]})
    return pd.DataFrame(rows).sort_values("f1_macro")


@st.cache_data(show_spinner="Comparing class-imbalance handling (first load only)...")
def _imbalance_comparison():
    from traffic_accident_severity import data

    df = shared.get_accidents_df()
    X, y = data.split_features_target(df)
    rows = [
        {"approach": "Plain Random Forest", **model.cross_validate_pipeline(X, y, estimator=model.random_forest_estimator(), cv=5)},
        {"approach": "Random Forest + ADASYN", **model.cross_validate_pipeline(X, y, estimator=model.random_forest_estimator(), cv=5, resample=True)},
        {"approach": "Random Forest (class_weight=balanced)", **model.cross_validate_pipeline(X, y, estimator=model.balanced_random_forest_estimator(), cv=5)},
    ]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Sweeping decision thresholds (first load only)...")
def _threshold_sweep():
    from traffic_accident_severity import data

    df = shared.get_accidents_df()
    X, y = data.split_features_target(df)
    proba = model.cross_val_severe_probabilities(X, y, cv=5)
    return model.severity_threshold_sweep(y, proba, target_recalls=[0.3, 0.5, 0.7, 0.9])


def render():
    st.title("🧠 Model Insights")
    st.caption(
        "Why macro-F1 (not accuracy) drives every comparison here, and "
        "what actually helps with this dataset's severe class imbalance."
    )

    try:
        shared.get_pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Model comparison (5-fold CV, macro-F1)")
    sweep = _model_sweep()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(sweep["model"], sweep["f1_macro"], color=PALETTE[0])
    ax.set_xlabel("Macro-F1")
    for bar, value in zip(bars, sweep["f1_macro"]):
        ax.text(value + 0.003, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=9)
    st.pyplot(fig)
    st.caption(
        "Notice accuracy (shown in the raw sweep data below) barely moves "
        "across models — it's dominated by the 84.6% majority class "
        "(\"Not Severe\"). Macro-F1 is what actually separates a model "
        "that's learning the minority class from one that isn't."
    )
    st.dataframe(sweep.sort_values("f1_macro", ascending=False), hide_index=True)

    st.subheader("Does class-imbalance handling help?")
    imbalance = _imbalance_comparison()
    fig2, ax2 = plt.subplots(figsize=(6, 3.2))
    bars2 = ax2.barh(imbalance["approach"], imbalance["f1_macro_mean"], color=[PALETTE[3], PALETTE[2], PALETTE[0]])
    ax2.set_xlabel("Macro-F1")
    for bar, value in zip(bars2, imbalance["f1_macro_mean"]):
        ax2.text(value + 0.003, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=9)
    st.pyplot(fig2)
    st.caption(
        "Class-weighting (reweighting the loss, no synthetic rows) clearly "
        "beats both plain training and ADASYN oversampling here. Neither "
        "result was assumed; both were measured directly."
    )

    st.subheader("Precision/recall tradeoff for catching Severe accidents")
    st.caption(
        "Since the model outputs a probability, the decision threshold is "
        "a choice, not a fixed rule — this shows what's actually available."
    )
    sweep = _threshold_sweep()
    fig_pr, ax_pr = plt.subplots(figsize=(6, 3.5))
    ax_pr.plot(sweep["recall"], sweep["precision"], marker="o", color=PALETTE[0])
    for _, row in sweep.iterrows():
        ax_pr.annotate(f"thr={row['threshold']:.2f}", (row["recall"], row["precision"]), textcoords="offset points", xytext=(6, 4))
    ax_pr.set_xlabel("Recall (share of Severe accidents caught)")
    ax_pr.set_ylabel("Precision (share of Severe alerts that are real)")
    st.pyplot(fig_pr)

    st.subheader("What does the saved model actually rely on? (SHAP)")
    st.caption(
        "Computed live from the saved model — first load takes a few "
        "seconds. Ranked by importance for the \"Severe\" class specifically."
    )
    explanation, _ = shared.get_shap_explanation(sample_size=500)
    top = top_shap_features(explanation, top_n=12, class_index=1).sort_values("mean_abs_shap")

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.barh(top["feature"], top["mean_abs_shap"], color=PALETTE[0])
    ax3.set_xlabel("Mean |SHAP value| for the Severe class")
    ax3.set_title("Top 12 features driving Severe predictions")
    st.pyplot(fig3)
