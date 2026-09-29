"""SHAP-based model interpretability, for the binary Severe/Not-Severe
target (see `model.py`'s module docstring for why the target is binary).

`shap.TreeExplainer` on this tree ensemble still returns one SHAP value
per class per feature (shape `(n_samples, n_features, 2)`), not a single
column — the two classes are mirror images of each other (`class_0 =
-class_1`), so `top_shap_features` defaults to averaging their absolute
values, but `class_index=1` ("Severe") is the more direct read when only
one class's explanation is wanted.
"""

import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def compute_shap_values(
    pipeline: Pipeline, X: pd.DataFrame, max_samples: int = 500, random_state: int = 42
) -> tuple[shap.Explanation, pd.DataFrame]:
    """Compute SHAP values for a fitted single-tree-model pipeline.

    `pipeline` must end in a plain tree-based classifier (LightGBM,
    Random Forest) — not the XGBoost wrapper class, whose `._model`
    would need unwrapping the same way academic_success does; this
    project's SHAP page always explains the LightGBM/Random Forest
    default, so no unwrapping is implemented here.
    """
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=random_state)

    preprocessing = pipeline[:-1]
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    transformed = preprocessing.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    X_transformed = pd.DataFrame(transformed, columns=feature_names, index=X.index)

    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    explanation = explainer(X_transformed)
    return explanation, X_transformed


def top_shap_features(
    explanation: shap.Explanation, top_n: int = 15, class_index: int | None = None
) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value.

    By default averages across all classes for this multiclass
    explanation. Pass `class_index` to instead rank by importance for one
    specific class only (e.g. the Fatal-injury class), since the
    all-class average can hide which features matter for a single rare
    class of particular interest.
    """
    values = explanation.values
    if values.ndim == 3:  # (n_samples, n_features, n_classes)
        importance = abs(values[:, :, class_index]).mean(axis=0) if class_index is not None else abs(values).mean(axis=(0, 2))
    else:
        importance = abs(values).mean(axis=0)

    return (
        pd.DataFrame({"feature": explanation.feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
