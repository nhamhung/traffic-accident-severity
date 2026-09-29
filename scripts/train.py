"""Train the production pipeline on the full dataset and save it.

Equivalent to the notebook's modeling steps, without the notebook.

Default is `Random Forest (balanced)`, not the LightGBM/XGBoost that
would be the usual first choice — verified directly (see the model
sweep in the notebook/report) that class-weighting clearly beats every
other option on macro-F1 for this severely imbalanced target, including
fancier gradient-boosted models.
"""

import argparse

from traffic_accident_severity import config, data, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=list(model.MODEL_FACTORIES.keys()),
        default="Random Forest (balanced)",
        help="Which model family to train (default: Random Forest (balanced)).",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Insert ADASYN oversampling instead of relying on class weighting.",
    )
    args = parser.parse_args()

    df = data.load_accidents()
    X, y = data.split_features_target(df)
    estimator = model.MODEL_FACTORIES[args.model]()
    pipeline = model.train_pipeline(X, y, estimator=estimator, resample=args.resample)
    model.save_pipeline(pipeline)
    print(f"Trained {args.model} (resample={args.resample}) on {len(df)} rows. Saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
