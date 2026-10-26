"""Stage 3: train the model chosen in params.yaml.

- xgboost (main model): saved as models/model.json, XGBoost's native format,
  which is what the API loads.
- logreg (baseline): saved as models/model.joblib, used only for comparison
  in DVC experiments, never released.
"""

from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import FEATURES, TARGET

MODEL_DIR = Path("models")


def build_xgboost(params: dict, y: pd.Series) -> XGBClassifier:
    xgb = dict(params["xgboost"])
    if xgb["scale_pos_weight"] == "auto":
        # Up-weight the rare positive class so the model doesn't ignore it.
        xgb["scale_pos_weight"] = float((y == 0).sum() / (y == 1).sum())
    return XGBClassifier(
        **xgb,
        tree_method="hist",
        random_state=params["seed"],
        n_jobs=params["n_jobs"],  # fixed thread count -> deterministic results
        eval_metric="logloss",
    )


def build_logreg(params: dict) -> object:
    lr = params["logreg"]
    # class_weight="balanced" is LR's equivalent of scale_pos_weight.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=lr["C"],
            max_iter=lr["max_iter"],
            class_weight="balanced",
            random_state=params["seed"],
        ),
    )


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))["train"]
    train = pd.read_csv("data/processed/train.csv")
    X, y = train[FEATURES], train[TARGET]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if params["model"] == "xgboost":
        model = build_xgboost(params, y)
        model.fit(X, y)
        model.save_model(MODEL_DIR / "model.json")
    elif params["model"] == "logreg":
        model = build_logreg(params)
        model.fit(X, y)
        joblib.dump(model, MODEL_DIR / "model.joblib")
    else:
        raise ValueError(f"unknown model type: {params['model']}")
    print(f"trained {params['model']} on {len(train)} rows")


if __name__ == "__main__":
    main()
