"""Stage 4: score the model on the held-out test set.

Writes metrics.json (tracked by DVC as metrics, committed to Git) and ROC/PR curve
points for `dvc plots`. Recall is the headline metric: for screening, missing a
person at risk costs more than a false alarm. Accuracy is deliberately not reported,
because always predicting "no diabetes" would already score ~86%.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

from src.features import FEATURES, TARGET

PLOTS_DIR = Path("plots")
MAX_PLOT_POINTS = 200  # keep the curve CSVs small enough to commit and diff


def load_model(model_type: str):
    if model_type == "xgboost":
        model = XGBClassifier()
        model.load_model("models/model.json")
        return model
    return joblib.load("models/model.joblib")


def thin(df: pd.DataFrame) -> pd.DataFrame:
    """Keep at most MAX_PLOT_POINTS evenly spaced rows (always keeps both ends)."""
    if len(df) <= MAX_PLOT_POINTS:
        return df
    idx = np.unique(np.linspace(0, len(df) - 1, MAX_PLOT_POINTS).round().astype(int))
    return df.iloc[idx]


def compute_metrics(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }
    # Round so tiny float noise doesn't show up as a metric change.
    return {k: round(float(v), 6) for k, v in metrics.items()}


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    test = pd.read_csv("data/processed/test.csv")
    y_true = test[TARGET]

    model = load_model(params["train"]["model"])
    y_prob = model.predict_proba(test[FEATURES])[:, 1]

    metrics = compute_metrics(y_true, y_prob, params["evaluate"]["threshold"])
    Path("metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    thin(pd.DataFrame({"fpr": fpr, "tpr": tpr})).round(6).to_csv(PLOTS_DIR / "roc.csv", index=False)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    thin(pd.DataFrame({"recall": recall, "precision": precision})).round(6).to_csv(
        PLOTS_DIR / "pr.csv", index=False
    )


if __name__ == "__main__":
    main()
