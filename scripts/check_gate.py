"""Metric gate used by train.yml: decide whether a newly trained model may be kept.

Compares the new metrics.json (just written by `dvc repro`) with the metrics.json
committed at HEAD, i.e. the currently accepted model:
  - fail if recall < gate.min_recall                     (absolute floor)
  - fail if ROC-AUC dropped by more than gate.max_auc_drop vs HEAD
If HEAD has no metrics.json yet (first ever run), only the recall floor applies.

Exits non-zero on failure, so the workflow stops before `dvc push` and the commit.
"""

import json
import subprocess
import sys
from pathlib import Path

import yaml


def baseline_metrics() -> dict | None:
    """metrics.json as committed at HEAD, or None if it isn't there yet."""
    result = subprocess.run(
        ["git", "show", "HEAD:metrics.json"], capture_output=True, text=True, check=False
    )
    return json.loads(result.stdout) if result.returncode == 0 else None


def check(new: dict, baseline: dict | None, min_recall: float, max_auc_drop: float) -> list[str]:
    """Return the list of failure reasons (empty list = gate passed)."""
    failures = []
    if new["recall"] < min_recall:
        failures.append(f"recall {new['recall']:.4f} is below the floor {min_recall}")
    if baseline is not None:
        drop = baseline["roc_auc"] - new["roc_auc"]
        if drop > max_auc_drop:
            failures.append(
                f"ROC-AUC dropped by {drop:.4f} ({baseline['roc_auc']:.4f} -> "
                f"{new['roc_auc']:.4f}), more than the allowed {max_auc_drop}"
            )
    return failures


def main() -> None:
    gate = yaml.safe_load(open("params.yaml"))["gate"]
    new = json.loads(Path("metrics.json").read_text())
    baseline = baseline_metrics()

    print(f"{'metric':<10} {'HEAD':>8} {'new':>8}")
    for name, value in new.items():
        old = f"{baseline[name]:.4f}" if baseline and name in baseline else "-"
        print(f"{name:<10} {old:>8} {value:>8.4f}")
    if baseline is None:
        print("No metrics.json at HEAD: applying only the recall floor.")

    failures = check(new, baseline, gate["min_recall"], gate["max_auc_drop"])
    if failures:
        print("\nGATE FAILED:")
        for reason in failures:
            print(f"  - {reason}")
        sys.exit(1)
    print("\nGate passed.")


if __name__ == "__main__":
    main()
