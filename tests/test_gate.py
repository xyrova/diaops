"""Metric gate tests: the pass/fail rules, without needing git or a trained model."""

from scripts.check_gate import check

GOOD = {"recall": 0.78, "roc_auc": 0.827}


def test_passes_when_metrics_hold():
    assert check(GOOD, GOOD, min_recall=0.70, max_auc_drop=0.01) == []


def test_fails_below_recall_floor():
    failures = check({"recall": 0.65, "roc_auc": 0.83}, GOOD, 0.70, 0.01)
    assert len(failures) == 1 and "recall" in failures[0]


def test_fails_when_auc_drops_too_much():
    failures = check({"recall": 0.78, "roc_auc": 0.80}, GOOD, 0.70, 0.01)
    assert len(failures) == 1 and "ROC-AUC" in failures[0]


def test_small_auc_drop_is_allowed():
    assert check({"recall": 0.78, "roc_auc": 0.820}, GOOD, 0.70, 0.01) == []


def test_auc_improvement_passes():
    assert check({"recall": 0.78, "roc_auc": 0.85}, GOOD, 0.70, 0.01) == []


def test_first_run_applies_only_recall_floor():
    # No baseline: a low AUC can't be compared, so only the floor matters.
    assert check({"recall": 0.75, "roc_auc": 0.50}, None, 0.70, 0.01) == []
    assert check({"recall": 0.60, "roc_auc": 0.90}, None, 0.70, 0.01) != []
