"""Stage 1: validate the raw batches before anything is trained on them.

Critical failures (wrong columns, nulls, non-numeric or out-of-range values) exit
non-zero, which stops `dvc repro` so bad data can never reach a model.
Duplicates are only reported: in survey data identical answers are legitimate.
"""

import json
import sys
from pathlib import Path

import pandas as pd

from src.features import COLUMNS, FEATURE_RANGES, TARGET

RAW_DIR = Path("data/raw")
REPORT_PATH = Path("reports/validation.json")


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read every batch CSV in data/raw/ (sorted, so the order is stable)."""
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files found in {raw_dir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def validate(df: pd.DataFrame) -> dict:
    """Return a report dict; report["passed"] is False on any critical failure."""
    errors = []

    missing = [c for c in COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in COLUMNS]
    if missing:
        errors.append(f"missing columns: {missing}")
    if extra:
        errors.append(f"unexpected columns: {extra}")
    if len(df) == 0:
        errors.append("no rows")

    # Column-level checks only make sense if the schema is right.
    if not missing:
        nulls = {c: int(n) for c, n in df[COLUMNS].isna().sum().items() if n > 0}
        if nulls:
            errors.append(f"null values: {nulls}")

        non_numeric = [c for c in COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            errors.append(f"non-numeric columns: {non_numeric}")
        else:
            ranges = {TARGET: (0, 1), **FEATURE_RANGES}
            for col, (lo, hi) in ranges.items():
                values = df[col].dropna()
                bad_range = int(((values < lo) | (values > hi)).sum())
                if bad_range:
                    errors.append(f"{col}: {bad_range} values outside [{lo}, {hi}]")
                # Every column in this dataset is a code or a count, so it must be whole.
                bad_int = int((values % 1 != 0).sum())
                if bad_int:
                    errors.append(f"{col}: {bad_int} non-integer values")

    report = {
        "passed": not errors,
        "errors": errors,
        "rows": int(len(df)),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if TARGET in df.columns and len(df) and not errors:
        report["positive_rate"] = round(float(df[TARGET].mean()), 6)
    return report


def main() -> None:
    report = validate(load_raw())
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
