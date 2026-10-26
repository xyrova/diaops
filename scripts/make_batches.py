"""Split the full Kaggle CSV into three batches that simulate data arriving over time.

batch_1 = 60%, batch_2 = 20%, batch_3 = 20%, stratified on the target with a fixed
seed so everyone who runs this gets byte-identical batches.

This is a one-off data-arrival simulation, not a pipeline stage: the batches land in
data/incoming/ (gitignored), and you copy one into data/raw/ and `dvc add` it to make
it "arrive".

Usage:
    python scripts/make_batches.py [path/to/diabetes_binary_health_indicators_BRFSS2015.csv]
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
DEFAULT_SOURCE = Path("data/diabetes_binary_health_indicators_BRFSS2015.csv")
OUT_DIR = Path("data/incoming")
TARGET = "Diabetes_binary"


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    df = pd.read_csv(source)

    # 60% -> batch_1, remaining 40% split in half -> batch_2 and batch_3.
    batch_1, rest = train_test_split(df, train_size=0.6, stratify=df[TARGET], random_state=SEED)
    batch_2, batch_3 = train_test_split(
        rest, train_size=0.5, stratify=rest[TARGET], random_state=SEED
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, batch in [("batch_1", batch_1), ("batch_2", batch_2), ("batch_3", batch_3)]:
        path = OUT_DIR / f"{name}.csv"
        batch.to_csv(path, index=False)
        print(f"{path}: {len(batch)} rows, {batch[TARGET].mean():.2%} positive")


if __name__ == "__main__":
    main()
