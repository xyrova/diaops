"""Stage 2: stratified train/test split.

Stratifying keeps the ~14% positive rate the same in both splits, so test metrics
reflect the real class balance. The seed comes from params.yaml.
"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.features import COLUMNS, TARGET
from src.validate import load_raw

OUT_DIR = Path("data/processed")


def split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Validation already proved every value is a whole number, so store ints.
    df = df[COLUMNS].astype(int)
    return train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))["prepare"]
    train, test = split(load_raw(), params["test_size"], params["seed"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    print(f"train: {len(train)} rows, test: {len(test)} rows")


if __name__ == "__main__":
    main()
