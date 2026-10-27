"""Data-validation tests: good data passes, each kind of bad data is caught."""

import pandas as pd
import pytest

from src.validate import validate

SAMPLE = "tests/sample/sample.csv"


@pytest.fixture
def df():
    return pd.read_csv(SAMPLE)


def test_sample_passes(df):
    report = validate(df)
    assert report["passed"], report["errors"]
    assert report["rows"] == len(df)


def test_missing_column_fails(df):
    report = validate(df.drop(columns=["Income"]))
    assert not report["passed"]
    assert any("missing columns" in e for e in report["errors"])


def test_extra_column_fails(df):
    report = validate(df.assign(Unexpected=0))
    assert not report["passed"]


@pytest.mark.parametrize(
    "column, bad_value",
    [("HighBP", 2), ("GenHlth", 0), ("Age", 14), ("Education", 7), ("Income", 9), ("BMI", 500)],
)
def test_out_of_range_fails(df, column, bad_value):
    df.loc[0, column] = bad_value
    report = validate(df)
    assert not report["passed"]
    assert any(column in e for e in report["errors"])


def test_null_fails(df):
    df.loc[0, "BMI"] = None
    assert not validate(df)["passed"]


def test_non_integer_fails(df):
    df.loc[0, "GenHlth"] = 2.5
    assert not validate(df)["passed"]


def test_target_must_be_binary(df):
    df.loc[0, "Diabetes_binary"] = 2
    assert not validate(df)["passed"]


def test_duplicates_reported_not_failed(df):
    doubled = pd.concat([df, df.head(10)], ignore_index=True)
    report = validate(doubled)
    assert report["passed"]
    assert report["duplicate_rows"] >= 10
