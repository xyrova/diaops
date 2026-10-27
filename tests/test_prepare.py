"""Preprocessing tests: the split is sized right, stratified and reproducible."""

import pandas as pd
import pytest

from src.features import COLUMNS, TARGET
from src.prepare import split


@pytest.fixture
def df():
    return pd.read_csv("tests/sample/sample.csv")


def test_split_sizes(df):
    train, test = split(df, test_size=0.2, seed=42)
    assert len(train) + len(test) == len(df)
    assert len(test) == pytest.approx(0.2 * len(df), abs=1)


def test_split_is_stratified(df):
    train, test = split(df, test_size=0.2, seed=42)
    assert train[TARGET].mean() == pytest.approx(df[TARGET].mean(), abs=0.01)
    assert test[TARGET].mean() == pytest.approx(df[TARGET].mean(), abs=0.01)


def test_split_is_reproducible(df):
    a_train, a_test = split(df, test_size=0.2, seed=42)
    b_train, b_test = split(df, test_size=0.2, seed=42)
    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_test, b_test)


def test_output_columns_and_types(df):
    train, _ = split(df, test_size=0.2, seed=42)
    assert list(train.columns) == COLUMNS
    assert all(pd.api.types.is_integer_dtype(train[c]) for c in COLUMNS)
