"""Tests for EDA helpers."""

from __future__ import annotations

import math

import pandas as pd

from ml_antiviral_diagnosis.eda import (
    count_unique_values_only_in_first,
    summarize_unique_values,
)


def test_summarize_unique_values_includes_empty_and_nan() -> None:
    """It includes empty strings and NaN values in the summary."""
    df = pd.DataFrame(
        {
            "status": ["new", "", "new", None],
            "score": [1.0, float("nan"), 1.0, 2.0],
        }
    )

    result = summarize_unique_values(df).set_index("column")

    assert result.loc["status", "unique_count"] == 3
    status_values = result.loc["status", "unique_values"]
    assert status_values[0] == "new"
    assert status_values[1] == ""
    assert math.isnan(status_values[2])

    score_values = result.loc["score", "unique_values"]
    assert result.loc["score", "unique_count"] == 3
    assert score_values[0] == 1.0
    assert math.isnan(score_values[1])
    assert score_values[2] == 2.0


def test_summarize_unique_values_preserves_column_order() -> None:
    """It returns rows in the same order as the input columns."""
    df = pd.DataFrame(
        {
            "b": [1, 2],
            "a": ["x", "y"],
        }
    )

    result = summarize_unique_values(df)

    assert result["column"].tolist() == ["b", "a"]
    assert result["unique_count"].tolist() == [2, 2]


def test_count_unique_values_only_in_first_counts_distinct_difference() -> None:
    """It counts distinct values found only in the first list."""
    result = count_unique_values_only_in_first(
        first_values=[1, 2, 2, 3, 4],
        second_values=[2, 4, 5],
        print_summary=False,
    )

    assert result == {"count": 2, "values": [1, 3]}


def test_count_unique_values_only_in_first_returns_zero_for_same_unique_values() -> None:
    """It ignores duplicates when both lists contain the same unique values."""
    result = count_unique_values_only_in_first(
        first_values=[1, 1, 2, 2],
        second_values=[2, 1, 1],
        print_summary=False,
    )

    assert result == {"count": 0, "values": []}


def test_count_unique_values_only_in_first_prints_summary(capsys: object) -> None:
    """It prints a short summary when summary printing is enabled."""
    result = count_unique_values_only_in_first(
        first_values=[10, 20, 30],
        second_values=[20],
    )

    captured = capsys.readouterr()

    assert result == {"count": 2, "values": [10, 30]}
    assert "Unique values only in first list: 2 found -> [10, 30]" in captured.out
