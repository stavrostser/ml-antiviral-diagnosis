"""Exploratory data analysis helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _unique_values(series: pd.Series) -> list[Any]:
    """Return unique values for a series, preserving missing values.

    Args:
        series: Input series to inspect.

    Returns:
        A list of unique values in first-seen order, including missing values
        such as ``NaN``, ``None``, or ``pd.NA``.
    """
    return pd.unique(series.astype("object")).tolist()


def summarize_unique_values(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize distinct values for every column in a DataFrame.

    This is intended for quick exploratory analysis when you want both the
    number of distinct values and the concrete values present in each column.
    Missing values are counted and included in the output.

    Args:
        df: DataFrame to inspect.

    Returns:
        A DataFrame with one row per input column and these fields:
        ``column`` for the column name, ``dtype`` for the pandas dtype,
        ``unique_count`` for the number of distinct values including missing
        values, and ``unique_values`` for the actual unique values.
    """
    rows: list[dict[str, Any]] = []

    for column_name in df.columns:
        series = df[column_name]
        unique_values = _unique_values(series)
        rows.append(
            {
                "column": column_name,
                "dtype": str(series.dtype),
                "unique_count": int(series.nunique(dropna=False)),
                "unique_values": unique_values,
            }
        )

    return pd.DataFrame(rows)


def count_unique_values_only_in_first(
    first_values: list[Any], second_values: list[Any], *, print_summary: bool = True
) -> dict[str, int | list[Any]]:
    """Return distinct values that appear only in the first list.

    Args:
        first_values: Hashable values to count from.
        second_values: Hashable values to compare against.
        print_summary: Whether to print a short summary of the result.

    Returns:
        A dictionary with ``count`` for the number of distinct values present
        in ``first_values`` but absent from ``second_values``, and ``values``
        for those distinct values in first-seen order.

    Raises:
        TypeError: If either list contains unhashable values.
    """
    difference = set(first_values) - set(second_values)
    only_in_first: list[Any] = []
    seen: set[Any] = set()

    for value in first_values:
        if value in difference and value not in seen:
            only_in_first.append(value)
            seen.add(value)

    result: dict[str, int | list[Any]] = {
        "count": len(only_in_first),
        "values": only_in_first,
    }

    if print_summary:
        print(
            "Unique values only in first list: "
            f"{result['count']} found -> {result['values']}"
        )

    return result
