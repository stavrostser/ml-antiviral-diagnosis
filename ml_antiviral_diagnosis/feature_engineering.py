"""Feature engineering helpers for model-table enrichment."""

from __future__ import annotations

import ast
from datetime import date
from typing import Any

import pandas as pd

from ml_antiviral_diagnosis.de import (
    DISEASE_X_DESCRIPTION,
    TransactionType,
    _find_first_transaction_by_description_on_date,
    _get_transactions_for_type,
    _normalize_transaction_text,
    _parse_transaction_date,
)


def _validate_model_table_columns(df: pd.DataFrame) -> None:
    """Validate that the model table contains the required columns.

    Args:
        df: Model table DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = {"PATIENT_ID", "DISEASEX_DT"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"model_table is missing required columns: {missing_text}")


def _validate_cleanup_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> None:
    """Validate model-table columns used during cleanup.

    Args:
        df: Model table DataFrame.
        columns: Columns that will be cleaned.

    Raises:
        ValueError: If any cleanup columns are missing.
    """
    missing_columns = set(columns) - set(df.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"model_table is missing cleanup columns: {missing_text}")


def _validate_diagnosis_feature_columns(df: pd.DataFrame) -> None:
    """Validate diagnosis dataset columns required for feature engineering.

    Args:
        df: Diagnosis dataset DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = {"patient_id", "transactions_by_type"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"diagnosis dataset is missing required columns: {missing_text}")


def _parse_transactions_by_type(value: Any) -> dict[str, list[dict[str, Any]]]:
    """Parse the nested transaction payload from in-memory or CSV form.

    Args:
        value: Either a nested transaction dictionary or its string form.

    Returns:
        A dictionary keyed by transaction type.

    Raises:
        ValueError: If the payload cannot be parsed into the expected shape.
    """
    if isinstance(value, dict):
        return {str(key): list(events) for key, events in value.items()}

    if isinstance(value, str):
        parsed_value = ast.literal_eval(value)
        if isinstance(parsed_value, dict):
            return {str(key): list(events) for key, events in parsed_value.items()}

    raise ValueError("transactions_by_type must be a dictionary or serialized dictionary")


def _find_latest_transaction_on_or_before_date(
    transactions: list[dict[str, Any]],
    cutoff_date: date,
) -> dict[str, Any] | None:
    """Find the closest transaction on or before a given date.

    Args:
        transactions: Transaction dictionaries from a single category.
        cutoff_date: Inclusive maximum transaction date.

    Returns:
        The closest matching transaction, or ``None`` when none qualify.
    """
    closest_transaction: dict[str, Any] | None = None
    closest_date: date | None = None

    for transaction in transactions:
        transaction_date = _parse_transaction_date(transaction["txn_dt"])
        if transaction_date > cutoff_date:
            continue

        if closest_date is None or transaction_date > closest_date:
            closest_transaction = transaction
            closest_date = transaction_date

    return closest_transaction


def _normalize_contraindication_value(value: Any) -> str:
    """Map raw contraindication descriptions to model categories.

    Args:
        value: Raw contraindication description.

    Returns:
        One of ``Low``, ``Medium``, ``High``, or ``Unspecified``.
    """
    normalized_value = _normalize_transaction_text(value)

    if normalized_value == "LOW CONTRAINDICATION":
        return "Low"
    if normalized_value == "MEDIUM CONTRAINDICATION":
        return "Medium"
    if normalized_value == "HIGH CONTRAINDICATION":
        return "High"

    return "Unspecified"


def add_model_table_transaction_features(
    model_table_df: pd.DataFrame,
    diagnosis_dataset_df: pd.DataFrame,
    diagnosis_description: str = DISEASE_X_DESCRIPTION,
) -> pd.DataFrame:
    """Add diagnosis-aligned transaction features to the model table.

    This helper enriches the existing model table with two features derived from
    the diagnosis-aligned nested transaction history:

    - ``INSURANCE_TYPE`` from the Disease X diagnosis transaction used by the
      model table row.
    - ``CONTRAINDICATIONS`` from the closest contraindication transaction on or
      before the model table's diagnosis date, normalized to ``Low``,
      ``Medium``, ``High``, or ``Unspecified``.

    Args:
        model_table_df: Existing model table.
        diagnosis_dataset_df: Diagnosis-aligned patient dataset.
        diagnosis_description: Description used to identify Disease X.

    Returns:
        A copy of ``model_table_df`` with ``INSURANCE_TYPE`` and
        ``CONTRAINDICATIONS`` appended.

    Raises:
        ValueError: If required columns are missing.
    """
    _validate_model_table_columns(model_table_df)
    _validate_diagnosis_feature_columns(diagnosis_dataset_df)

    diagnosis_lookup = {
        int(row.patient_id): _parse_transactions_by_type(row.transactions_by_type)
        for row in diagnosis_dataset_df.itertuples(index=False)
    }

    enriched_df = model_table_df.copy()
    insurance_types: list[str | None] = []
    contraindication_values: list[str] = []

    for row in enriched_df.itertuples(index=False):
        patient_id = int(row.PATIENT_ID)
        diagnosis_date = _parse_transaction_date(row.DISEASEX_DT)
        transactions_by_type = diagnosis_lookup.get(patient_id)

        diagnosis_event = None
        latest_contraindication = None
        if transactions_by_type is not None:
            diagnosis_event = _find_first_transaction_by_description_on_date(
                transactions=_get_transactions_for_type(
                    transactions_by_type, TransactionType.CONDITIONS
                ),
                description=diagnosis_description,
                event_date=diagnosis_date,
            )
            latest_contraindication = _find_latest_transaction_on_or_before_date(
                transactions=_get_transactions_for_type(
                    transactions_by_type, TransactionType.CONTRAINDICATIONS
                ),
                cutoff_date=diagnosis_date,
            )

        insurance_types.append(
            None if diagnosis_event is None else diagnosis_event.get("insurance_type")
        )
        contraindication_values.append(
            "Unspecified"
            if latest_contraindication is None
            else _normalize_contraindication_value(latest_contraindication.get("txn_desc", ""))
        )

    enriched_df["INSURANCE_TYPE"] = insurance_types
    enriched_df["CONTRAINDICATIONS"] = contraindication_values
    return enriched_df


def clean_model_table_categorical_nulls(
    model_table_df: pd.DataFrame,
    columns: list[str] | None = None,
    fill_value: str = "UNSPECIFIED",
) -> pd.DataFrame:
    """Fill missing categorical values in the model table.

    Args:
        model_table_df: Model table to clean.
        columns: Categorical columns to fill. Defaults to physician columns.
        fill_value: Replacement value for missing entries.

    Returns:
        A copy of the model table with missing values filled.

    Raises:
        ValueError: If any requested cleanup columns are missing.
    """
    cleanup_columns = ["PHYSICIAN_TYPE", "PHYSICIAN_STATE"] if columns is None else columns
    _validate_cleanup_columns(model_table_df, cleanup_columns)

    cleaned_df = model_table_df.copy()
    for column in cleanup_columns:
        cleaned_df[column] = cleaned_df[column].fillna(fill_value)

    return cleaned_df
