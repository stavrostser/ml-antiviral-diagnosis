"""Data engineering helpers for patient-level feature preparation."""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

DISEASE_X_DESCRIPTION = "Disease X"
DRUG_A_DESCRIPTION = "Drug A"


class TransactionType(StrEnum):
    """Allowed transaction categories in the fact transaction table."""

    SYMPTOMS = "SYMPTOMS"
    CONDITIONS = "CONDITIONS"
    CONTRAINDICATIONS = "CONTRAINDICATIONS"
    TREATMENTS = "TREATMENTS"


class PatientTransactionEvent(BaseModel):
    """Validated representation of a single patient transaction."""

    model_config = ConfigDict(use_enum_values=True)

    txn_dt: date
    physician_id: int | None
    txn_location_type: str
    insurance_type: str
    txn_desc: str
    txn_type: TransactionType


def _normalize_transaction_text(value: Any) -> str:
    """Normalize transaction text for robust matching.

    Args:
        value: Raw transaction text.

    Returns:
        Upper-cased text with punctuation replaced by spaces and repeated
        whitespace collapsed.
    """
    text = str(value).strip().upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_patient_transactions_columns(df: pd.DataFrame) -> None:
    """Validate the patient-level transaction DataFrame schema.

    Args:
        df: Patient-level transaction DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = {"patient_id", "transactions_by_type"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError("patient transactions are missing required columns: " f"{missing_text}")


def _validate_diagnosis_dataset_columns(df: pd.DataFrame) -> None:
    """Validate the diagnosis-aligned patient DataFrame schema.

    Args:
        df: Diagnosis-aligned patient DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = {"patient_id", "first_diagnosis_date", "transactions_by_type", "TARGET"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"diagnosis dataset is missing required columns: {missing_text}")


def _validate_dim_patient_columns(df: pd.DataFrame) -> None:
    """Validate the patient dimension schema.

    Args:
        df: Patient dimension DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = {"PATIENT_ID", "BIRTH_YEAR", "GENDER"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"dim_patient is missing required columns: {missing_text}")


def _validate_dim_physician_columns(df: pd.DataFrame) -> None:
    """Validate the physician dimension schema.

    Args:
        df: Physician dimension DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = {"PHYSICIAN_ID", "STATE", "PHYSICIAN_TYPE"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"dim_physician is missing required columns: {missing_text}")


def _parse_transaction_date(value: Any) -> date:
    """Convert a transaction date value to a ``date`` instance.

    Args:
        value: Raw transaction date value.

    Returns:
        Parsed calendar date.
    """
    return pd.to_datetime(value).date()


def _get_transactions_for_type(
    transactions_by_type: dict[str, list[dict[str, Any]]],
    transaction_type: TransactionType,
) -> list[dict[str, Any]]:
    """Return all events for a transaction type from a nested mapping.

    Args:
        transactions_by_type: Patient transaction dictionary.
        transaction_type: Transaction category to retrieve.

    Returns:
        The list of transaction dictionaries for the requested type.
    """
    return list(transactions_by_type.get(transaction_type.value, []))


def _find_first_transaction_date_by_description(
    transactions: list[dict[str, Any]],
    description: str,
) -> date | None:
    """Find the earliest transaction date for a normalized description.

    Args:
        transactions: Transaction dictionaries from a single category.
        description: Description to match after normalization.

    Returns:
        The earliest matching transaction date, or ``None`` if not found.
    """
    normalized_description = _normalize_transaction_text(description)
    matching_dates = [
        _parse_transaction_date(transaction["txn_dt"])
        for transaction in transactions
        if _normalize_transaction_text(transaction.get("txn_desc", "")) == normalized_description
    ]

    if not matching_dates:
        return None

    return min(matching_dates)


def _find_first_transaction_by_description_on_date(
    transactions: list[dict[str, Any]],
    description: str,
    event_date: date,
) -> dict[str, Any] | None:
    """Find the first matching transaction on a specific date.

    Args:
        transactions: Transaction dictionaries from a single category.
        description: Description to match after normalization.
        event_date: Required transaction date.

    Returns:
        The first matching transaction dictionary, or ``None`` if not found.
    """
    normalized_description = _normalize_transaction_text(description)

    for transaction in transactions:
        if (
            _normalize_transaction_text(transaction.get("txn_desc", "")) == normalized_description
            and _parse_transaction_date(transaction["txn_dt"]) == event_date
        ):
            return transaction

    return None


def _filter_transactions_on_or_before_date(
    transactions_by_type: dict[str, list[dict[str, Any]]],
    cutoff_date: date | None,
) -> dict[str, list[dict[str, Any]]]:
    """Keep only transactions occurring on or before the cutoff date.

    Args:
        transactions_by_type: Patient transaction dictionary.
        cutoff_date: Inclusive cutoff date.

    Returns:
        A transaction dictionary with only on-or-before events. If the cutoff is
        missing, all categories are returned with empty lists.
    """
    filtered_transactions: dict[str, list[dict[str, Any]]] = {
        transaction_type.value: [] for transaction_type in TransactionType
    }

    if cutoff_date is None:
        return filtered_transactions

    for transaction_type in TransactionType:
        filtered_transactions[transaction_type.value] = [
            transaction
            for transaction in _get_transactions_for_type(transactions_by_type, transaction_type)
            if _parse_transaction_date(transaction["txn_dt"]) <= cutoff_date
        ]

    return filtered_transactions


def _has_transaction_on_or_after_date_by_description(
    transactions: list[dict[str, Any]],
    description: str,
    minimum_date: date | None,
) -> bool:
    """Check whether a normalized transaction occurs on or after a date.

    Args:
        transactions: Transaction dictionaries from a single category.
        description: Description to match after normalization.
        minimum_date: Inclusive minimum date.

    Returns:
        ``True`` when a matching transaction exists on or after ``minimum_date``.
    """
    if minimum_date is None:
        return False

    normalized_description = _normalize_transaction_text(description)
    return any(
        _normalize_transaction_text(transaction.get("txn_desc", "")) == normalized_description
        and _parse_transaction_date(transaction["txn_dt"]) >= minimum_date
        for transaction in transactions
    )


def _normalize_physician_id(value: Any) -> int | None:
    """Normalize physician identifiers from the transaction table.

    Args:
        value: Raw physician identifier value.

    Returns:
        An integer physician identifier, or ``None`` when the value is missing.
    """
    if pd.isna(value):
        return None

    return int(value)


def _normalize_patient_gender(value: Any) -> str | None:
    """Normalize patient gender values to the model-table categories.

    Args:
        value: Raw patient gender value.

    Returns:
        ``M`` or ``F`` when available, otherwise ``None``.
    """
    if pd.isna(value):
        return None

    normalized_value = str(value).strip().upper()
    if normalized_value in {"M", "F"}:
        return normalized_value

    return None


def _calculate_patient_age_at_diagnosis(
    diagnosis_date: date,
    birth_year: Any,
) -> int | None:
    """Estimate patient age at diagnosis from birth year.

    Args:
        diagnosis_date: Disease X diagnosis date.
        birth_year: Patient birth year.

    Returns:
        Age in years, or ``None`` when birth year is missing.
    """
    if pd.isna(birth_year):
        return None

    return int(diagnosis_date.year - int(birth_year))


def _count_high_risk_conditions(
    condition_transactions: list[dict[str, Any]],
    diagnosis_description: str,
) -> int:
    """Count distinct non-diagnosis conditions present by diagnosis date.

    Args:
        condition_transactions: Filtered ``CONDITIONS`` transactions.
        diagnosis_description: Disease X diagnosis description.

    Returns:
        The number of distinct condition descriptions excluding Disease X.
    """
    normalized_diagnosis_description = _normalize_transaction_text(diagnosis_description)
    # Set comprehension to find distinct normalized condition descriptions excluding the diagnosis description
    distinct_conditions = {
        _normalize_transaction_text(transaction.get("txn_desc", ""))
        for transaction in condition_transactions
        if _normalize_transaction_text(transaction.get("txn_desc", ""))
        != normalized_diagnosis_description
    }
    distinct_conditions.discard("")
    return len(distinct_conditions)


def transform_fact_txn_to_patient_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Transform fact transactions into a patient-level nested structure.

    The output has one row per patient and a ``transactions_by_type`` mapping
    with the allowed transaction categories as keys. Each key contains a list
    of validated transaction event dictionaries for that patient.

    Args:
        df: The ``fact_txn`` DataFrame.

    Returns:
        A DataFrame with one row per patient and these columns:
        ``patient_id`` for the patient identifier and
        ``transactions_by_type`` for the nested transaction dictionary.

    Raises:
        ValueError: If required columns are missing or transaction types are
        outside the allowed set.
    """
    required_columns = {
        "TXN_DT",
        "PATIENT_ID",
        "PHYSICIAN_ID",
        "TXN_LOCATION_TYPE",
        "INSURANCE_TYPE",
        "TXN_TYPE",
        "TXN_DESC",
    }
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"fact_txn is missing required columns: {missing_text}")

    patient_rows: list[dict[str, Any]] = []

    for patient_id, patient_df in df.sort_values(["PATIENT_ID", "TXN_DT"]).groupby(
        "PATIENT_ID", sort=True
    ):
        transactions_by_type: dict[str, list[dict[str, Any]]] = {
            transaction_type.value: [] for transaction_type in TransactionType
        }

        for row in patient_df.itertuples(index=False):
            event = PatientTransactionEvent(
                txn_dt=pd.to_datetime(row.TXN_DT).date(),
                physician_id=_normalize_physician_id(row.PHYSICIAN_ID),
                txn_location_type=str(row.TXN_LOCATION_TYPE),
                insurance_type=str(row.INSURANCE_TYPE),
                txn_desc=str(row.TXN_DESC),
                txn_type=TransactionType(str(row.TXN_TYPE)),
            )
            event_payload = event.model_dump(mode="json")
            txn_type = event_payload.pop("txn_type")
            transactions_by_type[txn_type].append(event_payload)

        patient_rows.append(
            {
                "patient_id": int(patient_id),
                "transactions_by_type": transactions_by_type,
            }
        )

    return pd.DataFrame(patient_rows)


def build_patient_diagnosis_dataset(
    patient_transactions_df: pd.DataFrame,
    diagnosis_description: str = DISEASE_X_DESCRIPTION,
    treatment_description: str = DRUG_A_DESCRIPTION,
) -> pd.DataFrame:
    """Build a diagnosis-aligned patient dataset from nested transactions.

    For each patient, this helper finds the first diagnosis date using
    ``CONDITIONS`` transactions whose cleaned description matches
    ``diagnosis_description``. It then keeps only transactions occurring on or
    before that diagnosis date and assigns ``TARGET = 1`` when a cleaned
    ``TREATMENTS`` transaction matching ``treatment_description`` exists on or
    after the first diagnosis date.

    Args:
        patient_transactions_df: Output of
            ``transform_fact_txn_to_patient_transactions``.
        diagnosis_description: Description used to identify Disease X diagnosis.
        treatment_description: Description used to identify Drug A treatment.

    Returns:
        A patient-level DataFrame with diagnosed patients only and columns
        ``patient_id``, ``first_diagnosis_date``, ``transactions_by_type``,
        and ``TARGET``.

    Raises:
        ValueError: If the patient transaction input schema is invalid.
    """
    _normalize_patient_transactions_columns(patient_transactions_df)

    diagnosis_rows: list[dict[str, Any]] = []

    for row in patient_transactions_df.itertuples(index=False):
        transactions_by_type = dict(row.transactions_by_type)
        first_diagnosis_date = _find_first_transaction_date_by_description(
            transactions=_get_transactions_for_type(
                transactions_by_type, TransactionType.CONDITIONS
            ),
            description=diagnosis_description,
        )

        # Skip patients that do not have a Disease X diagnosis event.
        if first_diagnosis_date is None:
            continue

        filtered_transactions = _filter_transactions_on_or_before_date(
            transactions_by_type=transactions_by_type,
            cutoff_date=first_diagnosis_date,
        )
        target = int(
            _has_transaction_on_or_after_date_by_description(
                transactions=_get_transactions_for_type(
                    transactions_by_type, TransactionType.TREATMENTS
                ),
                description=treatment_description,
                minimum_date=first_diagnosis_date,
            )
        )

        diagnosis_rows.append(
            {
                "patient_id": int(row.patient_id),
                "first_diagnosis_date": first_diagnosis_date,
                "transactions_by_type": filtered_transactions,
                "TARGET": target,
            }
        )

    return pd.DataFrame(diagnosis_rows)


def build_model_table(
    diagnosis_dataset_df: pd.DataFrame,
    dim_patient_df: pd.DataFrame,
    dim_physician_df: pd.DataFrame,
    diagnosis_description: str = DISEASE_X_DESCRIPTION,
) -> pd.DataFrame:
    """Build the core model table columns from diagnosis and dimension data.

    Args:
        diagnosis_dataset_df: Diagnosis-aligned patient dataset.
        dim_patient_df: Patient dimension table.
        dim_physician_df: Physician dimension table.
        diagnosis_description: Description used to identify Disease X diagnosis.

    Returns:
        A DataFrame with the current model-table columns in the expected order.

    Raises:
        ValueError: If any input DataFrame is missing required columns.
    """
    _validate_diagnosis_dataset_columns(diagnosis_dataset_df)
    _validate_dim_patient_columns(dim_patient_df)
    _validate_dim_physician_columns(dim_physician_df)

    patient_lookup = dim_patient_df.set_index("PATIENT_ID")[["BIRTH_YEAR", "GENDER"]]
    physician_lookup = dim_physician_df.set_index("PHYSICIAN_ID")[["STATE", "PHYSICIAN_TYPE"]]

    model_rows: list[dict[str, Any]] = []

    for row in diagnosis_dataset_df.itertuples(index=False):
        patient_id = int(row.patient_id)
        diagnosis_date = row.first_diagnosis_date
        transactions_by_type = dict(row.transactions_by_type)
        diagnosis_event = _find_first_transaction_by_description_on_date(
            transactions=_get_transactions_for_type(transactions_by_type, TransactionType.CONDITIONS),
            description=diagnosis_description,
            event_date=diagnosis_date,
        )

        patient_birth_year = None
        patient_gender = None
        if patient_id in patient_lookup.index:
            patient_birth_year = patient_lookup.at[patient_id, "BIRTH_YEAR"]
            patient_gender = _normalize_patient_gender(patient_lookup.at[patient_id, "GENDER"])

        physician_state = None
        physician_type = None
        diagnosis_physician_id = None if diagnosis_event is None else diagnosis_event.get("physician_id")
        if diagnosis_physician_id is not None and diagnosis_physician_id in physician_lookup.index:
            physician_state = physician_lookup.at[diagnosis_physician_id, "STATE"]
            physician_type = physician_lookup.at[diagnosis_physician_id, "PHYSICIAN_TYPE"]

        location_type = None if diagnosis_event is None else diagnosis_event.get("txn_location_type")
        num_conditions = _count_high_risk_conditions(
            condition_transactions=_get_transactions_for_type(
                transactions_by_type, TransactionType.CONDITIONS
            ),
            diagnosis_description=diagnosis_description,
        )

        model_rows.append(
            {
                "PATIENT_ID": patient_id,
                "TARGET": int(row.TARGET),
                "DISEASEX_DT": diagnosis_date,
                "PATIENT_AGE": _calculate_patient_age_at_diagnosis(diagnosis_date, patient_birth_year),
                "PATIENT_GENDER": patient_gender,
                "NUM_CONDITIONS": num_conditions,
                "PHYSICIAN_TYPE": physician_type,
                "PHYSICIAN_STATE": physician_state,
                "LOCATION_TYPE": location_type,
            }
        )

    return pd.DataFrame(
        model_rows,
        columns=[
            "PATIENT_ID",
            "TARGET",
            "DISEASEX_DT",
            "PATIENT_AGE",
            "PATIENT_GENDER",
            "NUM_CONDITIONS",
            "PHYSICIAN_TYPE",
            "PHYSICIAN_STATE",
            "LOCATION_TYPE",
        ],
    )
