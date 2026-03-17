"""Data engineering helpers for patient-level feature preparation."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict


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
