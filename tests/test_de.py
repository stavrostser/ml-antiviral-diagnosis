"""Tests for data engineering helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from ml_antiviral_diagnosis.de import transform_fact_txn_to_patient_transactions


def test_transform_fact_txn_to_patient_transactions_builds_patient_rows() -> None:
    """It creates one patient-level row with grouped transaction dictionaries."""
    df = pd.DataFrame(
        {
            "TXN_DT": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "PATIENT_ID": [1, 1, 2],
            "PHYSICIAN_ID": [1001.0, 1002.0, None],
            "TXN_LOCATION_TYPE": ["OFFICE", "HOME", "OFFICE"],
            "INSURANCE_TYPE": ["COMMERCIAL", "COMMERCIAL", "MEDICARE"],
            "TXN_TYPE": ["SYMPTOMS", "CONDITIONS", "TREATMENTS"],
            "TXN_DESC": ["COUGH", "DIABETES", "DRUG_A"],
        }
    )

    result = transform_fact_txn_to_patient_transactions(df)

    assert result["patient_id"].tolist() == [1, 2]
    assert result.columns.tolist() == ["patient_id", "transactions_by_type"]

    patient_one = result.iloc[0]
    assert patient_one["transactions_by_type"]["SYMPTOMS"] == [
        {
            "txn_dt": "2024-01-01",
            "physician_id": 1001,
            "txn_location_type": "OFFICE",
            "insurance_type": "COMMERCIAL",
            "txn_desc": "COUGH",
        }
    ]
    assert patient_one["transactions_by_type"]["CONDITIONS"] == [
        {
            "txn_dt": "2024-01-02",
            "physician_id": 1002,
            "txn_location_type": "HOME",
            "insurance_type": "COMMERCIAL",
            "txn_desc": "DIABETES",
        }
    ]
    assert patient_one["transactions_by_type"]["CONTRAINDICATIONS"] == []
    assert patient_one["transactions_by_type"]["TREATMENTS"] == []


def test_transform_fact_txn_to_patient_transactions_rejects_invalid_txn_type() -> None:
    """It rejects rows with transaction types outside the allowed set."""
    df = pd.DataFrame(
        {
            "TXN_DT": ["2024-01-01"],
            "PATIENT_ID": [1],
            "PHYSICIAN_ID": [1001.0],
            "TXN_LOCATION_TYPE": ["OFFICE"],
            "INSURANCE_TYPE": ["COMMERCIAL"],
            "TXN_TYPE": ["UNKNOWN"],
            "TXN_DESC": ["COUGH"],
        }
    )

    with pytest.raises(ValueError):
        transform_fact_txn_to_patient_transactions(df)


def test_transform_fact_txn_to_patient_transactions_rejects_missing_columns() -> None:
    """It rejects data frames that do not have the required schema."""
    df = pd.DataFrame(
        {
            "PATIENT_ID": [1],
            "TXN_TYPE": ["SYMPTOMS"],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        transform_fact_txn_to_patient_transactions(df)
