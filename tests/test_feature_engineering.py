"""Tests for model-table feature engineering helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from ml_antiviral_diagnosis.feature_engineering import (
    add_model_table_transaction_features,
)


def test_add_model_table_transaction_features_uses_matching_diagnosis_event() -> None:
    """It uses the diagnosis event represented in the model table row."""
    model_table_df = pd.DataFrame(
        {
            "PATIENT_ID": [101],
            "TARGET": [1],
            "DISEASEX_DT": ["2024-01-05"],
        }
    )
    diagnosis_dataset_df = pd.DataFrame(
        {
            "patient_id": [101],
            "transactions_by_type": [
                {
                    "SYMPTOMS": [],
                    "CONDITIONS": [
                        {
                            "txn_dt": "2024-01-05",
                            "physician_id": 1,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "MEDICARE",
                            "txn_desc": "Disease X",
                        },
                        {
                            "txn_dt": "2024-01-05",
                            "physician_id": 2,
                            "txn_location_type": "HOME",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "DISEASE_X",
                        },
                    ],
                    "CONTRAINDICATIONS": [
                        {
                            "txn_dt": "2024-01-03",
                            "physician_id": 1,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "MEDICARE",
                            "txn_desc": "LOW_CONTRAINDICATION",
                        },
                        {
                            "txn_dt": "2024-01-05",
                            "physician_id": 1,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "MEDICARE",
                            "txn_desc": "high_contraindication",
                        },
                    ],
                    "TREATMENTS": [],
                }
            ],
        }
    )

    result = add_model_table_transaction_features(model_table_df, diagnosis_dataset_df)

    assert result.loc[0, "INSURANCE_TYPE"] == "MEDICARE"
    assert result.loc[0, "CONTRAINDICATIONS"] == "High"


def test_add_model_table_transaction_features_parses_serialized_transactions() -> None:
    """It supports the CSV-serialized diagnosis dataset format."""
    model_table_df = pd.DataFrame(
        {
            "PATIENT_ID": [202, 203],
            "DISEASEX_DT": ["2024-02-10", "2024-02-15"],
        }
    )
    diagnosis_dataset_df = pd.DataFrame(
        {
            "patient_id": [202, 203],
            "transactions_by_type": [
                "{'SYMPTOMS': [], 'CONDITIONS': [{'txn_dt': '2024-02-10', 'physician_id': 5, 'txn_location_type': 'OFFICE', 'insurance_type': 'COMMERCIAL', 'txn_desc': 'disease_x'}], 'CONTRAINDICATIONS': [{'txn_dt': '2024-02-08', 'physician_id': 5, 'txn_location_type': 'OFFICE', 'insurance_type': 'COMMERCIAL', 'txn_desc': 'Medium_Contraindication'}], 'TREATMENTS': []}",
                "{'SYMPTOMS': [], 'CONDITIONS': [{'txn_dt': '2024-02-15', 'physician_id': 8, 'txn_location_type': 'OFFICE', 'insurance_type': 'MEDICAID', 'txn_desc': 'disease_x'}], 'CONTRAINDICATIONS': [], 'TREATMENTS': []}",
            ],
        }
    )

    result = add_model_table_transaction_features(model_table_df, diagnosis_dataset_df)

    assert result["INSURANCE_TYPE"].tolist() == ["COMMERCIAL", "MEDICAID"]
    assert result["CONTRAINDICATIONS"].tolist() == ["Medium", "Unspecified"]


def test_add_model_table_transaction_features_rejects_missing_columns() -> None:
    """It validates both model-table and diagnosis-dataset schemas."""
    with pytest.raises(ValueError, match="model_table is missing required columns"):
        add_model_table_transaction_features(
            model_table_df=pd.DataFrame({"PATIENT_ID": [1]}),
            diagnosis_dataset_df=pd.DataFrame(
                {
                    "patient_id": [1],
                    "transactions_by_type": [
                        {
                            "SYMPTOMS": [],
                            "CONDITIONS": [],
                            "CONTRAINDICATIONS": [],
                            "TREATMENTS": [],
                        }
                    ],
                }
            ),
        )

    with pytest.raises(ValueError, match="diagnosis dataset is missing required columns"):
        add_model_table_transaction_features(
            model_table_df=pd.DataFrame(
                {
                    "PATIENT_ID": [1],
                    "DISEASEX_DT": ["2024-01-01"],
                }
            ),
            diagnosis_dataset_df=pd.DataFrame({"patient_id": [1]}),
        )