"""Tests for model-table feature engineering helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from ml_antiviral_diagnosis.feature_engineering import (
    CONTRAINDICATIONS_VALUES,
    INSURANCE_TYPE_VALUES,
    LOCATION_TYPE_VALUES,
    MODEL_TABLE_CATEGORICAL_COLUMNS,
    MODEL_TABLE_CATEGORICAL_ENUMS,
    PATIENT_GENDER_VALUES,
    PHYSICIAN_STATE_VALUES,
    PHYSICIAN_TYPE_VALUES,
    ContraindicationsLevel,
    InsuranceType,
    LocationType,
    PatientGender,
    PhysicianState,
    PhysicianType,
    add_model_table_transaction_features,
    clean_model_table_categorical_nulls,
)


def test_add_model_table_transaction_features_uses_matching_diagnosis_event() -> None:
    """It uses the diagnosis event represented in the model table row."""
    model_table_df = pd.DataFrame(
        {
            "PATIENT_ID": [101],
            "TARGET": [1],
            "DISEASEX_DT": ["2024-01-05"],
            "PATIENT_AGE": [42],
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
    assert result.loc[0, "HIGH_RISK"] == 0


def test_add_model_table_transaction_features_parses_serialized_transactions() -> None:
    """It supports the CSV-serialized diagnosis dataset format."""
    model_table_df = pd.DataFrame(
        {
            "PATIENT_ID": [202, 203],
            "DISEASEX_DT": ["2024-02-10", "2024-02-15"],
            "PATIENT_AGE": [30, 72],
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
    assert result["HIGH_RISK"].tolist() == [0, 1]


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
                    "PATIENT_AGE": [20],
                }
            ),
            diagnosis_dataset_df=pd.DataFrame({"patient_id": [1]}),
        )

    with pytest.raises(ValueError, match="model_table is missing required columns: PATIENT_AGE"):
        add_model_table_transaction_features(
            model_table_df=pd.DataFrame(
                {
                    "PATIENT_ID": [1],
                    "DISEASEX_DT": ["2024-01-01"],
                }
            ),
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


def test_add_model_table_transaction_features_sets_high_risk_from_age_or_condition() -> None:
    """It applies the age and underlying-condition high-risk rules."""
    model_table_df = pd.DataFrame(
        {
            "PATIENT_ID": [301, 302, 303, 304],
            "DISEASEX_DT": ["2024-03-01", "2024-03-01", "2024-03-01", "2024-03-01"],
            "PATIENT_AGE": [70, 35, 12, 35],
        }
    )
    diagnosis_dataset_df = pd.DataFrame(
        {
            "patient_id": [301, 302, 303, 304],
            "transactions_by_type": [
                {
                    "SYMPTOMS": [],
                    "CONDITIONS": [
                        {
                            "txn_dt": "2024-03-01",
                            "physician_id": 1,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "disease_x",
                        }
                    ],
                    "CONTRAINDICATIONS": [],
                    "TREATMENTS": [],
                },
                {
                    "SYMPTOMS": [],
                    "CONDITIONS": [
                        {
                            "txn_dt": "2024-02-20",
                            "physician_id": 1,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "diabetes",
                        },
                        {
                            "txn_dt": "2024-03-01",
                            "physician_id": 1,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "disease_x",
                        },
                    ],
                    "CONTRAINDICATIONS": [],
                    "TREATMENTS": [],
                },
                {
                    "SYMPTOMS": [],
                    "CONDITIONS": [
                        {
                            "txn_dt": "2024-02-20",
                            "physician_id": 1,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "obesity",
                        },
                        {
                            "txn_dt": "2024-03-01",
                            "physician_id": 1,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "disease_x",
                        },
                    ],
                    "CONTRAINDICATIONS": [],
                    "TREATMENTS": [],
                },
                {
                    "SYMPTOMS": [],
                    "CONDITIONS": [
                        {
                            "txn_dt": "2024-02-20",
                            "physician_id": 1,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "mental_health_disorders",
                        },
                        {
                            "txn_dt": "2024-03-01",
                            "physician_id": 1,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "disease_x",
                        },
                    ],
                    "CONTRAINDICATIONS": [],
                    "TREATMENTS": [],
                },
            ],
        }
    )

    result = add_model_table_transaction_features(model_table_df, diagnosis_dataset_df)

    assert result["HIGH_RISK"].tolist() == [1, 1, 0, 0]


def test_clean_model_table_categorical_nulls_fills_physician_columns() -> None:
    """It fills physician categorical nulls with UNSPECIFIED by default."""
    model_table_df = pd.DataFrame(
        {
            "PATIENT_ID": [1, 2],
            "PHYSICIAN_TYPE": [None, "UNSPECIFIED"],
            "PHYSICIAN_STATE": [None, "TX"],
            "INSURANCE_TYPE": ["COMMERCIAL", "MEDICARE"],
        }
    )

    result = clean_model_table_categorical_nulls(model_table_df)

    assert result["PHYSICIAN_TYPE"].tolist() == ["UNSPECIFIED", "UNSPECIFIED"]
    assert result["PHYSICIAN_STATE"].tolist() == ["UNSPECIFIED", "TX"]
    assert result["INSURANCE_TYPE"].tolist() == ["COMMERCIAL", "MEDICARE"]


def test_clean_model_table_categorical_nulls_rejects_missing_columns() -> None:
    """It validates that requested cleanup columns exist."""
    with pytest.raises(ValueError, match="model_table is missing cleanup columns"):
        clean_model_table_categorical_nulls(
            pd.DataFrame({"PHYSICIAN_TYPE": [None]}),
        )


def test_model_table_categorical_enums_match_declared_values() -> None:
    """It exposes exact categorical enums for the training schema."""
    assert MODEL_TABLE_CATEGORICAL_COLUMNS == (
        "PATIENT_GENDER",
        "PHYSICIAN_TYPE",
        "PHYSICIAN_STATE",
        "LOCATION_TYPE",
        "INSURANCE_TYPE",
        "CONTRAINDICATIONS",
    )
    assert [member.value for member in PatientGender] == list(PATIENT_GENDER_VALUES)
    assert [member.value for member in PhysicianType] == list(PHYSICIAN_TYPE_VALUES)
    assert [member.value for member in PhysicianState] == list(PHYSICIAN_STATE_VALUES)
    assert [member.value for member in LocationType] == list(LOCATION_TYPE_VALUES)
    assert [member.value for member in InsuranceType] == list(INSURANCE_TYPE_VALUES)
    assert [member.value for member in ContraindicationsLevel] == list(
        CONTRAINDICATIONS_VALUES
    )


def test_model_table_categorical_enum_mapping_points_to_expected_enums() -> None:
    """It provides a reusable mapping from model-table columns to enum types."""
    assert MODEL_TABLE_CATEGORICAL_ENUMS == {
        "PATIENT_GENDER": PatientGender,
        "PHYSICIAN_TYPE": PhysicianType,
        "PHYSICIAN_STATE": PhysicianState,
        "LOCATION_TYPE": LocationType,
        "INSURANCE_TYPE": InsuranceType,
        "CONTRAINDICATIONS": ContraindicationsLevel,
    }
    assert InsuranceType.UNSPECIFIED.value == "UNSPECIFIED"
    assert ContraindicationsLevel.UNSPECIFIED.value == "Unspecified"