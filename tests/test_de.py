"""Tests for data engineering helpers."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from ml_antiviral_diagnosis.de import (
    build_model_table,
    build_patient_diagnosis_dataset,
    transform_fact_txn_to_patient_transactions,
)


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


def test_build_patient_diagnosis_dataset_finds_first_diagnosis_and_target() -> None:
    """It aligns transactions to diagnosis and flags Drug A after diagnosis."""
    patient_transactions_df = pd.DataFrame(
        {
            "patient_id": [1],
            "transactions_by_type": [
                {
                    "SYMPTOMS": [
                        {
                            "txn_dt": "2024-01-01",
                            "physician_id": 1001,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "Fever",
                        },
                        {
                            "txn_dt": "2024-01-04",
                            "physician_id": 1001,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "Cough",
                        },
                    ],
                    "CONDITIONS": [
                        {
                            "txn_dt": "2024-01-03",
                            "physician_id": 1001,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "DISEASE_X",
                        },
                        {
                            "txn_dt": "2024-01-06",
                            "physician_id": 1002,
                            "txn_location_type": "HOME",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "Disease X",
                        },
                    ],
                    "CONTRAINDICATIONS": [
                        {
                            "txn_dt": "2024-01-02",
                            "physician_id": 1001,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "Warfarin",
                        }
                    ],
                    "TREATMENTS": [
                        {
                            "txn_dt": "2024-01-03",
                            "physician_id": 1001,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "drug a",
                        },
                        {
                            "txn_dt": "2024-01-07",
                            "physician_id": 1002,
                            "txn_location_type": "HOME",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "Other treatment",
                        },
                    ],
                }
            ],
        }
    )

    result = build_patient_diagnosis_dataset(patient_transactions_df)

    assert result.columns.tolist() == [
        "patient_id",
        "first_diagnosis_date",
        "transactions_by_type",
        "TARGET",
    ]
    assert result.iloc[0]["first_diagnosis_date"] == date(2024, 1, 3)
    assert result.iloc[0]["TARGET"] == 1
    assert result.iloc[0]["transactions_by_type"]["SYMPTOMS"] == [
        {
            "txn_dt": "2024-01-01",
            "physician_id": 1001,
            "txn_location_type": "OFFICE",
            "insurance_type": "COMMERCIAL",
            "txn_desc": "Fever",
        }
    ]
    assert result.iloc[0]["transactions_by_type"]["CONDITIONS"] == [
        {
            "txn_dt": "2024-01-03",
            "physician_id": 1001,
            "txn_location_type": "OFFICE",
            "insurance_type": "COMMERCIAL",
            "txn_desc": "DISEASE_X",
        }
    ]
    assert result.iloc[0]["transactions_by_type"]["TREATMENTS"] == [
        {
            "txn_dt": "2024-01-03",
            "physician_id": 1001,
            "txn_location_type": "OFFICE",
            "insurance_type": "COMMERCIAL",
            "txn_desc": "drug a",
        }
    ]


def test_build_patient_diagnosis_dataset_excludes_treatment_before_diagnosis() -> None:
    """It does not count Drug A when treatment only occurred before diagnosis."""
    patient_transactions_df = pd.DataFrame(
        {
            "patient_id": [2],
            "transactions_by_type": [
                {
                    "SYMPTOMS": [],
                    "CONDITIONS": [
                        {
                            "txn_dt": "2024-02-10",
                            "physician_id": 2001,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "MEDICARE",
                            "txn_desc": "disease x",
                        }
                    ],
                    "CONTRAINDICATIONS": [],
                    "TREATMENTS": [
                        {
                            "txn_dt": "2024-02-09",
                            "physician_id": 2001,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "MEDICARE",
                            "txn_desc": "DRUG A",
                        }
                    ],
                }
            ],
        }
    )

    result = build_patient_diagnosis_dataset(patient_transactions_df)

    assert result.iloc[0]["first_diagnosis_date"] == date(2024, 2, 10)
    assert result.iloc[0]["TARGET"] == 0
    assert result.iloc[0]["transactions_by_type"]["TREATMENTS"] == [
        {
            "txn_dt": "2024-02-09",
            "physician_id": 2001,
            "txn_location_type": "OFFICE",
            "insurance_type": "MEDICARE",
            "txn_desc": "DRUG A",
        }
    ]


def test_build_patient_diagnosis_dataset_rejects_missing_columns() -> None:
    """It rejects malformed patient transaction frames."""
    df = pd.DataFrame({"patient_id": [1]})

    with pytest.raises(ValueError, match="missing required columns"):
        build_patient_diagnosis_dataset(df)


def test_build_patient_diagnosis_dataset_skips_patients_without_diagnosis() -> None:
    """It excludes patients who never receive a Disease X diagnosis."""
    patient_transactions_df = pd.DataFrame(
        {
            "patient_id": [1, 2],
            "transactions_by_type": [
                {
                    "SYMPTOMS": [],
                    "CONDITIONS": [
                        {
                            "txn_dt": "2024-02-10",
                            "physician_id": 2001,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "MEDICARE",
                            "txn_desc": "disease x",
                        }
                    ],
                    "CONTRAINDICATIONS": [],
                    "TREATMENTS": [],
                },
                {
                    "SYMPTOMS": [
                        {
                            "txn_dt": "2024-02-09",
                            "physician_id": 2002,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "Cough",
                        }
                    ],
                    "CONDITIONS": [
                        {
                            "txn_dt": "2024-02-10",
                            "physician_id": 2002,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "Diabetes",
                        }
                    ],
                    "CONTRAINDICATIONS": [],
                    "TREATMENTS": [],
                },
            ],
        }
    )

    result = build_patient_diagnosis_dataset(patient_transactions_df)

    assert result["patient_id"].tolist() == [1]


def test_build_model_table_populates_expected_columns() -> None:
    """It derives the model-table columns from diagnosis and dimension data."""
    diagnosis_dataset_df = pd.DataFrame(
        {
            "patient_id": [1],
            "first_diagnosis_date": [date(2024, 1, 3)],
            "transactions_by_type": [
                {
                    "SYMPTOMS": [
                        {
                            "txn_dt": "2024-01-01",
                            "physician_id": 1001,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "Fever",
                        }
                    ],
                    "CONDITIONS": [
                        {
                            "txn_dt": "2024-01-02",
                            "physician_id": 1001,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "OBESITY",
                        },
                        {
                            "txn_dt": "2024-01-02",
                            "physician_id": 1001,
                            "txn_location_type": "OFFICE",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "obesity",
                        },
                        {
                            "txn_dt": "2024-01-03",
                            "physician_id": 1001,
                            "txn_location_type": "URGENT CARE",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "DISEASE_X",
                        },
                    ],
                    "CONTRAINDICATIONS": [],
                    "TREATMENTS": [
                        {
                            "txn_dt": "2024-01-03",
                            "physician_id": 1001,
                            "txn_location_type": "URGENT CARE",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "DRUG A",
                        }
                    ],
                }
            ],
            "TARGET": [1],
        }
    )
    dim_patient_df = pd.DataFrame(
        {
            "PATIENT_ID": [1],
            "BIRTH_YEAR": [1984],
            "GENDER": ["F"],
        }
    )
    dim_physician_df = pd.DataFrame(
        {
            "PHYSICIAN_ID": [1001],
            "STATE": ["TX"],
            "PHYSICIAN_TYPE": ["FAMILY MEDICINE"],
        }
    )

    result = build_model_table(diagnosis_dataset_df, dim_patient_df, dim_physician_df)

    assert result.columns.tolist() == [
        "PATIENT_ID",
        "TARGET",
        "DISEASEX_DT",
        "PATIENT_AGE",
        "PATIENT_GENDER",
        "NUM_CONDITIONS",
        "PHYSICIAN_TYPE",
        "PHYSICIAN_STATE",
        "LOCATION_TYPE",
    ]
    assert result.iloc[0].to_dict() == {
        "PATIENT_ID": 1,
        "TARGET": 1,
        "DISEASEX_DT": date(2024, 1, 3),
        "PATIENT_AGE": 40,
        "PATIENT_GENDER": "F",
        "NUM_CONDITIONS": 1,
        "PHYSICIAN_TYPE": "FAMILY MEDICINE",
        "PHYSICIAN_STATE": "TX",
        "LOCATION_TYPE": "URGENT CARE",
    }


def test_build_model_table_handles_unknown_patient_gender_and_missing_physician() -> None:
    """It leaves unsupported gender and unresolved physician fields empty."""
    diagnosis_dataset_df = pd.DataFrame(
        {
            "patient_id": [2],
            "first_diagnosis_date": [date(2024, 2, 1)],
            "transactions_by_type": [
                {
                    "SYMPTOMS": [],
                    "CONDITIONS": [
                        {
                            "txn_dt": "2024-02-01",
                            "physician_id": None,
                            "txn_location_type": "HOME",
                            "insurance_type": "COMMERCIAL",
                            "txn_desc": "Disease X",
                        }
                    ],
                    "CONTRAINDICATIONS": [],
                    "TREATMENTS": [],
                }
            ],
            "TARGET": [0],
        }
    )
    dim_patient_df = pd.DataFrame(
        {
            "PATIENT_ID": [2],
            "BIRTH_YEAR": [2000],
            "GENDER": ["U"],
        }
    )
    dim_physician_df = pd.DataFrame(
        {
            "PHYSICIAN_ID": [999],
            "STATE": ["CA"],
            "PHYSICIAN_TYPE": ["INTERNAL MEDICINE"],
        }
    )

    result = build_model_table(diagnosis_dataset_df, dim_patient_df, dim_physician_df)

    assert result.iloc[0]["PATIENT_GENDER"] is None
    assert result.iloc[0]["PHYSICIAN_TYPE"] is None
    assert result.iloc[0]["PHYSICIAN_STATE"] is None
    assert result.iloc[0]["LOCATION_TYPE"] == "HOME"


def test_build_model_table_rejects_missing_dim_columns() -> None:
    """It rejects malformed dimension inputs."""
    diagnosis_dataset_df = pd.DataFrame(
        {
            "patient_id": [1],
            "first_diagnosis_date": [date(2024, 1, 1)],
            "transactions_by_type": [
                {"SYMPTOMS": [], "CONDITIONS": [], "CONTRAINDICATIONS": [], "TREATMENTS": []}
            ],
            "TARGET": [0],
        }
    )
    dim_patient_df = pd.DataFrame({"PATIENT_ID": [1]})
    dim_physician_df = pd.DataFrame(
        {
            "PHYSICIAN_ID": [1],
            "STATE": ["TX"],
            "PHYSICIAN_TYPE": ["FAMILY MEDICINE"],
        }
    )

    with pytest.raises(ValueError, match="dim_patient is missing required columns"):
        build_model_table(diagnosis_dataset_df, dim_patient_df, dim_physician_df)
