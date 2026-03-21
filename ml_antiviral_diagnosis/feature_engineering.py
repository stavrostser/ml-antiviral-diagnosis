"""Feature engineering helpers for model-table enrichment."""

from __future__ import annotations

import ast
from datetime import date
from enum import StrEnum
import re
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

MODEL_TABLE_CATEGORICAL_COLUMNS = (
    "PATIENT_GENDER",
    "PHYSICIAN_TYPE",
    "PHYSICIAN_STATE",
    "LOCATION_TYPE",
    "INSURANCE_TYPE",
    "CONTRAINDICATIONS",
)

PATIENT_GENDER_VALUES = (
    "F",
    "M",
)

PHYSICIAN_TYPE_VALUES = (
    "ADOLESCENT MEDICINE (PEDIATRICS)",
    "ADVANCED REGISTERED NURSE",
    "ALLERGY & IMMUNOLOGY",
    "ANATOMIC PATHOLOGY",
    "ANATOMIC/CLINICAL PATHOLOGY",
    "ANESTHESIOLOGY",
    "BEHAVIORAL HEALTH & SOCIAL SERVICES",
    "BLOOD BANKING/TRANSFUSION MEDICINE",
    "CARDIOVASCULAR DISEASE",
    "CHILD & ADOLESCENT PSYCHIATRY",
    "CLINICAL SOCIAL WORKER",
    "CRITICAL CARE MEDICINE (INTERNAL MEDICINE)",
    "CYTOPATHOLOGY",
    "DERMATOPATHOLOGY",
    "DIAGNOSTIC RADIOLOGY",
    "EMERGENCY MEDICAL SERVICES",
    "EMERGENCY MEDICINE",
    "ENDOCRINOLOGY, DIABETES & METABOLISM",
    "FAMILY MEDICINE",
    "FORENSIC PATHOLOGY",
    "GASTROENTEROLOGY",
    "GENERAL PRACTICE",
    "GENERAL PREVENTIVE MEDICINE",
    "GENERAL SURGERY",
    "GERIATRIC MEDICINE (FAMILY MEDICINE)",
    "GERIATRIC MEDICINE (INTERNAL MEDICINE)",
    "HEMATOLOGY (PATHOLOGY)",
    "HEMATOLOGY/ONCOLOGY",
    "HOSPICE & PALLIATIVE MEDICINE",
    "HOSPITALIST",
    "INFECTIOUS DISEASE",
    "INTERNAL MEDICINE",
    "INTERNAL MEDICINE/EMERGENCY MEDICINE",
    "INTERNAL MEDICINE/FAMILY MEDICINE",
    "INTERNAL MEDICINE/PEDIATRICS",
    "MATERNAL & FETAL MEDICINE",
    "MEDICAL GENETICS",
    "NEPHROLOGY",
    "NEUROLOGY",
    "NEUROMUSCULAR MEDICINE (NEUROLOGY)",
    "NEURORADIOLOGY",
    "NOT APPLICABLE",
    "NUCLEAR MEDICINE",
    "NURSE PRACTITIONER",
    "OBSTETRICS & GYNECOLOGY",
    "ORTHOPEDIC SURGERY",
    "OTOLARYNGOLOGY",
    "PAIN MEDICINE",
    "PAIN MEDICINE (PHYSICAL MEDICINE & REHABILITATION)",
    "PEDIATRIC CARDIOLOGY",
    "PEDIATRIC EMERGENCY MEDICINE",
    "PEDIATRIC EMERGENCY MEDICINE (PEDIATRICS)",
    "PEDIATRIC HEMATOLOGY/ONCOLOGY",
    "PEDIATRIC RADIOLOGY",
    "PEDIATRICS",
    "PHARMACIST",
    "PHYSICAL MEDICINE & REHABILITATION",
    "PHYSICAL THERAPY",
    "PHYSICIAN ASSISTANT",
    "PODIATRIST",
    "PSYCHIATRY",
    "PUBLIC HEALTH & GENERAL PREVENTIVE MEDICINE",
    "PULMONARY CRITICAL CARE MEDICINE",
    "PULMONARY DISEASE",
    "RADIATION ONCOLOGY",
    "RADIOLOGY",
    "RHEUMATOLOGY",
    "SELECTIVE PATHOLOGY",
    "SLEEP MEDICINE",
    "SPORTS MEDICINE (EMERGENCY MEDICINE)",
    "SPORTS MEDICINE (FAMILY MEDICINE)",
    "SPORTS MEDICINE (PEDIATRICS)",
    "STUDENT, HEALTH CARE",
    "THORACIC SURGERY",
    "UNSPECIFIED",
    "UROLOGY",
    "VASCULAR & INTERVENTIONAL RADIOLOGY",
)

PHYSICIAN_STATE_VALUES = (
    "AK",
    "AL",
    "AR",
    "AZ",
    "CA",
    "CO",
    "CT",
    "DC",
    "DE",
    "FL",
    "GA",
    "HI",
    "IA",
    "ID",
    "IL",
    "IN",
    "KS",
    "KY",
    "LA",
    "MA",
    "MD",
    "ME",
    "MI",
    "MN",
    "MO",
    "MS",
    "MT",
    "NC",
    "ND",
    "NE",
    "NH",
    "NJ",
    "NM",
    "NV",
    "NY",
    "OH",
    "OK",
    "OR",
    "PA",
    "PR",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UNSPECIFIED",
    "UT",
    "VA",
    "VT",
    "WA",
    "WI",
    "WV",
    "WY",
)

LOCATION_TYPE_VALUES = (
    "AMBULANCE - AIR OR WATER",
    "AMBULANCE - LAND",
    "CLINIC  - FEDERALLY QUALIFIED HEALTH CENTER (FQHC)",
    "CLINIC - FREESTANDING",
    "CLINIC - RURAL HEALTH",
    "COMPREHENSIVE OUTPATIENT REHABILITATION FACILITY",
    "CRITICAL ACCESS HOSPITAL",
    "EMERGENCY ROOM - HOSPITAL",
    "FEDERALLY QUALIFIED HEALTH CENTER",
    "HOSPITAL - LABORATORY SERVICES PROVIDED TO NON-PATIENTS",
    "HOSPITAL INPATIENT (INCLUDING MEDICARE PART A)",
    "HOSPITAL INPATIENT (MEDICARE PART B ONLY)",
    "HOSPITAL OUTPATIENT",
    "INDEPENDENT LABORATORY",
    "INPATIENT HOSPITAL",
    "INPATIENT PSYCHIATRIC FACILITY",
    "MOBILE UNIT",
    "OFF CAMPUS-OUTPATIENT HOSPITAL",
    "OFFICE",
    "ON CAMPUS-OUTPATIENT HOSPITAL",
    "OTHER PLACE OF SERVICE",
    "PHARMACY",
    "RURAL HEALTH CLINIC",
    "TELEHEALTH PROVIDED IN PATIENT'S HOME",
    "TELEHEALTH PROVIDED OTHER THAN IN PATIENT'S HOME",
    "UNASSIGNED",
    "URGENT CARE FACILITY",
    "WALK-IN RETAIL HEALTH CLINIC",
)

INSURANCE_TYPE_VALUES = (
    "COMMERCIAL",
    "MEDICAID",
    "MEDICARE",
    "UNSPECIFIED",
)

CONTRAINDICATIONS_VALUES = (
    "High",
    "Low",
    "Medium",
    "Unspecified",
)

HIGH_RISK_AGE_THRESHOLD = 65
HIGH_RISK_CONDITION_MIN_AGE = 12
HIGH_RISK_CONDITION_VALUES = frozenset(
    {
        "ASTHMA",
        "CANCER",
        "CANCER TREATMENT",
        "CARDIOVASCULAR DISEASE",
        "CHRONIC LUNG DISEASE",
        "COPD",
        "CORONARY ARTERY DISEASE",
        "DIABETES",
        "HEART DISEASE",
        "HEART FAILURE",
        "HIV",
        "HIV AIDS",
        "IMMUNOCOMPROMISED",
        "OBESITY",
        "ORGAN TRANSPLANT",
        "SMOKING",
        "STROKE",
        "WEAKENED IMMUNE SYSTEM",
    }
)

HIGH_RISK_CONDITION_OPTIONS = tuple(sorted(HIGH_RISK_CONDITION_VALUES))


def _build_enum_member_name(value: str, used_names: set[str]) -> str:
    """Create a valid and unique enum member name for a categorical value.

    Args:
        value: Raw categorical value.
        used_names: Previously assigned enum member names.

    Returns:
        A valid ``StrEnum`` member name.
    """
    candidate = re.sub(r"[^A-Z0-9]+", "_", value.strip().upper()).strip("_")
    if not candidate:
        candidate = "EMPTY"
    if candidate[0].isdigit():
        candidate = f"VALUE_{candidate}"

    unique_candidate = candidate
    suffix = 2
    while unique_candidate in used_names:
        unique_candidate = f"{candidate}_{suffix}"
        suffix += 1

    used_names.add(unique_candidate)
    return unique_candidate


def _build_str_enum(enum_name: str, values: tuple[str, ...]) -> type[StrEnum]:
    """Build a ``StrEnum`` from a sequence of categorical values.

    Args:
        enum_name: Name of the enum class.
        values: Ordered categorical values.

    Returns:
        A ``StrEnum`` subtype with the provided values.
    """
    used_names: set[str] = set()
    members = {
        _build_enum_member_name(value, used_names): value
        for value in values
    }
    return StrEnum(enum_name, members)


PatientGender = _build_str_enum("PatientGender", PATIENT_GENDER_VALUES)
PhysicianType = _build_str_enum("PhysicianType", PHYSICIAN_TYPE_VALUES)
PhysicianState = _build_str_enum("PhysicianState", PHYSICIAN_STATE_VALUES)
LocationType = _build_str_enum("LocationType", LOCATION_TYPE_VALUES)
InsuranceType = _build_str_enum("InsuranceType", INSURANCE_TYPE_VALUES)
ContraindicationsLevel = _build_str_enum(
    "ContraindicationsLevel", CONTRAINDICATIONS_VALUES
)

MODEL_TABLE_CATEGORICAL_ENUMS: dict[str, type[StrEnum]] = {
    "PATIENT_GENDER": PatientGender,
    "PHYSICIAN_TYPE": PhysicianType,
    "PHYSICIAN_STATE": PhysicianState,
    "LOCATION_TYPE": LocationType,
    "INSURANCE_TYPE": InsuranceType,
    "CONTRAINDICATIONS": ContraindicationsLevel,
}


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


def _has_high_risk_underlying_condition(
    condition_transactions: list[dict[str, Any]],
) -> bool:
    """Check whether any qualifying high-risk condition is present.

    Args:
        condition_transactions: Condition transactions on or before diagnosis.

    Returns:
        ``True`` when a qualifying underlying condition is present.
    """
    normalized_conditions = {
        _normalize_transaction_text(transaction.get("txn_desc", ""))
        for transaction in condition_transactions
    }
    return any(condition in HIGH_RISK_CONDITION_VALUES for condition in normalized_conditions)


def _is_high_risk_patient(
    patient_age: Any,
    condition_transactions: list[dict[str, Any]],
) -> int:
    """Compute the high-risk flag from age and underlying conditions.

    Args:
        patient_age: Patient age at first diagnosis.
        condition_transactions: Condition transactions on or before diagnosis.

    Returns:
        ``1`` when the patient meets the high-risk rule, otherwise ``0``.
    """
    if pd.isna(patient_age):
        return 0

    normalized_age = int(patient_age)
    if normalized_age >= HIGH_RISK_AGE_THRESHOLD:
        return 1

    if normalized_age <= HIGH_RISK_CONDITION_MIN_AGE:
        return 0

    return int(_has_high_risk_underlying_condition(condition_transactions))


def get_high_risk_condition_options() -> tuple[str, ...]:
    """Return the supported high-risk underlying conditions.

    Returns:
        Sorted condition names used by the high-risk eligibility rule.
    """

    return HIGH_RISK_CONDITION_OPTIONS


def determine_high_risk_flag(
    patient_age: Any,
    condition_descriptions: list[str] | tuple[str, ...] | None,
) -> int:
    """Compute the high-risk flag from age and condition descriptions.

    Args:
        patient_age: Patient age at diagnosis time.
        condition_descriptions: Raw condition descriptions to evaluate.

    Returns:
        ``1`` when the patient meets the high-risk rule, otherwise ``0``.
    """

    condition_transactions = [
        {"txn_desc": description}
        for description in (condition_descriptions or [])
    ]
    return _is_high_risk_patient(
        patient_age=patient_age,
        condition_transactions=condition_transactions,
    )


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
        - ``HIGH_RISK`` using the diagnosis-age rule and qualifying underlying
            condition transactions on or before diagnosis.

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

    if "PATIENT_AGE" not in model_table_df.columns:
        raise ValueError("model_table is missing required columns: PATIENT_AGE")

    diagnosis_lookup = {
        int(row.patient_id): _parse_transactions_by_type(row.transactions_by_type)
        for row in diagnosis_dataset_df.itertuples(index=False)
    }

    enriched_df = model_table_df.copy()
    insurance_types: list[str | None] = []
    contraindication_values: list[str] = []
    high_risk_values: list[int] = []

    for row in enriched_df.itertuples(index=False):
        patient_id = int(row.PATIENT_ID)
        diagnosis_date = _parse_transaction_date(row.DISEASEX_DT)
        transactions_by_type = diagnosis_lookup.get(patient_id)

        diagnosis_event = None
        latest_contraindication = None
        condition_transactions: list[dict[str, Any]] = []
        if transactions_by_type is not None:
            condition_transactions = _get_transactions_for_type(
                transactions_by_type, TransactionType.CONDITIONS
            )
            diagnosis_event = _find_first_transaction_by_description_on_date(
                transactions=condition_transactions,
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
        high_risk_values.append(
            _is_high_risk_patient(
                patient_age=row.PATIENT_AGE,
                condition_transactions=condition_transactions,
            )
        )

    enriched_df["INSURANCE_TYPE"] = insurance_types
    enriched_df["CONTRAINDICATIONS"] = contraindication_values
    enriched_df["HIGH_RISK"] = high_risk_values
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
