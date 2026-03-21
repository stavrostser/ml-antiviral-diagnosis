"""FastAPI application for model inference."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from ml_antiviral_diagnosis.feature_engineering import (
    ContraindicationsLevel,
    InsuranceType,
    MODEL_TABLE_CATEGORICAL_COLUMNS,
    MODEL_TABLE_CATEGORICAL_ENUMS,
    LocationType,
    PatientGender,
    PhysicianState,
    PhysicianType,
    determine_high_risk_flag,
    get_high_risk_condition_options,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_SET_FILENAME = "training_set.csv"
MODEL_FILENAME = "20260321-004014_random_forest_classifier.joblib"
TRAINING_SET_PATH = DATASET_DIR / TRAINING_SET_FILENAME
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
PREDICTION_THRESHOLD = 0.5
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


class InferenceRequest(BaseModel):
    """Input payload for classifier inference.

    Attributes:
        patient_age: Patient age at diagnosis time.
        patient_gender: Patient gender category.
        num_conditions: Number of qualifying conditions.
        physician_type: Physician specialty category.
        physician_state: Physician state category.
        location_type: Location category.
        insurance_type: Insurance category.
        contraindications: Contraindication level category.
        underlying_conditions: Underlying conditions used for high-risk validation.
    """

    model_config = ConfigDict(use_enum_values=True)

    patient_age: int = Field(alias="PATIENT_AGE", ge=0)
    patient_gender: PatientGender = Field(alias="PATIENT_GENDER")
    num_conditions: int = Field(alias="NUM_CONDITIONS", ge=0)
    physician_type: PhysicianType = Field(alias="PHYSICIAN_TYPE")
    physician_state: PhysicianState = Field(alias="PHYSICIAN_STATE")
    location_type: LocationType = Field(alias="LOCATION_TYPE")
    insurance_type: InsuranceType = Field(alias="INSURANCE_TYPE")
    contraindications: ContraindicationsLevel = Field(alias="CONTRAINDICATIONS")
    underlying_conditions: list[str] = Field(
        default_factory=list,
        alias="UNDERLYING_CONDITIONS",
    )


class InferenceResponse(BaseModel):
    """Response payload for classifier inference.

    Attributes:
        high_risk: Whether the patient qualifies for this model.
        message: Summary of the outcome.
        prediction: Predicted class label when inference is run.
        predicted_probability: Probability of the positive class when inference is run.
        threshold: Classification threshold applied to the probability.
        model_filename: Model artifact used for inference.
    """

    high_risk: bool
    message: str
    prediction: int | None = None
    predicted_probability: float | None = None
    threshold: float | None = None
    model_filename: str | None = None


class CategoricalOptionsResponse(BaseModel):
    """Available categorical choices for the inference form."""

    options: dict[str, list[str]]


def _get_categorical_options() -> dict[str, list[str]]:
    """Return categorical values allowed by the training schema.

    Returns:
        Mapping from raw input column name to its allowed categorical values.
    """

    options = {
        column: [member.value for member in enum_type]
        for column, enum_type in MODEL_TABLE_CATEGORICAL_ENUMS.items()
    }
    options["UNDERLYING_CONDITIONS"] = list(get_high_risk_condition_options())
    return options


@lru_cache(maxsize=1)
def _get_model() -> Any:
    """Load the persisted classifier model.

    Returns:
        The joblib-loaded classifier.

    Raises:
        FileNotFoundError: If the model artifact does not exist.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _get_training_feature_columns() -> tuple[str, ...]:
    """Load the feature column order used during training.

    Returns:
        Ordered feature column names excluding the target column.

    Raises:
        FileNotFoundError: If the training-set CSV does not exist.
    """

    if not TRAINING_SET_PATH.exists():
        raise FileNotFoundError(f"Training set file not found: {TRAINING_SET_PATH}")

    training_df = pd.read_csv(TRAINING_SET_PATH, nrows=1)
    feature_columns = [column for column in training_df.columns if column != "TARGET"]
    return tuple(feature_columns)


def _validate_categorical_values(raw_row: dict[str, Any]) -> None:
    """Validate categorical values against the training enums.

    Args:
        raw_row: Raw input row using training column names.

    Raises:
        ValueError: If any categorical value is outside the training schema.
    """

    categorical_options = _get_categorical_options()
    invalid_values = {
        column: raw_row[column]
        for column in MODEL_TABLE_CATEGORICAL_COLUMNS
        if raw_row[column] not in categorical_options[column]
    }
    if invalid_values:
        invalid_text = ", ".join(
            f"{column}={value!r}" for column, value in sorted(invalid_values.items())
        )
        raise ValueError(f"Invalid categorical values provided: {invalid_text}")


def _prepare_inference_frame(request: InferenceRequest) -> pd.DataFrame:
    """Apply the notebook preprocessing steps to an inference request.

    Args:
        request: Raw API request payload.

    Returns:
        A one-row DataFrame aligned to the trained model's feature order.

    Raises:
        ValueError: If the request cannot be transformed into a valid model row.
    """

    raw_row = request.model_dump(by_alias=True)

    _validate_categorical_values(raw_row)

    raw_df = pd.DataFrame(
        [
            {
                key: value
                for key, value in raw_row.items()
                if key != "UNDERLYING_CONDITIONS"
            }
        ]
    )
    for column in MODEL_TABLE_CATEGORICAL_COLUMNS:
        enum_type = MODEL_TABLE_CATEGORICAL_ENUMS[column]
        allowed_values = [member.value for member in enum_type]
        raw_df[column] = pd.Categorical(raw_df[column], categories=allowed_values)

    encoded_df = pd.get_dummies(raw_df, columns=list(MODEL_TABLE_CATEGORICAL_COLUMNS), dtype=int)
    feature_columns = list(_get_training_feature_columns())
    aligned_df = encoded_df.reindex(columns=feature_columns, fill_value=0)
    return aligned_df


app = FastAPI(title="ML Antiviral Diagnosis Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a minimal health response.

    Returns:
        A JSON-serializable health payload.
    """

    return {"status": "ok"}


@app.get("/categorical-options", response_model=CategoricalOptionsResponse)
def get_categorical_options() -> CategoricalOptionsResponse:
    """Return categorical dropdown values for the frontend.

    Returns:
        Categorical values keyed by raw request field name.
    """

    return CategoricalOptionsResponse(options=_get_categorical_options())


@app.post("/predict", response_model=InferenceResponse)
def predict(request: InferenceRequest) -> InferenceResponse:
    """Run model inference on a single JSON payload.

    Args:
        request: Input payload containing the raw model features.

    Returns:
        Predicted class and probability.

    Raises:
        HTTPException: If preprocessing or model loading fails.
    """

    try:
        is_high_risk = bool(
            determine_high_risk_flag(
                patient_age=request.patient_age,
                condition_descriptions=request.underlying_conditions,
            )
        )
        if not is_high_risk:
            return InferenceResponse(
                high_risk=False,
                message="Patient does not meet the high-risk criteria, so inference was not run.",
            )

        inference_df = _prepare_inference_frame(request)
        model = _get_model()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    probability = float(model.predict_proba(inference_df.to_numpy())[:, 1][0])
    prediction = int(probability >= PREDICTION_THRESHOLD)
    return InferenceResponse(
        high_risk=True,
        message="Patient is high risk. Inference completed successfully.",
        prediction=prediction,
        predicted_probability=probability,
        threshold=PREDICTION_THRESHOLD,
        model_filename=MODEL_FILENAME,
    )