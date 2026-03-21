"""Tests for the FastAPI inference module."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml_antiviral_diagnosis import api


class _DummyModel:
    """Minimal classifier stub for API tests."""

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """Return fixed probabilities for deterministic tests.

        Args:
            data: Encoded feature matrix.

        Returns:
            Two-column class probabilities.
        """

        return np.array([[0.25, 0.75] for _ in range(len(data))])


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create a test client with stubbed model-loading helpers.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        FastAPI test client.
    """

    monkeypatch.setattr(api, "_get_model", lambda: _DummyModel())
    monkeypatch.setattr(
        api,
        "_get_training_feature_columns",
        lambda: (
            "PATIENT_AGE",
            "NUM_CONDITIONS",
            "PATIENT_GENDER_F",
            "PATIENT_GENDER_M",
            "PHYSICIAN_TYPE_UNSPECIFIED",
            "PHYSICIAN_STATE_UNSPECIFIED",
            "LOCATION_TYPE_INDEPENDENT LABORATORY",
            "INSURANCE_TYPE_COMMERCIAL",
            "CONTRAINDICATIONS_Unspecified",
        ),
    )
    return TestClient(api.app)


def test_prepare_inference_frame_matches_training_feature_order() -> None:
    """It one-hot encodes and reorders a request like the training notebook."""

    request = api.InferenceRequest(
        PATIENT_AGE=71,
        PATIENT_GENDER="M",
        NUM_CONDITIONS=0,
        PHYSICIAN_TYPE="UNSPECIFIED",
        PHYSICIAN_STATE="UNSPECIFIED",
        LOCATION_TYPE="INDEPENDENT LABORATORY",
        INSURANCE_TYPE="COMMERCIAL",
        CONTRAINDICATIONS="Unspecified",
        UNDERLYING_CONDITIONS=["DIABETES"],
    )

    original_loader = api._get_training_feature_columns
    api._get_training_feature_columns = lambda: (
        "PATIENT_AGE",
        "NUM_CONDITIONS",
        "PATIENT_GENDER_F",
        "PATIENT_GENDER_M",
        "PHYSICIAN_TYPE_UNSPECIFIED",
        "PHYSICIAN_STATE_UNSPECIFIED",
        "LOCATION_TYPE_INDEPENDENT LABORATORY",
        "INSURANCE_TYPE_COMMERCIAL",
        "CONTRAINDICATIONS_Unspecified",
    )

    try:
        result = api._prepare_inference_frame(request)
    finally:
        api._get_training_feature_columns = original_loader

    assert result.columns.tolist() == [
        "PATIENT_AGE",
        "NUM_CONDITIONS",
        "PATIENT_GENDER_F",
        "PATIENT_GENDER_M",
        "PHYSICIAN_TYPE_UNSPECIFIED",
        "PHYSICIAN_STATE_UNSPECIFIED",
        "LOCATION_TYPE_INDEPENDENT LABORATORY",
        "INSURANCE_TYPE_COMMERCIAL",
        "CONTRAINDICATIONS_Unspecified",
    ]
    assert result.iloc[0].to_dict() == {
        "PATIENT_AGE": 71,
        "NUM_CONDITIONS": 0,
        "PATIENT_GENDER_F": 0,
        "PATIENT_GENDER_M": 1,
        "PHYSICIAN_TYPE_UNSPECIFIED": 1,
        "PHYSICIAN_STATE_UNSPECIFIED": 1,
        "LOCATION_TYPE_INDEPENDENT LABORATORY": 1,
        "INSURANCE_TYPE_COMMERCIAL": 1,
        "CONTRAINDICATIONS_Unspecified": 1,
    }


def test_predict_endpoint_skips_inference_for_non_high_risk_patient(
    client: TestClient,
) -> None:
    """It returns a non-high-risk response instead of scoring ineligible patients."""

    response = client.post(
        "/predict",
        json={
            "PATIENT_AGE": 40,
            "PATIENT_GENDER": "F",
            "NUM_CONDITIONS": 1,
            "PHYSICIAN_TYPE": "UNSPECIFIED",
            "PHYSICIAN_STATE": "UNSPECIFIED",
            "LOCATION_TYPE": "OFFICE",
            "INSURANCE_TYPE": "COMMERCIAL",
            "CONTRAINDICATIONS": "Low",
            "UNDERLYING_CONDITIONS": ["MENTAL HEALTH DISORDERS"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "high_risk": False,
        "message": "Patient does not meet the high-risk criteria, so inference was not run.",
        "prediction": None,
        "predicted_probability": None,
        "threshold": None,
        "model_filename": None,
    }


def test_predict_endpoint_returns_json_prediction(client: TestClient) -> None:
    """It serves a JSON prediction payload for a valid request."""

    response = client.post(
        "/predict",
        json={
            "PATIENT_AGE": 71,
            "PATIENT_GENDER": "M",
            "NUM_CONDITIONS": 0,
            "PHYSICIAN_TYPE": "UNSPECIFIED",
            "PHYSICIAN_STATE": "UNSPECIFIED",
            "LOCATION_TYPE": "INDEPENDENT LABORATORY",
            "INSURANCE_TYPE": "COMMERCIAL",
            "CONTRAINDICATIONS": "Unspecified",
            "UNDERLYING_CONDITIONS": ["DIABETES"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "high_risk": True,
        "message": "Patient is high risk. Inference completed successfully.",
        "prediction": 1,
        "predicted_probability": 0.75,
        "threshold": 0.5,
        "model_filename": api.MODEL_FILENAME,
    }


def test_categorical_options_endpoint_returns_training_enums(client: TestClient) -> None:
    """It exposes dropdown-ready categorical values for the frontend."""

    response = client.get("/categorical-options")

    assert response.status_code == 200
    payload = response.json()
    assert payload["options"]["PATIENT_GENDER"] == ["F", "M"]
    assert "UNSPECIFIED" in payload["options"]["PHYSICIAN_TYPE"]
    assert "COMMERCIAL" in payload["options"]["INSURANCE_TYPE"]
    assert "Unspecified" in payload["options"]["CONTRAINDICATIONS"]
    assert "DIABETES" in payload["options"]["UNDERLYING_CONDITIONS"]