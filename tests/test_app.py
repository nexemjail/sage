import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def sample_data() -> dict:
    return {
        "Pclass": 3,
        "Sex": "female",
        "Age": 27,
        "Fare": 7,
        "Embarked": "S",
    }


@pytest.fixture(scope="module")
def application() -> FastAPI:
    from lgbm_titanic.inference_fastapi import app

    yield app


def test_prediction(application, sample_data):
    # to ensure before_request is called
    with TestClient(application) as test_client:
        prediction = test_client.post(
            "/invocations", data=json.dumps(sample_data)
        )
        response_json = prediction.json()
        assert "Survived" in response_json
        assert response_json["Survived"] in {0, 1}
