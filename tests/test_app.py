import json
from itertools import combinations, chain
from typing import List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def get_sample_data() -> dict:
    return {
        "Pclass": 3,
        "Sex": "female",
        "Age": 27,
        "Fare": 7,
        "Embarked": "S",
    }


@pytest.fixture(scope="module")
def sample_data() -> dict:
    return get_sample_data()


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


# every possible missing field combination
@pytest.mark.parametrize(
    "missing_fields",
    list(
        chain(
            *[
                combinations(get_sample_data(), r)
                for r in range(1, len(get_sample_data()) + 1)
            ]
        )
    ),
)
def test_missing_field(application, sample_data, missing_fields: List[tuple]):
    invalid_sample_data = sample_data.copy()
    for missing_field in missing_fields:
        del invalid_sample_data[missing_field]
    with TestClient(application) as test_client:
        prediction = test_client.post(
            "/invocations", data=json.dumps(invalid_sample_data)
        )
        assert prediction.status_code == 422  # unprocessable entity
        response_json = prediction.json()
        errors_count = 0
        for obj in response_json["detail"]:
            if (
                obj["msg"] == "field required"
                and obj["type"] == "value_error.missing"
                and any(map(lambda mf: mf in obj["loc"], missing_fields))
            ):
                errors_count += 1

        assert errors_count == len(
            missing_fields
        ), "Missing field not validated properly"


def test_ping(application):
    with TestClient(application) as test_client:
        response = test_client.get("/ping")
        assert response.status_code == 200
