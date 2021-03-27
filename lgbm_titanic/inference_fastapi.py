import json
import logging
import os

import joblib
import pandas as pd
from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse

from lgbm_titanic.data import COLUMN_FEATURES, COLUMN_TARGET

logger = logging.getLogger(__name__)

model = None


app = FastAPI()


@app.on_event("startup")
async def startup_event():
    global model
    logger.info(
        "Loading a model from folder",
        extra={"folder_name": str(os.listdir("/opt/ml/model/"))},
    )
    model = joblib.load(
        os.path.join(
            os.environ.get("SM_MODEL_DIR", "/opt/ml/model/"), "model.joblib"
        )
    )


@app.post("/invocations")
async def predict(request: Request):
    try:
        data_json = await request.json()
    except json.JSONDecodeError as e:
        logger.info(
            "Got invalid json message with content",
            extra={"message_data": await request.body()},
        )
        return JSONResponse({"message": "Invalid json"}, status_code=400)

    # TODO: add validation via marshmallow or similar
    for c in COLUMN_FEATURES:
        if c not in data_json:
            return JSONResponse(
                {"message": f"{c} not present in data"}, status_code=400
            )

    df = pd.DataFrame(
        {k: v for k, v in data_json.items() if k in COLUMN_FEATURES}, index=[0]
    )
    label = int(model.predict(df)[0])
    return JSONResponse({"message": {COLUMN_TARGET: label}}, status_code=200)


@app.get("/ping")
async def status(request: Request):
    return JSONResponse({}, status_code=200)
