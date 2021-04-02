import logging
import os

import joblib
import pandas as pd
from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse

from lgbm_titanic.data import COLUMN_FEATURES
from lgbm_titanic.validation import ResponseModel, RequestModel

logger = logging.getLogger(__name__)

model = None


app = FastAPI()


def load_model():
    logger.info(
        "Loading a model from folder",
        extra={"folder_name": str(os.listdir("/opt/ml/model/"))},
    )
    return joblib.load(
        os.path.join(
            os.environ.get("SM_MODEL_DIR", "/opt/ml/model/"), "model.joblib"
        )
    )


@app.on_event("startup")
async def startup_event():
    global model
    model = load_model()


@app.post("/invocations", response_model=ResponseModel)
async def predict(request: RequestModel):
    df = pd.DataFrame(request.dict(), index=[0])
    # ordering columns
    label = int(model.predict(df[COLUMN_FEATURES])[0])
    return ResponseModel(Survived=label)


@app.get("/ping")
async def status(request: Request):
    return JSONResponse({}, status_code=200)
