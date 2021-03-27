import json
import logging
import os
from typing import Union

import joblib
from sklearn.metrics import f1_score

from lgbm_titanic.data import get_data
from lgbm_titanic.pipeline import get_pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)


def train_model(
    train_data_path: str,
    validation_data_path: str,
    save_model_path: str,
    model_params: Union[None, dict] = None,
):

    pipeline = get_pipeline(model_params)

    train_df = get_data(train_data_path, "train")
    pipeline.fit(train_df.drop(columns=["Survived"]), train_df.Survived)

    validation_df = get_data(validation_data_path, "validation")

    logger.info(
        "F1={}".format(
            f1_score(
                validation_df.Survived,
                pipeline.predict(validation_df.drop(columns=["Survived"])),
            )
        )
    )
    output_path = os.path.join(save_model_path, "model.joblib")
    joblib.dump(pipeline, output_path)


if __name__ == "__main__":
    train_model(
        train_data_path=os.environ.get("SM_CHANNEL_TRAIN"),
        validation_data_path=os.environ.get("SM_CHANNEL_VALIDATION"),
        save_model_path=os.environ.get("SM_MODEL_DIR"),
        model_params=json.loads(os.environ.get("SM_HPS")),
    )
