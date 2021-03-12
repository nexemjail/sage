import os
from argparse import ArgumentParser
from typing import Union

import joblib
from sklearn.metrics import f1_score

from data import get_data
from pipeline import get_pipeline


def train_model(
    data_path: str,
    save_model_path: str,
    model_params: Union[None, dict] = None,
):

    pipeline = get_pipeline(model_params)

    train_df = get_data(data_path, "train")
    pipeline.fit(train_df.drop(columns=["Survived"]), train_df.Survived)

    print(
        "F1={}".format(
            f1_score(
                train_df.Survived,
                pipeline.predict(train_df.drop(columns=["Survived"])),
            )
        )
    )
    output_path = os.path.join(save_model_path, "model.joblib")
    joblib.dump(pipeline, output_path)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--train_data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN"),
    )

    ap.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("SM_MODEL_DIR"),
    )

    args = ap.parse_args()

    train_model(data_path=args.train_data_dir, save_model_path=args.model_path)
