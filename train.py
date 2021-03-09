import os
from argparse import ArgumentParser

import joblib
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train_model(data_path: str, save_model_path: str):
    num_columns = ["Age", "Fare"]
    median_imputer = SimpleImputer(strategy="median")
    cat_columns = ["Sex", "Embarked", "Pclass"]
    ohe = OneHotEncoder(handle_unknown="ignore")
    ct = ColumnTransformer(
        transformers=[
            (
                "num_columns",
                median_imputer,
                num_columns,
            ),
            ("cat_columns", ohe, cat_columns),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", ct),
            ("model", LGBMClassifier(n_estimators=150, verbose=1)),
        ]
    )

    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    pipeline.fit(train_df.drop(columns=["Survived"]), train_df.Survived)

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
