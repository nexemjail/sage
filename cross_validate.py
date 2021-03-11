from data import get_data
from pipeline import get_pipeline
from sklearn.model_selection import cross_val_score
import numpy as np


def cross_validate(data_path, model_params=None):
    pipeline = get_pipeline(model_params=model_params)

    train_df = get_data(data_path, "train")
    f1_metrics = cross_val_score(
        pipeline,
        train_df.drop(columns=["Survived"]),
        train_df["Survived"],
        cv=5,
        n_jobs=-1,
        scoring="f1",
        random_state=42,
    )
    print(f"F1={np.mean(f1_metrics)}")


if __name__ == "__main__":
    raise NotImplemented()
