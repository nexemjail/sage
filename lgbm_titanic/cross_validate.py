import numpy as np
from sklearn.model_selection import cross_val_score

from lgbm_titanic.data import get_data, COLUMN_TARGET
from pipeline import get_pipeline


def cross_validate(data_path: str, model_params: dict = None):
    pipeline = get_pipeline(model_params=model_params)

    train_df = get_data(data_path, "train")
    f1_metrics = cross_val_score(
        pipeline,
        train_df.drop(columns=[COLUMN_TARGET]),
        train_df[COLUMN_TARGET],
        cv=5,
        n_jobs=-1,
        scoring="f1",
        random_state=42,
    )
    print(f"F1={np.mean(f1_metrics)}")


if __name__ == "__main__":
    raise NotImplemented()
