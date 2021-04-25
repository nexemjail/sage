from sklearn.model_selection import cross_val_score

from lgbm_titanic.data import get_data, COLUMN_TARGET
from pipeline import get_pipeline


# used to optimize a model in sagemaker.
# Printing values to console and capturing it in sagemaker via regex
def cross_validate(data_path: str, model_params: dict = None):
    pipeline = get_pipeline(model_params=model_params)
    folds = 5

    train_df = get_data(data_path, "train")
    f1_metrics = cross_val_score(
        pipeline,
        train_df.drop(columns=[COLUMN_TARGET]),
        train_df[COLUMN_TARGET],
        cv=folds,
        n_jobs=-1,
        scoring="f1",
        random_state=42,
    )
    print(f"F1={sum(f1_metrics) / folds}")


if __name__ == "__main__":
    raise NotImplemented()
