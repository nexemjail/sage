import os

import pandas as pd

COLUMN_FEATURES = [
    "Age",
    "Fare",
    "Sex",
    "Embarked",
    "Pclass",
]
COLUMN_TARGET = "Survived"


def get_data(data_path, channel="train") -> pd.DataFrame:
    assert channel in {"train", "validation", "test"}, "Invalid channel"
    df = pd.read_csv(os.path.join(data_path, f"{channel}.csv"))

    if channel in {"train", "validation"}:
        return df[COLUMN_FEATURES + [COLUMN_TARGET]]

    return df[COLUMN_FEATURES]
