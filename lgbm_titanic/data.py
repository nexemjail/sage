import os

import pandas as pd

COLUMN_FEATURES = [
    "PassengerId",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
]
COLUMN_TARGET = "Survived"


def get_data(data_path, channel="train") -> pd.DataFrame:
    assert channel in {"train", "validation", "test"}, "Invalid channel"
    df = pd.read_csv(os.path.join(data_path, f"{channel}.csv"))
    return df
