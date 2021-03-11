import pandas as pd
import os


def get_data(data_path, channel="train") -> pd.DataFrame:
    assert channel in {"train", "validation", "test"}, "Invalid channel"
    df = pd.read_csv(os.path.join(data_path, f"{channel}.csv"))
    return df
