from typing import Union

from sklearn.pipeline import Pipeline

from lgbm_titanic.model import get_model
from lgbm_titanic.preprocessing import get_column_transformer


def get_pipeline(model_params: Union[None, dict] = None):
    pipeline = Pipeline(
        steps=[
            ("preprocess", get_column_transformer()),
            ("model", get_model(model_params)),
        ]
    )
    return pipeline
