from typing import Union
from lightgbm.sklearn import LGBMClassifier


def get_model(params: Union[None, dict] = None) -> LGBMClassifier:
    model_param = dict(n_estimators=150, n_jobs=-1)
    if params is not None:
        model_param.update(params)
    return LGBMClassifier(**model_param)
