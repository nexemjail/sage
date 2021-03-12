from typing import Union
from lightgbm.sklearn import LGBMClassifier


def get_model(params: Union[None, dict] = None):
    model_param = dict(n_estimators=150)
    if params is not None:
        model_param.update(params)
    return LGBMClassifier(**model_param)
