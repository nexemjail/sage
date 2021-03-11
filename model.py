from typing import Union
from lightgbm.sklearn import LGBMClassifier


def get_model(params: Union[None, dict] = None):
    default_params = dict(n_estimators=150)
    if params is not None:
        default_params.update(params)
    return LGBMClassifier(**params)
