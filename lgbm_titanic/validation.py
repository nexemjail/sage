from typing import Optional

import pydantic


# TODO: make it beautiful
class RequestModel(pydantic.BaseModel):
    Pclass: int
    Sex: str
    Age: float
    Fare: float
    Embarked: str


class ResponseModel(pydantic.BaseModel):
    Survived: int
