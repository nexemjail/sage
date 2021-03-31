from typing import Optional

import pydantic


# TODO: make it beautiful
class RequestModel(pydantic.BaseModel):
    PassengerId: int
    Pclass: int
    Name: str
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: Optional[str]
    Embarked: str


class ResponseModel(pydantic.BaseModel):
    Survived: int
