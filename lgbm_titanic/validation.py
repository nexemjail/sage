import pydantic


# TODO: make it beautiful
class RequestModel(pydantic.BaseModel):
    PassengerId: int = None
    Pclass: int = None
    Name: str = None
    Sex: str = None
    Age: float = None
    SibSp: int = None
    Parch: int = None
    Ticket: str = None
    Fare: float = None
    Cabin: str = None
    Embarked: str = None


class ResponseModel(pydantic.BaseModel):
    Survived: int
