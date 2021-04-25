import pydantic


class RequestModel(pydantic.BaseModel):
    Pclass: int
    Sex: str
    Age: float
    Fare: float
    Embarked: str


class ResponseModel(pydantic.BaseModel):
    Survived: int
