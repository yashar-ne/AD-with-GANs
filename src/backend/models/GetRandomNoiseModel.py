from pydantic import BaseModel


class GetRandomNoiseModel(BaseModel):
    dim: int
