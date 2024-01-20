from pydantic import BaseModel


class GetRandomNoiseModel(BaseModel):
    dim: int
    dataset_name: str
