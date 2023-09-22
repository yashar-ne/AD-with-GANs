from pydantic import BaseModel


class GetSingleImageModel(BaseModel):
    dataset_name: str
    z: list[float]

