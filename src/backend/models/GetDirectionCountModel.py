from pydantic import BaseModel


class GetDirectionCountModel(BaseModel):
    dataset_name: str
    direction_matrix_name: str
