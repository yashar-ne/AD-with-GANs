from pydantic import BaseModel


class GetShiftedImagesModel(BaseModel):
    z: list[float]
    shifts_range: int
    shifts_count: int
    dim: int
    direction: int
    dataset_name: str
    direction_matrix_name: str

