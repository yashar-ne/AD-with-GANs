from pydantic import BaseModel


class GetShiftedImagesModel(BaseModel):
    z: list[float]
    shifts_range: int
    shifts_count: int
    dim: int
    direction: int
    dataset: str
    direction_matrix: str

