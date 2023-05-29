from pydantic import BaseModel


class GetShiftedImagesModel(BaseModel):
    z: list[int]
    shifts_range: int
    shifts_count: int
    dim: int
