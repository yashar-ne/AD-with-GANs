from pydantic import BaseModel


class SessionLabelsModel(BaseModel):
    z: list[float]
    anomalous_dims: list[tuple[int, int]]
    shifts_range: int
    shifts_count: int
