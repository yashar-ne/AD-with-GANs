from pydantic import BaseModel


class SaveLabelToDbModel(BaseModel):
    z: list[float]
    shifts_range: int
    shifts_count: int
    dim: int
    is_anomaly: bool
