from pydantic import BaseModel


class SaveLabelToDbModel(BaseModel):
    shifts_range: int
    shifts_count: int
    dim: int
    is_anomaly: bool
    dataset: str
