from pydantic import BaseModel

from src.backend.models.SaveLabelToDbModel import SaveLabelToDbModel


class SessionLabelsModel(BaseModel):
    z: list[float]
    labels: list[SaveLabelToDbModel]
