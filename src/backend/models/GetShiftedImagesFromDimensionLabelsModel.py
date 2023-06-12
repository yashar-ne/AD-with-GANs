from pydantic import BaseModel

from src.backend.models.GetShiftedImagesModel import GetShiftedImagesModel


class GetShiftedImagesFromDimensionLabelsModel(BaseModel):
    dimension_labels: [](GetShiftedImagesModel, bool)

