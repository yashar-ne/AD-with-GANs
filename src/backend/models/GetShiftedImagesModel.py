from pydantic import BaseModel


class GetShiftedImagesModel(BaseModel):
    z: list[float]
    shifts_range: int
    shifts_count: int
    dim: int
    direction: int
    pca_component_count: int
    pca_skipped_components_count: int
    dataset: str
    direction_matrix: str

