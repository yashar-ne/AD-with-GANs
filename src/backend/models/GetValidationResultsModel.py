from pydantic import BaseModel


class GetValidationResultsModel(BaseModel):
    weighted_dims: list[tuple[int, int]]
    pca_component_count: int
    skipped_components_count: int
    dataset: str
    direction_matrix: str
