from pydantic import BaseModel


class GetRocAucModel(BaseModel):
    weighted_dims: list[int]
    pca_component_count: int
    skipped_components_count: int
