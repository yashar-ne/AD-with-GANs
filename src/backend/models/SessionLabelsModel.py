from pydantic import BaseModel


class SessionLabelsModel(BaseModel):
    z: list[float]
    anomalous_dims: list[tuple[int, int]]
    shifts_range: int
    shifts_count: int
    use_pca: bool
    pca_component_count: int
    pca_skipped_components_count: int
