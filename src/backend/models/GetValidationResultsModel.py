from pydantic import BaseModel


class GetValidationResultsModel(BaseModel):
    weighted_dims: list[tuple[int, int]]
    dataset: str
    direction_matrix: str
    random_noise: list[float]
