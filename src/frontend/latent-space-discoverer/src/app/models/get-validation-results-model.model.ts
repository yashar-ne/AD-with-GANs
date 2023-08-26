export interface GetValidationResultsModel {
  weighted_dims: number[]
  pca_component_count: number
  skipped_components_count: number
  dataset: string
  direction_matrix: string
}
