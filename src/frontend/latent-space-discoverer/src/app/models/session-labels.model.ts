import {SaveLabelModel} from "./save-label-to-db-model.model";
export interface SessionLabelsModel {
  z: number[]
  anomalous_dims: any[]
  shifts_range: number
  shifts_count: number
  use_pca: boolean
  pca_component_count: number
  pca_skipped_components_count: number
  dataset: string
  direction_matrix: string
}
