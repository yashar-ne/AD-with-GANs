import {SaveLabelModel} from "./save-label-to-db-model.model";

export interface SessionLabelsModel {
  z: number[]
  anomalous_dims: number[]
  shifts_range: number
  shifts_count: number
  use_pca: boolean
  pca_component_count: number
  pca_skipped_components_count: number
  pca_use_standard_scaler: boolean
}
