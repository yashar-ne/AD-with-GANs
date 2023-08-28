import {SaveLabelModel} from "./save-label-to-db-model.model";
export interface SessionLabelsModel {
  z: number[]
  anomalous_dims: any[]
  shifts_range: number
  shifts_count: number
  dataset: string
  direction_matrix: string
}
