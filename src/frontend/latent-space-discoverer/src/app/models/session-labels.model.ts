import {SaveLabelModel} from "./save-label-to-db-model.model";

export interface SessionLabelsModel {
  z: number[]
  labels: SaveLabelModel[]
}
