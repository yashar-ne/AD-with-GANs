import { Injectable } from '@angular/core';
import {SaveLabelModel} from "../models/save-label-to-db-model.model";
import {SessionLabelsModel} from "../models/session-labels.model";

@Injectable({
  providedIn: 'root'
})
export class LabelingService {

  data: SessionLabelsModel = {
    z: [],
    anomalous_dims: [],
    shifts_range: 0,
    shifts_count: 0,
    dataset: '',
    direction_matrix: '',
  }
  constructor() { }

  getData() {
    return this.data
  }

  setData(data: SessionLabelsModel) {
    this.data = data
  }

  getNoiseArray() {
    return this.data.z
  }

  setNoiseArray(z: number[]) {
    this.data.z = z
  }

  addToLocalLabels(dim: number, direction: number) {
    this.data.anomalous_dims.push([dim, direction])
  }
}
