import { Injectable } from '@angular/core';
import {SaveLabelModel} from "../models/save-label-to-db-model.model";
import {SessionLabelsModel} from "../models/session-labels.model";

@Injectable({
  providedIn: 'root'
})
export class LabelingService {

  data: SessionLabelsModel = {
    z: [],
    labels: []
  }
  constructor() { }

  getData() {
    return this.data
  }

  getNoiseArray() {
    return this.data.z
  }

  setNoiseArray(z: number[]) {
    this.data.z = z
  }

  addToLocalLabels(data: SaveLabelModel) {
    this.data.labels.push(data)
    console.log("addToLocalLabels", this.data)
  }
}
