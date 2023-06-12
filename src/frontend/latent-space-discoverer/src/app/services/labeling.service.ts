import { Injectable } from '@angular/core';
import {SaveLabelToDbModel} from "../models/save-label-to-db-model.model";

@Injectable({
  providedIn: 'root'
})
export class LabelingService {

  labels: SaveLabelToDbModel[] = []
  constructor() { }

  getLocalLabels() {
    return this.labels
  }

  addToLocalLabels(data: SaveLabelToDbModel) {
    this.labels.push(data)
  }
}
