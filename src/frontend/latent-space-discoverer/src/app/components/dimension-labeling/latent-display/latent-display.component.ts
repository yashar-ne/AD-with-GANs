import {Component, EventEmitter, Input, OnDestroy, OnInit, Output} from '@angular/core';
import {BackendService} from "../../../services/backend.service";
import {Observable, take} from "rxjs";
import {ImageStrip} from "../../../models/image-strip.model";
import {SaveLabelToDbModel} from "../../../models/save-label-to-db-model.model";

@Component({
  selector: 'latent-display',
  templateUrl: './latent-display.component.html',
  styleUrls: ['./latent-display.component.scss']
})
export class LatentDisplayComponent {

  @Input() shiftedImages$: Observable<Array<ImageStrip>> | undefined

  @Input() z: number[] = []
  @Input() shiftRange: number = 0
  @Input() shiftRangeSelectOptions: number[] = []
  @Input() shiftCount: number = 0
  @Input() shiftCountSelectOptions: number[] = []
  @Input() dim: number = 0
  @Input() maxdim: number = 0

  @Output() updateImages: EventEmitter<any> = new EventEmitter();

  localLabels: SaveLabelToDbModel[] = []

  constructor(private bs: BackendService) {}

  yesClickHandler() {
    this.save(true)
    this.updateImages.emit()
  }

  noClickHandler() {
    this.save(false)
    this.updateImages.emit()
  }

  save(label: boolean) {
    const data = {
      z: this.z,
      shifts_count: this.shiftCount,
      shifts_range: this.shiftRange,
      dim: this.dim,
      is_anomaly: label
    }
    this.saveToLocalLabels(data)
    this.saveToDb(data)

    console.log(this.localLabels)
  }

  saveToLocalLabels(data: SaveLabelToDbModel) {
    this.localLabels.push(data)
  }

  saveToDb(data: SaveLabelToDbModel) {
    this.bs.saveToDb(data)
      .pipe(take(1))
      .subscribe((value) => {console.log(value)})
  }

  restartHandler() {
    location.reload()
  }
}
