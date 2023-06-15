import {Component, EventEmitter, Input, OnDestroy, OnInit, Output} from '@angular/core';
import {BackendService} from "../../../services/backend.service";
import {Observable, take} from "rxjs";
import {ImageStrip} from "../../../models/image-strip.model";
import {SaveLabelModel} from "../../../models/save-label-to-db-model.model";
import {LabelingService} from "../../../services/labeling.service";

@Component({
  selector: 'latent-display',
  templateUrl: './latent-display.component.html',
  styleUrls: ['./latent-display.component.scss']
})
export class LatentDisplayComponent {

  @Input() shiftedImages$: Observable<Array<ImageStrip>> | undefined

  @Input() shiftRange: number = 0
  @Input() shiftRangeSelectOptions: number[] = []
  @Input() shiftCount: number = 0
  @Input() shiftCountSelectOptions: number[] = []
  @Input() dim: number = 0
  @Input() maxdim: number = 0

  @Output() updateImages: EventEmitter<any> = new EventEmitter();

  constructor(private bs: BackendService, private ls: LabelingService) {}

  yesClickHandler() {
    this.saveLabel(true)
    this.updateImages.emit()
  }

  noClickHandler() {
    this.saveLabel(false)
    this.updateImages.emit()
  }

  saveLabel(label: boolean) {
    this.ls.addToLocalLabels({
      shifts_count: this.shiftCount,
      shifts_range: this.shiftRange,
      dim: this.dim,
      is_anomaly: label
    })
  }

  saveToDb(data: SaveLabelModel) {
    this.bs.saveToDb(data)
      .pipe(take(1))
      .subscribe((value) => {console.log(value)})
  }

  restartHandler() {
    location.reload()
  }
}
