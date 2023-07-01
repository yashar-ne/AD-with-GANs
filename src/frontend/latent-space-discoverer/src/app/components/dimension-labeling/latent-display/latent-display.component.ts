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
    console.log("Adding dimension", this.dim)
    this.ls.addToLocalLabels(this.dim)
    this.updateImages.emit()
  }

  noClickHandler() {
    this.updateImages.emit()
  }
}
