import {Component, EventEmitter, Input, OnDestroy, OnInit, Output} from '@angular/core';
import {BackendService} from "../services/backend.service";
import {map, Observable, Subscription, take} from "rxjs";
import {ImageStrip} from "../models/image-strip.model";

@Component({
  selector: 'latent-display',
  templateUrl: './latent-display.component.html',
  styleUrls: ['./latent-display.component.scss']
})
export class LatentDisplayComponent {

  @Input() shiftedImages$: Observable<Array<ImageStrip>> | undefined

  @Input() z: number[] = [];
  @Input() shiftRange: number = 0;
  @Input() shiftRangeSelectOptions: number[] = [];
  @Input() shiftCount: number = 0;
  @Input() shiftCountSelectOptions: number[] = [];
  @Input() dim: number = 0;
  @Input() maxdim: number = 0;

  @Output() updateImages: EventEmitter<any> = new EventEmitter();

  constructor(private bs: BackendService) {}

  yesClickHandler() {
    this.saveToDb(true)
    this.updateImages.emit()
  }

  noClickHandler() {
    this.saveToDb(false)
    this.updateImages.emit()
  }

  saveToDb(label: boolean) {
    this.bs.saveToDb({
      z: this.z,
      shifts_count: this.shiftCount,
      shifts_range: this.shiftRange,
      dim: this.dim,
      is_anomaly: label
    })
      .pipe(take(1))
      .subscribe((value) => {console.log(value)})
  }

  restartHandler() {
    location.reload()
  }
}
