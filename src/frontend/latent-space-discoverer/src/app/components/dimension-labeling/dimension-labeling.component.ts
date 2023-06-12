import {Component, OnDestroy, OnInit} from '@angular/core';
import {Observable, Subscription, take} from "rxjs";
import {ImageStrip} from "../../models/image-strip.model";
import {BackendService} from "../../services/backend.service";

@Component({
  selector: 'dimension-labeling',
  templateUrl: './dimension-labeling.component.html',
  styleUrls: ['./dimension-labeling.component.scss']
})
export class DimensionLabelingComponent implements OnInit, OnDestroy {
  subscriptionZ$: Subscription | undefined
  shiftedImages$: Observable<Array<ImageStrip>> | undefined

  sessionStarted: boolean = false
  shiftRangeSelectOptions: number[] = Array.from(Array(100).keys())
  shiftCountSelectOptions: number[] = Array.from(Array(21).keys())

  z: number[] = []
  shiftRange: number = 10
  shiftCount: number = 10
  dim: number = -1
  maxdim: number = 100

  usePCA: boolean = true
  pcaComponentCount: number = 20
  pcaSkippedComponentsCount: number = 3
  pcaUseStandardScaler: boolean = true

  constructor(private bs: BackendService) {
  }

  ngOnInit(): void {
    this.subscriptionZ$ = this.bs.getRandomNoise({dim: 100})
      .pipe(take(1))
      .subscribe((z) => this.z = z)
  }

  updateImages() {
    this.shiftedImages$ = this.bs.getShiftedImages({
      dim: this.dim,
      z: this.z,
      shifts_count: this.shiftCount,
      shifts_range: this.shiftRange,
      pca_component_count: this.usePCA ? this.pcaComponentCount : 0,
      pca_skipped_components_count: this.usePCA ? this.pcaSkippedComponentsCount : 0,
      pca_apply_standard_scaler: this.usePCA ? this.pcaUseStandardScaler : false
    })
    this.dim++
  }

  startHandler() {
    this.sessionStarted = true
    this.maxdim = this.usePCA ? this.pcaComponentCount : this.maxdim
    this.updateImages()
  }

  ngOnDestroy(): void {
    this.subscriptionZ$?.unsubscribe()
  }
}
