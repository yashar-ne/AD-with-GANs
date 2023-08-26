import {Component, OnDestroy, OnInit} from '@angular/core';
import {Observable, Subscription, take} from "rxjs";
import {ImageStrip} from "../../models/image-strip.model";
import {BackendService} from "../../services/backend.service";
import {LabelingService} from "../../services/labeling.service";
import {Router} from "@angular/router";

@Component({
  selector: 'dimension-labeling',
  templateUrl: './dimension-labeling.component.html',
  styleUrls: ['./dimension-labeling.component.scss']
})
export class DimensionLabelingComponent implements OnInit, OnDestroy {
  subscriptionZ$: Subscription | undefined
  subscriptionDatasets$: Subscription | undefined
  shiftedImages$: Observable<Array<ImageStrip>> | undefined

  sessionStarted: boolean = false
  datasetSelectOptions: any[] = []
  shiftRangeSelectOptions: number[] = Array.from(Array(100).keys())
  shiftCountSelectOptions: number[] = Array.from(Array(21).keys())

  dataset: string = ''
  directionMatrix: string = ''
  shiftRange: number = 20
  shiftCount: number = 20
  dim: number = 0
  direction: number = 1
  maxdim: number = 100

  usePCA: boolean = true
  pcaComponentCount: number = 20
  pcaSkippedComponentsCount: number = 0

  directionSequence: Array<DirectionSequence> = []
  sequenceIndex: number = 0

  constructor(private router: Router, private bs: BackendService, private ls: LabelingService) { }

  ngOnInit(): void {
    this.subscriptionZ$ = this.bs.getRandomNoise({dim: 100})
      .pipe(take(1))
      .subscribe((z: number[]) => {
        this.ls.data.z = z
      })

    this.subscriptionDatasets$ = this.bs.listAvailableDatasets()
      .pipe(take(1))
      .subscribe((datasetSelectOptions: any[]) => {
        console.log(datasetSelectOptions)
        this.datasetSelectOptions = datasetSelectOptions
      }
    )
  }

  updateImages() {
    if (this.sequenceIndex >= this.directionSequence.length && this.ls.getData().anomalous_dims.length > 0) {
      console.log("Labeling Done. Navigating to Shift-Labeling")
      this.router.navigate(['/labeling-results'])
      return
    } else if (this.sequenceIndex >= this.directionSequence.length && this.ls.getData().anomalous_dims.length === 0) {
      alert("No dimension was labeled as anomalous. Calculation can only take place if at least one dimension is labeled as anomalous. Please start over.")
      location.reload()
    }

    this.dim = this.directionSequence[this.sequenceIndex].dimension
    this.direction = this.directionSequence[this.sequenceIndex].direction

    this.shiftedImages$ = this.bs.getShiftedImages({
      dim: this.dim,
      direction: this.direction,
      z: this.ls.getNoiseArray(),
      shifts_count: this.shiftCount,
      shifts_range: this.shiftRange,
      pca_component_count: this.usePCA ? this.pcaComponentCount : 0,
      pca_skipped_components_count: this.usePCA ? this.pcaSkippedComponentsCount : 0,
      dataset: this.dataset[0],
      direction_matrix: this.directionMatrix,
    })

    this.sequenceIndex++
  }

  startHandler() {
    if (this.usePCA && (this.pcaComponentCount <= this.pcaSkippedComponentsCount)){
      alert("PCA Components must be larger than Skip Components")
    }
    else if (this.dataset === '' || this.directionMatrix === '') {
      alert("Please select a dataset and direction matrix")
    }
    else {
      this.ls.setData({
        z: this.ls.data.z,
        anomalous_dims: [],
        shifts_count: this.shiftCount,
        shifts_range: this.shiftRange,
        use_pca: this.usePCA,
        pca_component_count: this.usePCA ? this.pcaComponentCount : 0,
        pca_skipped_components_count: this.usePCA ? this.pcaSkippedComponentsCount : 0,
        dataset: this.dataset,
        direction_matrix: this.directionMatrix,
      })

      this.sessionStarted = true
      this.maxdim = this.usePCA ? this.pcaComponentCount : this.maxdim
      this.directionSequence = this.generateDirectionSequence()
      this.updateImages()
    }
  }

  generateDirectionSequence(): DirectionSequence[] {
    let direction = 1
    let dimension = 0
    let result = []
    while (dimension < this.maxdim) {
      result.push({dimension: dimension, direction: direction})
      result.push({dimension: dimension, direction: direction*(-1)})
      dimension++
    }

    return result
  }

  ngOnDestroy(): void {
    this.subscriptionZ$?.unsubscribe()
    this.subscriptionDatasets$?.unsubscribe()
  }
}

type DirectionSequence = {
  dimension: number,
  direction: number
}
