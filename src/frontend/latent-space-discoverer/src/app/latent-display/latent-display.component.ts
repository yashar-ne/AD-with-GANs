import {Component, OnDestroy, OnInit} from '@angular/core';
import {BackendService} from "../services/backend.service";
import {map, Observable, Subscription, take} from "rxjs";
import {ImageStrip} from "../models/image-strip.model";

@Component({
  selector: 'latent-display',
  templateUrl: './latent-display.component.html',
  styleUrls: ['./latent-display.component.scss']
})
export class LatentDisplayComponent implements OnInit, OnDestroy {

  subscriptionZ$: Subscription | undefined
  shiftedImages$: Observable<Array<ImageStrip>> | undefined

  sessionStarted: boolean = false
  shiftRangeSelectOptions: number[] = Array.from(Array(100).keys())
  shiftCountSelectOptions: number[] = Array.from(Array(21).keys())

  z: number[] = []
  shiftRange: number = 20
  shiftCount: number = 10
  dim: number = -1
  maxdim: number = 10

  constructor(private bs: BackendService) {}

  ngOnInit(): void {
    this.subscriptionZ$ = this.bs.getRandomNoise({dim: 100})
      .pipe(take(1))
      .subscribe((z) => {this.z = z; console.log(z)})
  }

  updateImages() {
    this.shiftedImages$ = this.bs.getShiftedImages({
      dim: this.dim,
      z: this.z,
      shifts_count: this.shiftCount,
      shifts_range: this.shiftRange,
    })
    this.dim++
  }

  startHandler() {
    this.sessionStarted = true
    this.updateImages()
  }


  yesClickHandler() {
    console.log("YES")
    this.saveToDb(true)
    this.updateImages()
  }

  noClickHandler() {
    console.log("NO")
    this.saveToDb(false)
    this.updateImages()
  }

  saveToDb(label: boolean) {
    console.log("Saving to DB")
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

  ngOnDestroy(): void {
    this.subscriptionZ$?.unsubscribe()
  }
}
