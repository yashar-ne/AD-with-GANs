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

  sessionStarted: boolean = false
  shiftRangeSelectOptions: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  shiftCountSelectOptions: number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

  subscriptionZ$: Subscription | undefined

  z: number[] = []
  shiftRange: number = 5
  shiftCount: number = 5
  dim = 0
  shiftedImages$: Observable<Array<ImageStrip>> | undefined

  constructor(private bs: BackendService) {}

  ngOnInit(): void {
    this.subscriptionZ$ = this.bs.getRandomNoise({dim: 100})
      .pipe(take(1))
      .subscribe((z) => {this.z = z; console.log(z)})
  }

  updateImages() {
    this. dim = Math.floor(Math.random() * 100)
    this.shiftedImages$ = this.bs.getShiftedImages({
      dim: this.dim,
      z: this.z,
      shifts_count: this.shiftCount,
      shifts_range: this.shiftRange,
    })
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

  ngOnDestroy(): void {
    this.subscriptionZ$?.unsubscribe()
  }
}
