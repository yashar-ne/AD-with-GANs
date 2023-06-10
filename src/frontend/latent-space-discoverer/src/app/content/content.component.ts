import {Component, OnDestroy, OnInit} from '@angular/core';
import {Observable, Subscription, take} from "rxjs";
import {ImageStrip} from "../models/image-strip.model";
import {BackendService} from "../services/backend.service";

@Component({
  selector: 'content',
  templateUrl: './content.component.html',
  styleUrls: ['./content.component.scss']
})
export class ContentComponent implements OnInit, OnDestroy {
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

  ngOnDestroy(): void {
    this.subscriptionZ$?.unsubscribe()
  }
}
