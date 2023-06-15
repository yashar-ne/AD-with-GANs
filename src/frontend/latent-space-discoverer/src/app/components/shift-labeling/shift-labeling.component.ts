import {Component, OnInit} from '@angular/core';
import {BackendService} from "../../services/backend.service";
import {LabelingService} from "../../services/labeling.service";
import {Observable, take} from "rxjs";
import {ImageStrip} from "../../models/image-strip.model";

@Component({
  selector: 'shift-labeling',
  templateUrl: './shift-labeling.component.html',
  styleUrls: ['./shift-labeling.component.scss']
})
export class ShiftLabelingComponent implements OnInit {
  images$: Observable<Array<ImageStrip>> | undefined

  constructor(private bs: BackendService, private ls: LabelingService) {}

  ngOnInit(): void {
    this.bs.saveSessionLabelsToDb(this.ls.getData())
      .pipe(take(1))
      .subscribe((value) => {
        this.images$ = this.bs.getShiftedImagesFromDimensionLabels(this.ls.getData())
      })
  }
}
