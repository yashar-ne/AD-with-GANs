import { Component } from '@angular/core';
import {BackendService} from "../../services/backend.service";
import {LabelingService} from "../../services/labeling.service";

@Component({
  selector: 'shift-labeling',
  templateUrl: './shift-labeling.component.html',
  styleUrls: ['./shift-labeling.component.scss']
})
export class ShiftLabelingComponent {
  constructor(private bs: BackendService, private ls: LabelingService) {}
}
