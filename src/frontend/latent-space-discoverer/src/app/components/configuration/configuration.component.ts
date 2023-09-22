import {Component, EventEmitter, Input, Output} from '@angular/core';
import {MatSelectChange} from "@angular/material/select";
import {MatCheckboxChange} from "@angular/material/checkbox";

@Component({
  selector: 'configuration',
  templateUrl: './configuration.component.html',
  styleUrls: ['./configuration.component.scss']
})
export class ConfigurationComponent {
  @Input() dataset: string | undefined
  @Output() datasetChange: EventEmitter<string> = new EventEmitter();
  @Input() datasets: any[] | undefined

  @Input() directionMatrix: string | undefined
  @Output() directionMatrixChange: EventEmitter<string> = new EventEmitter();
  @Input() directionMatrixSelectOptions: string[] | undefined

  @Input() shiftRange: number | undefined
  @Output() shiftRangeChange: EventEmitter<number> = new EventEmitter();
  @Input() shiftRangeSelectOptions: number[] | undefined;

  @Input() shiftCount: number | undefined;
  @Output() shiftCountChange: EventEmitter<number> = new EventEmitter();
  @Input() shiftCountSelectOptions: number[] | undefined;

  @Output() startHandler: EventEmitter<any> = new EventEmitter();
  @Output() generateDirectionSequence: EventEmitter<any> = new EventEmitter();
  @Output() updateDatasetPreviewImage: EventEmitter<any> = new EventEmitter();

  startHandlerClick() {
    this.startHandler.emit()
  }

  onDatasetSelectChange(value: MatSelectChange) {
    this.datasetChange.emit(value.value)
    this.directionMatrixSelectOptions = value.value[1]
  }

  onDirectionMatrixSelectChange(value: MatSelectChange) {
    this.directionMatrixChange.emit(value.value)
    this.generateDirectionSequence.emit(10)
  }

  onShiftRangeChange(value: MatSelectChange) {
    this.shiftRangeChange.emit(value.value)
  }

  onShiftCountChange(value: MatSelectChange) {
    this.shiftCountChange.emit(value.value)
  }
}
