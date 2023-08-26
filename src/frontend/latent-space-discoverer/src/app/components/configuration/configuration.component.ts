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

  @Input() usePCA: boolean | undefined;
  @Output() usePCAChange: EventEmitter<boolean> = new EventEmitter();

  @Input() pcaComponentCount: number | undefined
  @Output() pcaComponentCountChange: EventEmitter<number> = new EventEmitter();

  @Input() pcaSkippedComponentsCount: number | undefined
  @Output() pcaSkippedComponentsCountChange: EventEmitter<number> = new EventEmitter();

  @Output() startHandler: EventEmitter<any> = new EventEmitter();

  startHandlerClick() {
    this.startHandler.emit()
  }

  onDatasetSelectChange(value: MatSelectChange) {
    console.log("DATASET SELECTED", value)
    this.datasetChange.emit(value.value)
    this.directionMatrixSelectOptions = value.value[1]
  }

  onDirectionMatrixSelectChange(value: MatSelectChange) {
    this.directionMatrixChange.emit(value.value)
  }

  onShiftRangeChange(value: MatSelectChange) {
    this.shiftRangeChange.emit(value.value)
  }

  onShiftCountChange(value: MatSelectChange) {
    this.shiftCountChange.emit(value.value)
  }

  onUsePCAChange(value: MatCheckboxChange) {
    this.usePCAChange.emit(value.checked)
  }

  onPcaComponentCountChange(value: number) {
    this.pcaComponentCountChange.emit(value)
  }

  onPcaSkippedComponentCountChange(value: number) {
    this.pcaSkippedComponentsCountChange.emit(value)
  }
}
