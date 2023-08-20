import {Component, EventEmitter, Input, Output} from '@angular/core';
import {MatSelectChange} from "@angular/material/select";
import {MatCheckboxChange} from "@angular/material/checkbox";

@Component({
  selector: 'configuration',
  templateUrl: './configuration.component.html',
  styleUrls: ['./configuration.component.scss']
})
export class ConfigurationComponent {
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
