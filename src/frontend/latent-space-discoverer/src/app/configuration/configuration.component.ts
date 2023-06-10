import {Component, EventEmitter, Input, Output} from '@angular/core';

@Component({
  selector: 'configuration',
  templateUrl: './configuration.component.html',
  styleUrls: ['./configuration.component.scss']
})
export class ConfigurationComponent {
  @Input() shiftRange: number | undefined;
  @Input() shiftRangeSelectOptions: number[] | undefined;

  @Input() shiftCount: number | undefined;
  @Input() shiftCountSelectOptions: number[] | undefined;

  @Output() startHandler: EventEmitter<any> = new EventEmitter();

  startHandlerClick() {
    this.startHandler.emit()
  }

}
