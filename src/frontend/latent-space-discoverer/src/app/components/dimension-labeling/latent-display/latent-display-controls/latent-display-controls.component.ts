import {Component, EventEmitter, Output} from '@angular/core';

@Component({
  selector: 'latent-display-controls',
  templateUrl: './latent-display-controls.component.html',
  styleUrls: ['./latent-display-controls.component.scss']
})
export class LatentDisplayControlsComponent {

  @Output() yesClick = new EventEmitter();
  @Output() noClick = new EventEmitter();
  yesHandler() {
    this.yesClick.emit()
  }

  noHandler() {
    this.noClick.emit()
  }
}
