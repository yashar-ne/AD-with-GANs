import { Component } from '@angular/core';

@Component({
  selector: 'latent-display-controls',
  templateUrl: './latent-display-controls.component.html',
  styleUrls: ['./latent-display-controls.component.scss']
})
export class LatentDisplayControlsComponent {

  yesHandler() {
    console.log("YES")
  }

  noHandler() {
    console.log("NO")
  }
}
