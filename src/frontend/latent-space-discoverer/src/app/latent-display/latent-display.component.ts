import {Component, OnInit} from '@angular/core';
import {BackendService} from "../services/backend.service";
import {Observable} from "rxjs";
import {Data} from "../models/data.model";
import {ImageStrip} from "../models/image-strip.model";

@Component({
  selector: 'latent-display',
  templateUrl: './latent-display.component.html',
  styleUrls: ['./latent-display.component.scss']
})
export class LatentDisplayComponent {

  constructor(private bs: BackendService) { }

  imageStrip$: Observable<Array<ImageStrip>> = this.bs.getImageStrip()
}
