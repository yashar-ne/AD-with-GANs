import {Component, Input} from '@angular/core';
import {ImageStrip} from "../../models/image-strip.model";

@Component({
  selector: 'image-strip',
  templateUrl: './image-strip.component.html',
  styleUrls: ['./image-strip.component.scss']
})
export class ImageStripComponent {
  @Input() images: Array<ImageStrip> = []
}
