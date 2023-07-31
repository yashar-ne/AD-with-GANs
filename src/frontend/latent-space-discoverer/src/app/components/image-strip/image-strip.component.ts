import {Component, Input} from '@angular/core';
import {ImageStrip} from "../../models/image-strip.model";

@Component({
  selector: 'image-strip',
  templateUrl: './image-strip.component.html',
  styleUrls: ['./image-strip.component.scss']
})
export class ImageStripComponent {
  @Input() nImages: number = 0
  @Input() direction: number = 0
  @Input() images: Array<ImageStrip> = []

  indexOfOriginalImage = 0 ? this.direction === -1 : this.nImages

}
