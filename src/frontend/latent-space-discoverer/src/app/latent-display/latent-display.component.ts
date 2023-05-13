import {Component, OnInit} from '@angular/core';
import {BackendService} from "../services/backend.service";
import {Observable} from "rxjs";
import {Data} from "../models/data.model";

@Component({
  selector: 'latent-display',
  templateUrl: './latent-display.component.html',
  styleUrls: ['./latent-display.component.scss']
})
export class LatentDisplayComponent implements OnInit{

  constructor(private bs: BackendService) { }

  data$: Observable<Data> = this.bs.getData()
  imageToShow: any;
  isImageLoading = false;

  ngOnInit(): void {
    this.getImageFromService()
  }

  getImageFromService() {
      this.isImageLoading = true;
      this.bs.getImages().subscribe({
        next: (data) => {
          this.createImageFromBlob(data);
          this.isImageLoading = false;
        },
        error: (e) => {
          this.isImageLoading = false;
          console.error(e);
        },
        complete: () => console.info('DONE')
      })
  }
  createImageFromBlob(image: Blob) {
     let reader = new FileReader();
     reader.addEventListener("load", () => {
        this.imageToShow = reader.result;
     }, false);

     if (image) {
        reader.readAsDataURL(image);
     }
  }

}
