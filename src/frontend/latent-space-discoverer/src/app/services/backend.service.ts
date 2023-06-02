import { Injectable } from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {catchError, Observable} from "rxjs";
import {ImageStrip} from "../models/image-strip.model";

@Injectable({
  providedIn: 'root'
})
export class BackendService {
  api_root: string = 'http://127.0.0.1:8000'
  constructor(private httpClient: HttpClient) { }

  getImageStrip(): Observable<Array<ImageStrip>> {
    console.log("Getting Image-Strip")
    return this.httpClient.get<Array<ImageStrip>>(`${this.api_root}/get_image_strip`, { responseType: 'json' })
  }

  getShiftedImages(getShiftedImages: any): Observable<Array<ImageStrip>> {
    console.log("Getting Shifted Images")
    return this.httpClient.post<Array<ImageStrip>>(`${this.api_root}/get_shifted_images`, getShiftedImages, { responseType: 'json' })
  }

}
