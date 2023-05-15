import { Injectable } from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {Observable} from "rxjs";
import {Data} from "../models/data.model";
import {ImageStrip} from "../models/image-strip.model";

@Injectable({
  providedIn: 'root'
})
export class BackendService {
  api_root: string = 'http://127.0.0.1:8000'
  constructor(private httpClient: HttpClient) { }

  getData(): Observable<Data> {
    console.log("Getting Data")
    return this.httpClient.get<Data>(`${this.api_root}/get_sample`)
  }

  getImages(): Observable<Blob> {
    console.log("Getting Images")
    return this.httpClient.get(`${this.api_root}/get_images`, { responseType: 'blob' });
  }

  getImageStrip(): Observable<Array<ImageStrip>> {
    console.log("Getting Image-Strip")
    return this.httpClient.get<Array<ImageStrip>>(`${this.api_root}/get_image_strip`, { responseType: 'json' })
  }
}
