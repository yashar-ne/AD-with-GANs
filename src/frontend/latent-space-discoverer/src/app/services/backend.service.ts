import { Injectable } from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {Observable} from "rxjs";
import {ImageStrip} from "../models/image-strip.model";
import {GetShiftedImages} from "../models/get-shifted-images.model";
import {GetRandomNoise} from "../models/get-random-noise.model";

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

  getRandomNoise(getRandomNoise: GetRandomNoise): Observable<Array<number>> {
    console.log("Getting Random Noise")
    return this.httpClient.post<Array<number>>(`${this.api_root}/get_random_noise`, getRandomNoise, { responseType: 'json' })
  }

  getShiftedImages(getShiftedImages: GetShiftedImages): Observable<Array<ImageStrip>> {
    console.log("Getting Shifted Images")
    return this.httpClient.post<Array<ImageStrip>>(`${this.api_root}/get_shifted_images`, getShiftedImages, { responseType: 'json' })
  }

  saveToDb(): Observable<boolean> {
    console.log("Calling save_to_db endpoint")
    return this.httpClient.get<boolean>(`${this.api_root}/save_to_db`, { responseType: 'json' })
  }
}
