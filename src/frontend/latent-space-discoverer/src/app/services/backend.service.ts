import { Injectable } from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {Observable} from "rxjs";
import {ImageStrip} from "../models/image-strip.model";
import {GetShiftedImages} from "../models/get-shifted-images.model";
import {GetRandomNoise} from "../models/get-random-noise.model";
import {SaveLabelModel} from "../models/save-label-to-db-model.model";
import {SessionLabelsModel} from "../models/session-labels.model";
import {GetValidationResultsModel} from "../models/get-validation-results-model.model";
import {ValidationResultsModel} from "../models/validation-results-model.model";

@Injectable({
  providedIn: 'root'
})
export class BackendService {
  api_root: string = 'http://127.0.0.1:8000'
  constructor(private httpClient: HttpClient) { }

  getRandomNoise(getRandomNoise: GetRandomNoise): Observable<Array<number>> {
    console.log("Getting Random Noise")
    return this.httpClient.post<Array<number>>(`${this.api_root}/get_random_noise`, getRandomNoise, { responseType: 'json' })
  }

  getShiftedImages(getShiftedImages: GetShiftedImages): Observable<Array<ImageStrip>> {
    console.log("Getting Shifted Images")
    return this.httpClient.post<Array<ImageStrip>>(`${this.api_root}/get_shifted_images`, getShiftedImages, { responseType: 'json' })
  }

  saveSessionLabelsToDb(sessionLabelsModel: SessionLabelsModel): Observable<boolean> {
    console.log("Calling save_session_labels_to_db endpoint")
    return this.httpClient.post<boolean>(`${this.api_root}/save_session_labels_to_db`, sessionLabelsModel, { responseType: 'json' })
  }

  getImageStripFromPrerenderedSample(): Observable<Array<ImageStrip>> {
    console.log("Getting Image-Strip")
    return this.httpClient.get<Array<ImageStrip>>(`${this.api_root}/get_image_strip_from_prerendered_sample`, { responseType: 'json' })
  }

  getValidationResults(getValidationResultsModel: GetValidationResultsModel): Observable<ValidationResultsModel>{
    console.log("Getting validation results as base64 strings")
    return this.httpClient.post<ValidationResultsModel>(`${this.api_root}/get_validation_results`, getValidationResultsModel, { responseType: 'json' })
  }
}
