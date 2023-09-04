import { Injectable } from '@angular/core';
import {HttpClient, HttpResponse} from "@angular/common/http";
import {Observable} from "rxjs";
import {ImageStrip} from "../models/image-strip.model";
import {GetShiftedImages} from "../models/get-shifted-images.model";
import {GetRandomNoise} from "../models/get-random-noise.model";
import {SessionLabelsModel} from "../models/session-labels.model";
import {GetValidationResultsModel} from "../models/get-validation-results-model.model";
import {ValidationResultsModel} from "../models/validation-results-model.model";
import {GetDirectionCountModel} from "../models/get-direction-count-model.model";

@Injectable({
  providedIn: 'root'
})
export class BackendService {
  api_root: string = 'http://127.0.0.1:8000'
  responseOptions = {
    responseType: 'json',
    observe: 'response'
  }

  constructor(private httpClient: HttpClient) { }

  listAvailableDatasets(): Observable<Array<any>> {
    console.log("Listing available datasets")
    return this.httpClient.get<Array<string>>(`${this.api_root}/list_available_datasets`, { responseType: 'json' })
  }

  getDirectionCount(getDirectionCountModel: GetDirectionCountModel): Observable<number> {
    console.log("Getting direction count")
    return this.httpClient.post<number>(`${this.api_root}/get_direction_count`, getDirectionCountModel, { responseType: 'json' })
  }

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

  getValidationResults(getValidationResultsModel: GetValidationResultsModel): Observable<ValidationResultsModel>{
    console.log("Getting validation results as base64 strings")
    return this.httpClient.post<ValidationResultsModel>(`${this.api_root}/get_validation_results`, getValidationResultsModel, {responseType: 'json'})
  }
}
