import { Injectable } from '@angular/core';
import {HttpClient} from "@angular/common/http";
import {Observable} from "rxjs";
import {Data} from "../models/data.model";

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
}
