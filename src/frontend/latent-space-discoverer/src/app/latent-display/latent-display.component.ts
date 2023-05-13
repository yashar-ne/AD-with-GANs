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
  ngOnInit(): void {
    this.data$ = this.bs.getData()
  }

}
