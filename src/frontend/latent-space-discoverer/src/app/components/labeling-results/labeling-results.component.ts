import {Component, OnInit} from '@angular/core';
import {BackendService} from "../../services/backend.service";
import {LabelingService} from "../../services/labeling.service";
import {Observable, take} from "rxjs";
import {Router} from "@angular/router";
import {ValidationResultsModel} from "../../models/validation-results-model.model";

@Component({
  selector: 'labeling-results',
  templateUrl: './labeling-results.component.html',
  styleUrls: ['./labeling-results.component.scss'],
})
export class LabelingResultsComponent implements OnInit {
  validationResults$: Observable<ValidationResultsModel> | undefined

  constructor(private router: Router, private bs: BackendService, private ls: LabelingService) {}

  restartHandler() {
    this.router.navigate(['/'])
  }

  ngOnInit(): void {
    this.bs.saveSessionLabelsToDb(this.ls.getData())
      .pipe(take(1))
      .subscribe((value) => {
        this.validationResults$ = this.bs.getValidationResults({
          weighted_dims: this.ls.getData().anomalous_dims,
          pca_component_count: this.ls.getData().pca_component_count,
          skipped_components_count: this.ls.getData().pca_skipped_components_count
        })
      })
  }
}
