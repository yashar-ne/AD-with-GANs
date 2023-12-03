import {Component, OnDestroy, OnInit} from '@angular/core';
import {BackendService} from "../../services/backend.service";
import {LabelingService} from "../../services/labeling.service";
import {catchError, map, Observable, ObservedValueOf, of, Subscription, switchMap, take, throwError} from "rxjs";
import {Router} from "@angular/router";
import {ValidationResultsModel} from "../../models/validation-results-model.model";
import {GetValidationResultsModel} from "../../models/get-validation-results-model.model";

@Component({
  selector: 'labeling-results',
  templateUrl: './labeling-results.component.html',
  styleUrls: ['./labeling-results.component.scss'],
})
export class LabelingResultsComponent implements OnInit, OnDestroy {
  // validationResults$: Observable<ValidationResultsModel> | undefined
  subscription$: Subscription | undefined
  validationResults: ValidationResultsModel | undefined = undefined
  constructor(private router: Router, private bs: BackendService, private ls: LabelingService) {}

  restartHandler() {
    this.router.navigate(['/'])
  }

  ngOnInit(): void {
    this.subscription$ = this.bs.saveSessionLabelsToDb(this.ls.getData())
      .pipe(
        take(1),
        switchMap( () =>
          this.bs.getValidationResults({
            weighted_dims: this.ls.getData().anomalous_dims,
            dataset: this.ls.getData().dataset[0],
            direction_matrix: this.ls.getData().direction_matrix,
          })
        ),
        catchError((err) => {
            if (err.status === 406) {
              alert('No validation can be generated for given labels')
              this.restartHandler()
            }
            return of(undefined)
          }
        )
      )
      .subscribe((value: ValidationResultsModel | undefined) => {
        return this.validationResults = value;
      })
  }
  ngOnDestroy(): void {
    this.subscription$?.unsubscribe()
  }
}
