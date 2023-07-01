import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {DimensionLabelingComponent} from "./components/dimension-labeling/dimension-labeling.component";
import {LabelingResultsComponent} from "./components/labeling-results/labeling-results.component";

const routes: Routes = [
  {path: '', component: DimensionLabelingComponent},
  {path: 'dimension-labeling', component: DimensionLabelingComponent},
  {path: 'labeling-results', component: LabelingResultsComponent},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
