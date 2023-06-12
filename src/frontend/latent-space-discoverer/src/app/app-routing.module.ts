import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {DimensionLabelingComponent} from "./components/dimension-labeling/dimension-labeling.component";

const routes: Routes = [
  { path: '', component: DimensionLabelingComponent },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
