import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {DimensionLabelingComponent} from "./components/dimension-labeling/dimension-labeling.component";
import {ShiftLabelingComponent} from "./components/shift-labeling/shift-labeling.component";

const routes: Routes = [
  {path: '', component: DimensionLabelingComponent},
  {path: 'shift-labeling', component: ShiftLabelingComponent},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
