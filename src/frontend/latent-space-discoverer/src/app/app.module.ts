import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { ToolbarComponent } from './components/toolbar/toolbar.component';
import { LatentDisplayComponent } from './components/dimension-labeling/latent-display/latent-display.component';
import {HttpClientModule} from "@angular/common/http";
import {NgOptimizedImage} from "@angular/common";
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import {MatSlideToggleModule} from "@angular/material/slide-toggle";
import {MatButtonModule} from "@angular/material/button";
import { LatentDisplayControlsComponent } from './components/dimension-labeling/latent-display/latent-display-controls/latent-display-controls.component';
import {MatIconModule} from "@angular/material/icon";
import {MatFormFieldModule} from "@angular/material/form-field";
import {MatSelectModule} from "@angular/material/select";
import {FormsModule} from "@angular/forms";
import { ConfigurationComponent } from './components/configuration/configuration.component';
import {MatCheckboxModule} from "@angular/material/checkbox";
import {MatInputModule} from "@angular/material/input";
import { ImageStripComponent } from './components/image-strip/image-strip.component';
import { DimensionLabelingComponent } from './components/dimension-labeling/dimension-labeling.component';
import { ShiftLabelingComponent } from './components/shift-labeling/shift-labeling.component';

@NgModule({
  declarations: [
    AppComponent,
    ToolbarComponent,
    LatentDisplayComponent,
    LatentDisplayControlsComponent,
    ConfigurationComponent,
    ImageStripComponent,
    DimensionLabelingComponent,
    ShiftLabelingComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule,
    NgOptimizedImage,
    BrowserAnimationsModule,
    MatSlideToggleModule,
    MatButtonModule,
    MatIconModule,
    MatFormFieldModule,
    MatSelectModule,
    FormsModule,
    MatCheckboxModule,
    MatInputModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
