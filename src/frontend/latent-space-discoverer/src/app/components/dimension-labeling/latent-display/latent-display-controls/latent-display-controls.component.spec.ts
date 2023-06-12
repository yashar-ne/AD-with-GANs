import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LatentDisplayControlsComponent } from './latent-display-controls.component';

describe('LatentDisplayControlsComponent', () => {
  let component: LatentDisplayControlsComponent;
  let fixture: ComponentFixture<LatentDisplayControlsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ LatentDisplayControlsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(LatentDisplayControlsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
