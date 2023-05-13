import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LatentDisplayComponent } from './latent-display.component';

describe('LatentDisplayComponent', () => {
  let component: LatentDisplayComponent;
  let fixture: ComponentFixture<LatentDisplayComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ LatentDisplayComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(LatentDisplayComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
