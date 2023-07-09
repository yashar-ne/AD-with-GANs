import torch
import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src.backend.models.GetValidationResultsModel import GetValidationResultsModel
from src.backend.models.SaveLabelToDbModel import SaveLabelToDbModel
from src.backend.models.GetRandomNoiseModel import GetRandomNoiseModel
from src.backend.models.GetShiftedImagesModel import GetShiftedImagesModel
from src.backend.controller.main_controller import MainController
from src.backend.models.SessionLabelsModel import SessionLabelsModel

main_controller: MainController = MainController(generator_path="../saved_models/generator.pkl",
                                                 matrix_a_path="../saved_models/matrix_a.pkl",
                                                 z_dim=100)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/get_shifted_images")
async def get_shifted_images(body: GetShiftedImagesModel):
    return main_controller.get_shifted_images(body.z,
                                              body.shifts_range,
                                              body.shifts_count,
                                              body.dim,
                                              body.pca_component_count,
                                              body.pca_skipped_components_count,
                                              body.pca_apply_standard_scaler)


@app.post("/get_shifted_images_from_dimension_labels")
async def get_shifted_images_from_dimension_labels(body: SessionLabelsModel):
    return main_controller.get_shifted_image_from_dimension_labels(body)


@app.post("/get_random_noise")
async def get_shifted_images(body: GetRandomNoiseModel):
    z = main_controller.get_random_noise(body.dim)
    return torch.squeeze(z).tolist()


@app.post("/save_to_db")
async def save_to_db(body: SaveLabelToDbModel):
    return main_controller.save_to_db(z=body.z,
                                      shifts_count=body.shifts_count,
                                      shifts_range=body.shifts_range,
                                      dim=body.dim, is_anomaly=body.is_anomaly)


@app.post("/save_session_labels_to_db")
async def save_session_labels_to_db(body: SessionLabelsModel):
    return main_controller.save_session_labels_to_db(session_labels=body)


@app.get("/get_image_strip_from_prerendered_sample")
async def get_image_strip_from_prerendered_sample():
    return main_controller.get_image_strip_from_prerendered_sample()


@app.post("/get_validation_results")
async def get_validation_results(body: GetValidationResultsModel):
    return main_controller.get_validation_results(weighted_dims=body.weighted_dims,
                                                  pca_component_count=body.pca_component_count,
                                                  skipped_components_count=body.skipped_components_count,
                                                  n_neighbours=20)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
