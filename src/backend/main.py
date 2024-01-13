import torch
import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src.backend.controller.main_controller import MainController
from src.backend.models.GetDirectionCountModel import GetDirectionCountModel
from src.backend.models.GetRandomNoiseModel import GetRandomNoiseModel
from src.backend.models.GetShiftedImagesModel import GetShiftedImagesModel
from src.backend.models.GetSingleImageModel import GetSingleImageModel
from src.backend.models.GetValidationResultsModel import GetValidationResultsModel
from src.backend.models.SessionLabelsModel import SessionLabelsModel

main_controller: MainController = MainController(base_path='../data', z_dim=100)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/list_available_datasets")
async def list_available_datasets():
    return main_controller.list_available_datasets()


@app.post("/get_direction_count")
async def get_direction_count(body: GetDirectionCountModel):
    return main_controller.get_direction_count(body.dataset_name, body.direction_matrix_name)


@app.post("/get_single_image")
async def get_shifted_images(body: GetSingleImageModel):
    return main_controller.get_single_image(body.dataset_name,
                                            body.z
                                            )


@app.post("/get_shifted_images")
async def get_shifted_images(body: GetShiftedImagesModel):
    return main_controller.get_shifted_images(body.dataset_name,
                                              body.direction_matrix_name,
                                              body.z,
                                              body.shifts_range,
                                              body.shifts_count,
                                              body.dim,
                                              body.direction
                                              )


@app.post("/get_random_noise")
async def get_shifted_images(body: GetRandomNoiseModel):
    z = main_controller.get_random_noise(body.dim)
    return torch.squeeze(z).tolist()


@app.post("/save_session_labels_to_db")
async def save_session_labels_to_db(body: SessionLabelsModel):
    return main_controller.save_session_labels_to_db(session_labels=body)


@app.post("/get_validation_results")
async def get_validation_results(body: GetValidationResultsModel):
    return main_controller.get_validation_results(dataset_name=body.dataset,
                                                  direction_matrix_name=body.direction_matrix,
                                                  anomalous_directions=body.weighted_dims,
                                                  random_noise=body.random_noise)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
