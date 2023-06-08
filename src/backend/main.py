import sys

import torch

from src.backend.models.SaveLabelToDbModel import SaveLabelToDbModel
from src.backend.models.GetRandomNoiseModel import GetRandomNoiseModel
from src.backend.models.GetShiftedImagesModel import GetShiftedImagesModel
from src.backend.controller.main_controller import MainController

sys.path.append('../ml')

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

main_controller: MainController = MainController(generator_path="../saved_models/generator.pkl",
                                                 matrix_a_path="../saved_models/matrix_a_after_pca.pkl",
                                                 z_dim=100,
                                                 )
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/get_image_strip")
async def get_image_strip():
    return main_controller.get_image_strip()


@app.post("/get_random_noise")
async def get_shifted_images(body: GetRandomNoiseModel):
    z = main_controller.get_random_noise(body.dim)
    return torch.squeeze(z).tolist()


@app.post("/get_shifted_images")
async def get_shifted_images(body: GetShiftedImagesModel):
    return main_controller.get_shifted_images(body.z, body.shifts_range, body.shifts_count, body.dim)


@app.post("/save_to_db")
async def save_to_db(body: SaveLabelToDbModel):
    return main_controller.save_to_db(z=body.z, shifts_count=body.shifts_count, shifts_range=body.shifts_range, dim=body.dim, is_anomaly=body.is_anomaly)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
