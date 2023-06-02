import os
import sys
import base64
import numpy as np

from src.backend.models.ImageStripModel import ImageStripModel
from src.backend.models.GetShiftedImagesModel import GetShiftedImagesModel
from src.backend.controller.main_controller import MainController

sys.path.append('../ml')
from src.ml.latent_direction_visualizer import get_random_strip_as_numpy_array

import io
import uvicorn
from PIL import Image
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

main_controller: MainController = MainController(generator_path="../saved_models/generator.pkl", z_dim=100)
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


@app.post("/get_shifted_images")
async def get_shifted_images(body: GetShiftedImagesModel):
    return main_controller.get_shifted_images(body.z, body.shifts_range, body.shifts_count, body.dim)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
