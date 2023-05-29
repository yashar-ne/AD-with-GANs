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
    image_list = []
    img_arr = get_random_strip_as_numpy_array(os.path.abspath("../out_dir/data.npy"))
    for idx, i in enumerate(img_arr):
        two_d = (np.reshape(i, (28, 28)) * 255).astype(np.uint8)
        img = Image.fromarray(two_d, 'L')

        with io.BytesIO() as buf:
            img.save(buf, format='PNG')
            img_str = base64.b64encode(buf.getvalue())

        image_list.append(ImageStripModel(position=idx, image=img_str))

    return image_list

@app.post("/get_shifted_images")
async def get_shifted_images(body: GetShiftedImagesModel):
    main_controller: MainController
    return main_controller.get_image_strip(body.z, body.shifts_range, body.shifts_count, body.dim)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
