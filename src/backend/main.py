import os
import sys
import base64
import numpy as np

from src.backend.models.ImageStrip import ImageStrip

sys.path.append('../ml')
from src.ml.latent_direction_visualizer import get_random_strip_as_numpy_array

import io
import uvicorn
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost:4200/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/get_sample")
async def get_sample():
    return {"message": "Stuff Is Working"}


@app.get("/get_images")
async def get_images():
    return FileResponse("../../out_dir/10/0_19.jpg")


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

        image_list.append(ImageStrip(position=idx, image=img_str))

    return image_list


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
