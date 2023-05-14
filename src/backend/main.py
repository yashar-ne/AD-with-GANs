import os
import sys

import numpy as np

sys.path.append('../ml')
from src.ml.latent_direction_visualizer import get_random_strip_as_numpy_array

import io
import uvicorn
from PIL import Image
from fastapi import FastAPI, Response
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
    byte_list = []
    img_arr = get_random_strip_as_numpy_array(os.path.abspath("../out_dir/data.npy"))
    for i in img_arr:
        two_d = (np.reshape(i, (28, 28)) * 255).astype(np.uint8)
        img = Image.fromarray(two_d, 'L')

        with io.BytesIO() as buf:
            img.save(buf, format='PNG')
            im_bytes = buf.getvalue()

        byte_list.append(im_bytes)

    return Response(im_bytes)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
