import uvicorn
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


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
