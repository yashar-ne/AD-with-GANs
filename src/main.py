import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/get_sample")
async def get_sample():
    return {"message": "Getting Sample..."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
