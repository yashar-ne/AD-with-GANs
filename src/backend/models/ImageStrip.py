from pydantic import BaseModel


class ImageStrip(BaseModel):
    position: int
    image: str
