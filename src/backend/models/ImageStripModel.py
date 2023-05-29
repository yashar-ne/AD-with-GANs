from pydantic import BaseModel


class ImageStripModel(BaseModel):
    position: int
    image: str
