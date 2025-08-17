from pydantic import BaseModel
from typing import List, Optional

class Recipe(BaseModel):
    id: str
    title: str
    url: Optional[str] = None
    ingredients: List[str]
    instructions: List[str] | str
    tags: List[str] | None = None
    time: dict | None = None
    yields: str | None = None

class Chunk(BaseModel):
    doc_id: str
    section: str
    position: int
    text: str
    title: str
    url: str | None = None
    tags: List[str] | None = None
    time: dict | None = None
    yields: str | None = None
