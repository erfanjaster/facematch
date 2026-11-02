"""Request bodies (responses are plain JSON dicts documented in README)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ImageB64Request(BaseModel):
    image_b64: str = Field(description="base64 encoded image (jpeg/png/webp)")
    top_k: int | None = None
    threshold: float | None = None


class SearchRequest(ImageB64Request):
    pass


class EnrollJsonRequest(BaseModel):
    image_b64: str
    person: str
    create_person: bool = True


class PersonCreate(BaseModel):
    name: str
    meta: dict = {}


class RawFaceAdd(BaseModel):
    person_id: str
    embedding: list[float]
    embedder: str | None = None
    image_b64: str = ""
    crop_b64: str = ""
    meta: dict = {}


class BenchmarkRequest(BaseModel):
    sizes: list[int] = [1_000, 10_000, 100_000]
    iterations: int = 200


class IngestRequest(BaseModel):
    path: str
    max_images: int = 500
