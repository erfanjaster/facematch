"""Storage adapter contract.

v1 ships a *folder* store (human-browsable, perfect for dev & tests) and a
*sqlite* store (single file, scales to ~100k faces).  An *http* store lets one
FaceMatch instance use another as its database (federation / future cloud
tier).  Because every consumer talks to this protocol, adding Postgres,
MongoDB or S3 later is a new file — not a rewrite.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator, Protocol, runtime_checkable

import numpy as np


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_ts() -> float:
    return time.time()


@dataclass(slots=True)
class Person:
    id: str
    name: str
    created_at: float
    meta: dict = field(default_factory=dict)
    face_count: int = 0

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "meta": self.meta,
            "face_count": self.face_count,
        }


@dataclass(slots=True)
class FaceRecord:
    id: str
    person_id: str
    embedder: str
    dim: int
    added_at: float
    meta: dict = field(default_factory=dict)  # bbox, score, source filename, ...
    embedding: np.ndarray | None = None

    def as_dict(self, include_embedding: bool = False) -> dict:
        d = {
            "id": self.id,
            "person_id": self.person_id,
            "embedder": self.embedder,
            "dim": self.dim,
            "added_at": self.added_at,
            "meta": self.meta,
        }
        if include_embedding and self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        return d


@runtime_checkable
class Store(Protocol):
    name: str
    description: str

    # -- lifecycle -----------------------------------------------------------
    def init(self) -> None: ...
    def close(self) -> None: ...
    def health(self) -> dict: ...

    # -- persons ---------------------------------------------------------------
    def get_or_create_person(self, name: str, meta: dict | None = None) -> Person: ...
    def list_persons(self) -> list[Person]: ...
    def get_person(self, person_id: str) -> Person | None: ...
    def delete_person(self, person_id: str) -> int: ...

    # -- faces -----------------------------------------------------------------
    def add_face(
        self,
        person_id: str,
        embedding: np.ndarray,
        embedder: str,
        image_bytes: bytes,
        crop_bytes: bytes,
        meta: dict | None = None,
    ) -> FaceRecord: ...
    def get_face(self, face_id: str) -> FaceRecord | None: ...
    def list_faces(self, person_id: str | None = None, limit: int = 100, offset: int = 0) -> list[FaceRecord]: ...
    def delete_face(self, face_id: str) -> bool: ...
    def read_image(self, face_id: str, kind: str = "crop") -> bytes | None: ...

    # -- bulk ------------------------------------------------------------------
    def iter_embeddings(self, embedder: str | None = None) -> Iterator[tuple[str, np.ndarray]]: ...
    def count_faces(self) -> int: ...
    def export_bundle(self) -> bytes: ...
    def import_bundle(self, payload: bytes) -> dict: ...


STORE_REGISTRY: dict[str, type] = {}


def register_store(name: str):
    def wrapper(cls):
        cls.name = name
        STORE_REGISTRY[name] = cls
        return cls

    return wrapper


def build_store(name: str, settings) -> Store:
    from . import folder_store, http_store, sqlite_store  # noqa: F401

    if name not in STORE_REGISTRY:
        raise KeyError(f"unknown store '{name}'. available: {sorted(STORE_REGISTRY)}")
    return STORE_REGISTRY[name](settings)  # type: ignore[call-arg]


def list_stores() -> list[dict]:
    from . import folder_store, http_store, sqlite_store  # noqa: F401

    return [
        {"name": name, "description": getattr(cls, "description", "")}
        for name, cls in sorted(STORE_REGISTRY.items())
    ]
