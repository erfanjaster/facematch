"""Embedding backend protocol + registry.

A *backend* knows how to (a) locate faces inside an image and (b) turn an
aligned face into a fixed-length embedding vector.  Everything downstream
(index, stores, API) only speaks "float32 vector", which is what makes the
system pluggable: swap OpenCV SFace for ArcFace, or for a cloud API, without
touching anything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .config import Settings
from .imaging import Box


@dataclass(slots=True)
class FaceDetection:
    box: Box
    score: float
    keypoints: list[float] = field(default_factory=list)  # 5 x (x, y) flattened

    def bbox_list(self) -> list[float]:
        return self.box.as_list()


@runtime_checkable
class Embedder(Protocol):
    """Contract every embedding backend must fulfil."""

    name: str
    dim: int
    #: cosine similarity at which two embeddings are considered same-identity
    default_threshold: float
    #: human readable blurb shown in the UI / /api/config
    description: str

    def detect(self, img: np.ndarray) -> list[FaceDetection]:
        """Return faces found in ``img`` (BGR uint8)."""
        ...

    def embed(self, img: np.ndarray, detection: FaceDetection) -> np.ndarray:
        """Return an L2-normalised float32 vector of length ``dim``."""
        ...

    def warmup(self) -> None:
        """Load models / run a dummy forward pass so the first real request
        is not slow."""
        ...


REGISTRY: dict[str, type] = {}


def register(name: str):
    """Class decorator adding a backend to the registry."""

    def wrapper(cls):
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return wrapper


def get_embedder_class(name: str) -> type:
    from . import backends  # noqa: F401  (triggers registrations)

    if name not in REGISTRY:
        raise KeyError(f"unknown embedder '{name}'. available: {sorted(REGISTRY)}")
    return REGISTRY[name]


def build_embedder(settings: Settings) -> Embedder:
    cls = get_embedder_class(settings.embedder)
    return cls(settings)  # type: ignore[call-arg]


def list_backends() -> list[dict]:
    from . import backends  # noqa: F401

    out = []
    for name, cls in sorted(REGISTRY.items()):
        out.append(
            {
                "name": name,
                "dim": getattr(cls, "dim", None),
                "default_threshold": getattr(cls, "default_threshold", None),
                "description": getattr(cls, "description", ""),
                "available": bool(cls.available()),
            }
        )
    return out


def models_path(settings: Settings, filename: str) -> Path:
    return Path(settings.models_dir) / filename
