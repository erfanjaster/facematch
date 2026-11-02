"""Shared fixtures: a model-free service (deterministic embedder + tmp folder
store) so the whole suite runs offline in milliseconds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facematch.api import create_app  # noqa: E402
from facematch.config import Settings  # noqa: E402
from facematch.imaging import encode_jpeg  # noqa: E402
from facematch.service import FaceMatchService  # noqa: E402


def make_image(seed: int, size: int = 96, kind: str = "pattern") -> bytes:
    """Deterministic synthetic 'photo'."""
    rng = np.random.default_rng(seed)
    if kind == "pattern":
        xs = np.linspace(0, 8 * np.pi, size)
        field = np.sin(xs)[:, None] * np.cos(xs * (1 + seed % 3))[None, :]
        img = ((field + 1) * 127).astype(np.uint8)
        img = np.dstack([img, rng.integers(0, 255, (size, size), dtype=np.uint8), img // 2])
    else:
        img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return encode_jpeg(img)


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        embedder="deterministic",
        store="folder",
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        top_k=3,
        _env_file=None,
    )


@pytest.fixture()
def service(settings: Settings) -> FaceMatchService:
    svc = FaceMatchService(settings)
    svc.start()
    return svc


@pytest.fixture()
def client(settings: Settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def real_models_dir() -> Path | None:
    """Locate downloaded ONNX models (project models/ or FACEMATCH_MODELS_DIR)."""
    import os

    candidates = [Path(os.environ.get("FACEMATCH_MODELS_DIR", "")), Path("models"), Path("/opt/facematch/models")]
    for c in candidates:
        if c and (c / "face_detection_yunet_2023mar.onnx").exists():
            return c
    return None
