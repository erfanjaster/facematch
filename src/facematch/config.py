"""Configuration for FaceMatch.

Everything is driven by environment variables (or a ``.env`` file) with the
``FACEMATCH_`` prefix so the same code can run on a laptop, in Docker or on a
server without edits.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. See ``.env.example`` for documentation."""

    model_config = SettingsConfigDict(
        env_prefix="FACEMATCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    # ---------------------------------------------------------------- pipeline
    #: Which embedding backend to use. Registered names live in
    #: ``facematch.backends.REGISTRY``.
    embedder: str = "opencv_sface"
    #: Directory holding the ONNX models (downloaded by scripts/download_models.py)
    models_dir: Path = Path("models")
    #: Face detector confidence floor (0..1).
    detection_score_threshold: float = 0.6
    #: Cosine similarity needed to call two faces "the same person".
    #: ``0`` means "use the backend's recommended default".
    match_threshold: float = 0.0
    #: How many candidates to return per detected face.
    top_k: int = 5
    #: Reject uploads bigger than this (decodes are memory-hungry).
    max_upload_mb: float = 25.0
    #: Downscale huge uploads before detection (keeps latency predictable).
    max_detection_edge: int = 1600

    # ------------------------------------------------------------------- index
    #: Vector index flavour. "numpy" = exact BLAS dot-product (fastest exact
    #: option on typical CPUs up to ~1M vectors), "faiss" = FAISS IndexFlatIP,
    #: "auto" = numpy, upgrading to faiss past ``faiss_min_vectors``.
    index_backend: Literal["auto", "numpy", "faiss"] = "auto"
    faiss_min_vectors: int = 1_000_000

    # ------------------------------------------------------------------- store
    #: Storage adapter: "folder" (default, dev friendly), "sqlite", "http".
    store: str = "folder"
    #: Root directory used by the folder store (and sqlite file location).
    data_dir: Path = Path("data")
    #: Used by the sqlite store.
    sqlite_filename: str = "facematch.db"
    #: Used by the http store (point at another FaceMatch instance or any
    #: service implementing the same REST contract).
    store_url: str = ""
    store_api_key: str = ""

    # ------------------------------------------------- remote embedder backend
    #: Only used when ``embedder = remote_api``.
    remote_embed_url: str = ""
    remote_embed_api_key: str = ""
    remote_embed_dim: int = 512

    # ------------------------------------------------------------------- misc
    log_level: str = "INFO"
    cors_origins: str = "*"

    # ---------------------------------------------------------------- helpers
    @property
    def resolved_match_threshold(self) -> float:
        """Threshold actually used: explicit setting or backend default."""
        if self.match_threshold > 0:
            return self.match_threshold
        from .pipeline import get_embedder_class

        return get_embedder_class(self.embedder).default_threshold

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
