"""Remote embedding backend — delegate detection+embedding to an HTTP API.

This is the extension point the roadmap promised: point FaceMatch at any
cloud face service that exposes a simple JSON contract and the rest of the
system (index, stores, UI) keeps working unchanged.

Contract (POST ``image/jpeg`` body to ``FACEMATCH_REMOTE_EMBED_URL``)::

    -> 200 {"faces": [{"bbox": [x, y, w, h], "score": 0.99,
                      "embedding": [0.01, ...]}]}

A service may also accept ``{"image_b64": ...}``; both are tried in order.
"""

from __future__ import annotations

import base64

import httpx
import numpy as np

from ..config import Settings
from ..imaging import Box, l2_normalize
from ..pipeline import FaceDetection, register


@register("remote_api")
class RemoteAPI:
    name = "remote_api"
    dim = 0  # resolved on first response
    default_threshold = 0.5
    description = "Delegate detection+embedding to an external HTTP face API"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.remote_embed_url:
            raise ValueError("embedder=remote_api requires FACEMATCH_REMOTE_EMBED_URL")
        self._client = httpx.Client(timeout=30.0, headers=self._headers())

    def _headers(self) -> dict[str, str]:
        h = {}
        if self.settings.remote_embed_api_key:
            h["Authorization"] = f"Bearer {self.settings.remote_embed_api_key}"
        return h

    @classmethod
    def available(cls) -> bool:
        from ..config import get_settings

        return bool(get_settings().remote_embed_url)

    def warmup(self) -> None:
        return None

    def _post(self, img_bytes: bytes) -> dict:
        from ..imaging import encode_jpeg  # local import to avoid cycles

        try:
            r = self._client.post(
                self.settings.remote_embed_url,
                content=img_bytes,
                headers={"Content-Type": "image/jpeg"},
            )
            if r.status_code == 415:
                raise httpx.HTTPError("retry with json")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError:
            r = self._client.post(
                self.settings.remote_embed_url,
                json={"image_b64": base64.b64encode(img_bytes).decode()},
            )
            r.raise_for_status()
            return r.json()

    def detect_and_embed(self, img: np.ndarray, img_bytes: bytes) -> list[FaceDetection]:
        data = self._post(img_bytes)
        dets = []
        for face in data.get("faces", []):
            x, y, w, h = face["bbox"]
            emb = np.asarray(face["embedding"], dtype=np.float32)
            if self.dim == 0:
                type(self).dim = int(emb.size)
            det = FaceDetection(Box(x, y, w, h), float(face.get("score", 1.0)))
            det.embedding = l2_normalize(emb)  # type: ignore[attr-defined]
            dets.append(det)
        return dets

    # Protocol methods ------------------------------------------------------
    def detect(self, img: np.ndarray) -> list[FaceDetection]:
        from ..imaging import encode_jpeg

        return self.detect_and_embed(img, encode_jpeg(img))

    def embed(self, img: np.ndarray, detection: FaceDetection) -> np.ndarray:
        emb = getattr(detection, "embedding", None)
        if emb is None:  # pragma: no cover - defensive
            emb = self.detect(img)[0].embedding
        return emb
