"""Deterministic, model-free backend used by the test-suite and for offline
development.  Detection returns a single whole-image "face"; the embedding is
a normalised gradient-signature (a high-dimensional dHash) so identical or
near-identical images match and different images do not.  Zero downloads.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import Settings
from ..imaging import Box, l2_normalize
from ..pipeline import FaceDetection, register


@register("deterministic")
class Deterministic:
    name = "deterministic"
    dim = 128
    default_threshold = 0.90
    description = "Model-free hash backend for tests / offline dev (not a real face matcher)"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @classmethod
    def available(cls) -> bool:
        return True

    def warmup(self) -> None:  # nothing to load
        return None

    def detect(self, img: np.ndarray) -> list[FaceDetection]:
        h, w = img.shape[:2]
        return [FaceDetection(Box(0, 0, w, h), 1.0)]

    def embed(self, img: np.ndarray, detection: FaceDetection) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        small = cv2.resize(gray, (17, 17), interpolation=cv2.INTER_AREA).astype(np.float32)
        # horizontal + vertical gradient signs -> 16*16 = 256 bits -> fold to 128
        gx = (small[1:, 1:] - small[1:, :-1]).ravel()
        gy = (small[1:, 1:] - small[:-1, 1:]).ravel()
        vec = np.concatenate([gx, gy])[: self.dim * 2]
        vec = vec[: self.dim] + np.roll(vec, self.dim)[: self.dim] * 0.5
        return l2_normalize(vec)
