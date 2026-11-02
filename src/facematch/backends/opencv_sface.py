"""Default backend: OpenCV YuNet (detection) + SFace (recognition).

Tiny ONNX models (~39 MB total), CPU friendly, no GPU required. Cosine
similarity threshold recommended by the OpenCV model zoo: 0.363.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from ..config import Settings
from ..imaging import Box, l2_normalize
from ..pipeline import FaceDetection, models_path, register

DET_MODEL = "face_detection_yunet_2023mar.onnx"
REC_MODEL = "face_recognition_sface_2021dec.onnx"


@register("opencv_sface")
class OpenCVSFace:
    name = "opencv_sface"
    dim = 128
    default_threshold = 0.363
    description = "OpenCV YuNet detector + SFace recognizer (128-d, CPU, ~40 MB of ONNX)"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._det = None
        self._rec = None

    # ------------------------------------------------------------ availability
    @classmethod
    def available(cls) -> bool:
        from ..config import get_settings

        s = get_settings()
        return models_path(s, DET_MODEL).exists() and models_path(s, REC_MODEL).exists()

    # -------------------------------------------------------------- lazy load
    def _ensure(self) -> None:
        if self._det is not None and self._rec is not None:
            return
        with self._lock:
            if self._det is not None:
                return
            det_path = models_path(self.settings, DET_MODEL)
            rec_path = models_path(self.settings, REC_MODEL)
            if not det_path.exists() or not rec_path.exists():
                raise FileNotFoundError(
                    f"missing ONNX models in {self.settings.models_dir}. "
                    "Run: python scripts/download_models.py"
                )
            self._det = cv2.FaceDetectorYN.create(
                str(det_path),
                "",
                (320, 320),
                self.settings.detection_score_threshold,
                0.3,
                5000,
            )
            self._rec = cv2.FaceRecognizerSF.create(str(rec_path), "")

    def warmup(self) -> None:
        self._ensure()
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        face = FaceDetection(Box(0, 0, 112, 112), 1.0)
        self.embed(dummy, face)

    # ------------------------------------------------------------------ API
    def detect(self, img: np.ndarray) -> list[FaceDetection]:
        self._ensure()
        h, w = img.shape[:2]
        with self._lock:
            self._det.setInputSize((w, h))
            self._det.setScoreThreshold(self.settings.detection_score_threshold)
            result = self._det.detect(img)
        # OpenCV >=5 returns (retval, faces); older bindings return faces only
        faces = result[1] if isinstance(result, tuple) else result
        out: list[FaceDetection] = []
        if faces is None:
            return out
        for row in faces:
            x, y, bw, bh = (float(v) for v in row[:4])
            score = float(row[14])
            kps = [float(v) for v in row[4:14]]
            out.append(FaceDetection(Box(x, y, bw, bh), score, kps))
        out.sort(key=lambda d: d.score, reverse=True)
        return out

    def embed(self, img: np.ndarray, detection: FaceDetection) -> np.ndarray:
        self._ensure()
        row = np.array(
            [detection.box.x, detection.box.y, detection.box.w, detection.box.h]
            + detection.keypoints[:10]
            + [detection.score],
            dtype=np.float32,
        )
        with self._lock:
            aligned = self._rec.alignCrop(img, row)
            feat = self._rec.feature(aligned)
        return l2_normalize(np.asarray(feat, dtype=np.float32).ravel())
