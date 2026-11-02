"""High-accuracy backend: SCRFD detection + ArcFace (w600k_r50) recognition,
both served straight from ONNX Runtime with 5-point landmark alignment.

This is the "serious production" option: 512-d embeddings, noticeably better
on pose/lighting changes than SFace, at the cost of ~190 MB of models and more
CPU.  Disabled by default; enable with ``FACEMATCH_EMBEDDER=arcface`` after
running ``scripts/download_models.py --pack buffalo_l``.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from ..config import Settings
from ..imaging import Box, l2_normalize
from ..pipeline import FaceDetection, models_path, register

DET_MODEL = "det_10g.onnx"
REC_MODEL = "w600k_r50.onnx"
STRIDES = (8, 16, 32)
NUM_ANCHORS = 2
#: ArcFace standard 112x112 reference landmarks
REF_KPS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@register("arcface")
class ArcFace:
    name = "arcface"
    dim = 512
    default_threshold = 0.40
    description = "SCRFD detector + ArcFace w600k_r50 recognizer (512-d, aligned, higher accuracy)"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._det_sess = None
        self._rec_sess = None
        self._det_size = (640, 640)

    @classmethod
    def available(cls) -> bool:
        from ..config import get_settings

        s = get_settings()
        return models_path(s, DET_MODEL).exists() and models_path(s, REC_MODEL).exists()

    # ------------------------------------------------------------------ load
    def _ensure(self) -> None:
        if self._det_sess is not None:
            return
        with self._lock:
            if self._det_sess is not None:
                return
            import onnxruntime as ort

            det_path = models_path(self.settings, DET_MODEL)
            rec_path = models_path(self.settings, REC_MODEL)
            if not det_path.exists() or not rec_path.exists():
                raise FileNotFoundError(
                    f"missing buffalo_l models in {self.settings.models_dir}. "
                    "Run: python scripts/download_models.py --pack buffalo_l"
                )
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._det_sess = ort.InferenceSession(str(det_path), opts, providers=["CPUExecutionProvider"])
            self._rec_sess = ort.InferenceSession(str(rec_path), opts, providers=["CPUExecutionProvider"])

    def warmup(self) -> None:
        self._ensure()
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        self.embed(dummy, FaceDetection(Box(0, 0, 112, 112), 1.0))

    # ------------------------------------------------------------- detection
    def detect(self, img: np.ndarray) -> list[FaceDetection]:
        """SCRFD decode — mirrors insightface's reference implementation
        (anchor centres at ``grid * stride``, kps = centre + prediction)."""
        self._ensure()
        det_w, det_h = self._det_size
        im_ratio = float(img.shape[0]) / img.shape[1]
        if im_ratio > det_h / det_w:
            new_h, new_w = det_h, int(det_h / im_ratio)
        else:
            new_w, new_h = det_w, int(det_w * im_ratio)
        resized = cv2.resize(img, (new_w, new_h))
        det_img = np.zeros((det_h, det_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized
        scale_y = float(img.shape[0]) / new_h  # back to source coordinates
        scale_x = float(img.shape[1]) / new_w

        blob = det_img.astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[None]

        with self._lock:
            outs = self._det_sess.run(None, {self._det_sess.get_inputs()[0].name: blob})

        thr = float(self.settings.detection_score_threshold)
        fmc = len(STRIDES)
        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(STRIDES):
            scores = outs[idx].reshape(-1)
            bbox_preds = outs[idx + fmc].reshape(-1, 4) * stride
            kps_preds = outs[idx + 2 * fmc].reshape(-1, 10) * stride
            h, w = det_h // stride, det_w // stride
            ys, xs = np.mgrid[:h, :w]
            centers = np.stack([xs, ys], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape(-1, 2)
            if NUM_ANCHORS > 1:
                centers = np.stack([centers] * NUM_ANCHORS, axis=1).reshape(-1, 2)
            pos = np.where(scores >= thr)[0]
            if pos.size == 0:
                continue
            c, b, k = centers[pos], bbox_preds[pos], kps_preds[pos]
            x1, y1 = c[:, 0] - b[:, 0], c[:, 1] - b[:, 1]
            x2, y2 = c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]
            bboxes_list.append(np.stack([x1, y1, x2, y2], axis=-1))
            preds = []
            for i in range(0, 10, 2):
                preds.append(c[:, i % 2] + k[:, i])
                preds.append(c[:, i % 2 + 1] + k[:, i + 1])
            kpss_list.append(np.stack(preds, axis=-1))
            scores_list.append(scores[pos])

        if not scores_list:
            return []
        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list)   # xyxy in detection coords
        kpss = np.concatenate(kpss_list)
        xywh = np.stack(
            [bboxes[:, 0], bboxes[:, 1], bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]],
            axis=-1,
        )
        keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), thr, 0.4)
        keep = np.asarray(keep).ravel() if len(keep) else np.array([], dtype=int)
        out: list[FaceDetection] = []
        for i in keep:
            x1, y1, x2, y2 = bboxes[i]
            kps = (kpss[i].reshape(5, 2) * np.array([scale_x, scale_y])).ravel().tolist()
            out.append(
                FaceDetection(
                    Box(float(x1 * scale_x), float(y1 * scale_y),
                        float((x2 - x1) * scale_x), float((y2 - y1) * scale_y)),
                    float(scores[i]),
                    kps,
                )
            )
        out.sort(key=lambda d: d.score, reverse=True)
        return out

    # ------------------------------------------------------------ recognition
    def embed(self, img: np.ndarray, detection: FaceDetection) -> np.ndarray:
        self._ensure()
        kps_flat = np.asarray(detection.keypoints, dtype=np.float32).ravel()
        if kps_flat.size == 10 and np.count_nonzero(kps_flat) > 0:
            kps = kps_flat.reshape(5, 2)
            mat, _ = cv2.estimateAffinePartial2D(kps, REF_KPS)
            if mat is None:  # pragma: no cover
                mat = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            aligned = cv2.warpAffine(img, mat, (112, 112), borderValue=0)
        else:  # pragma: no cover - fallback: centre crop of the box
            from ..imaging import crop_face

            aligned = cv2.resize(crop_face(img, detection.box), (112, 112))
        blob = aligned.astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[None]
        with self._lock:
            feat = self._rec_sess.run(None, {self._rec_sess.get_inputs()[0].name: blob})[0]
        return l2_normalize(np.asarray(feat, dtype=np.float32).ravel())
