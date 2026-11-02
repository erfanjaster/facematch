"""Image decoding / encoding / cropping helpers (pure OpenCV + Pillow)."""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImageError(ValueError):
    """Raised when an uploaded payload cannot be decoded as an image."""


@dataclass(slots=True)
class Box:
    """Axis aligned bounding box in pixel coordinates of the source image."""

    x: float
    y: float
    w: float
    h: float

    def as_list(self) -> list[float]:
        return [round(self.x, 1), round(self.y, 1), round(self.w, 1), round(self.h, 1)]

    def scaled(self, factor: float) -> "Box":
        return Box(self.x * factor, self.y * factor, self.w * factor, self.h * factor)


def decode_image(payload: bytes) -> np.ndarray:
    """Decode bytes into a BGR uint8 array, honouring EXIF orientation."""
    if not payload:
        raise ImageError("empty payload")
    # Pillow first: handles EXIF rotation + exotic containers.
    try:
        with Image.open(io.BytesIO(payload)) as pil:
            pil = ImageOps.exif_transpose(pil)
            rgb = pil.convert("RGB")
            arr = np.asarray(rgb, dtype=np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:  # pragma: no cover - fallback path
        buf = np.frombuffer(payload, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ImageError("could not decode image bytes")
        return img


def encode_jpeg(img: np.ndarray, quality: int = 88) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:  # pragma: no cover
        raise ImageError("jpeg encoding failed")
    return buf.tobytes()


def downscale_for_detection(img: np.ndarray, max_edge: int) -> tuple[np.ndarray, float]:
    """Return (image, scale) where scale maps detection coords back to source."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return img, 1.0
    factor = max_edge / float(longest)
    small = cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)
    return small, factor


def crop_face(img: np.ndarray, box: Box, margin: float = 0.28) -> np.ndarray:
    """Crop a face with a margin so hair/chin context survives (for display)."""
    h, w = img.shape[:2]
    pad_x = box.w * margin
    pad_y = box.h * margin
    x0 = max(0, int(box.x - pad_x))
    y0 = max(0, int(box.y - pad_y))
    x1 = min(w, int(box.x + box.w + pad_x))
    y1 = min(h, int(box.y + box.h + pad_y))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        raise ImageError("face box falls outside of the image")
    return crop


def thumbnail(img: np.ndarray, max_edge: int = 220) -> np.ndarray:
    h, w = img.shape[:2]
    factor = min(1.0, max_edge / float(max(h, w)))
    if factor >= 1.0:
        return img
    return cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)


def annotate(img: np.ndarray, boxes: list[Box], labels: list[str] | None = None) -> np.ndarray:
    """Draw labelled rectangles — used to highlight the matched face in the
    *original* stored photo (the user asked for the face, but seeing where it
    was found is reassuring)."""
    out = img.copy()
    labels = labels or [""] * len(boxes)
    for box, label in zip(boxes, labels):
        p0 = (int(box.x), int(box.y))
        p1 = (int(box.x + box.w), int(box.y + box.h))
        cv2.rectangle(out, p0, p1, (0, 220, 120), max(2, int(min(out.shape[:2]) / 250)))
        if label:
            font_scale = max(0.45, min(out.shape[:2]) / 1400)
            cv2.putText(out, label, (p0[0], max(12, p0[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 220, 120), 1, cv2.LINE_AA)
    return out


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).ravel()
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm
