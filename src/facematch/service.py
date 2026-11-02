"""Service layer: glue between the embedder, the vector index and the store.

All heavy thinking happens here; the HTTP layer stays thin.
"""

from __future__ import annotations

import base64
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Settings
from .imaging import (
    Box,
    ImageError,
    annotate,
    crop_face,
    decode_image,
    downscale_for_detection,
    encode_jpeg,
    thumbnail,
)
from .index import VectorIndex
from .pipeline import Embedder, FaceDetection, build_embedder
from .stores import Store, build_store


@dataclass
class QueryFaceResult:
    face_index: int
    bbox: list[float]
    score: float
    crop_thumb_b64: str
    matched: bool
    best_similarity: float | None
    candidates: list[dict] = field(default_factory=list)


@dataclass
class SearchResult:
    faces: list[QueryFaceResult]
    timings_ms: dict[str, float]
    index_stats: dict
    threshold: float
    embedder: str


class FaceMatchService:
    """Owns embedder + store + index; exposes enroll/search/admin operations."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder: Embedder = build_embedder(settings)
        self.store: Store = build_store(settings.store, settings)
        self.index = VectorIndex(
            dim=self.embedder.dim or 512,
            prefer=settings.index_backend,
            faiss_min_vectors=settings.faiss_min_vectors,
        )
        self._lock = threading.RLock()
        self._started = False

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._started:
            return
        self.store.init()
        self.embedder.warmup()
        self.reindex()
        self._started = True

    def reindex(self) -> dict:
        """Rebuild the in-memory vector index from the store (same embedder)."""
        t0 = time.perf_counter()
        self.index = VectorIndex(
            dim=self.embedder.dim or 512,
            prefer=self.settings.index_backend,
            faiss_min_vectors=self.settings.faiss_min_vectors,
        )
        pairs = list(self.store.iter_embeddings(self.embedder.name))
        if pairs:
            self.index.add_many(pairs)
        return {"vectors": len(pairs), "took_ms": round((time.perf_counter() - t0) * 1000, 2)}

    # ------------------------------------------------------------------- stats
    def stats(self) -> dict:
        health = self.store.health()
        return {
            "store": health,
            "index": {**self.index.stats(), "embedder": self.embedder.name},
            "embedder": {
                "name": self.embedder.name,
                "dim": self.embedder.dim,
                "description": getattr(self.embedder, "description", ""),
            },
            "threshold": self.settings.resolved_match_threshold,
        }

    # ------------------------------------------------------------------ enroll
    def enroll(
        self,
        image_bytes: bytes,
        person_name: str,
        create_person: bool = True,
        source_name: str | None = None,
    ) -> dict:
        self.start()
        person = self.store.get_or_create_person(person_name) if create_person else None
        if person is None:
            persons = {p.name.lower(): p for p in self.store.list_persons()}
            person = persons.get(person_name.lower())
            if person is None:
                raise KeyError(f"unknown person '{person_name}' and create_person=False")
        timings: dict[str, float] = {}
        t_start = time.perf_counter()
        t0 = time.perf_counter()
        img = decode_image(image_bytes)
        timings["decode"] = time.perf_counter() - t0
        faces, timings["detect"], embeddings, timings["embed"] = self._detect_and_embed(img)
        added = []
        t0 = time.perf_counter()
        for det, vec in zip(faces, embeddings):
            crop = crop_face(img, det.box)
            rec = self.store.add_face(
                person.id,
                vec,
                self.embedder.name,
                image_bytes,
                encode_jpeg(crop),
                meta={
                    "bbox": det.bbox_list(),
                    "score": round(det.score, 4),
                    "source": source_name,
                    "image_w": img.shape[1],
                    "image_h": img.shape[0],
                },
            )
            with self._lock:
                self.index.add(rec.id, vec)
            added.append(
                {
                    "face_id": rec.id,
                    "bbox": det.bbox_list(),
                    "score": round(det.score, 4),
                    "crop_thumb_b64": _b64_thumb(crop),
                }
            )
        timings["store"] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - t_start
        return {
            "person": person.as_dict(),
            "faces": added,
            "timings_ms": {k: round(v * 1000, 2) for k, v in timings.items()},
        }

    # ------------------------------------------------------------------ search
    def search(self, image_bytes: bytes, top_k: int | None = None, threshold: float | None = None) -> SearchResult:
        self.start()
        top_k = top_k or self.settings.top_k
        thr = threshold if threshold is not None else self.settings.resolved_match_threshold
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        img = decode_image(image_bytes)
        timings["decode"] = time.perf_counter() - t0

        faces, timings["detect"], embeddings, timings["embed"] = self._detect_and_embed(img)

        t0 = time.perf_counter()
        results: list[QueryFaceResult] = []
        for i, (det, vec) in enumerate(zip(faces, embeddings)):
            hits, _ = self.index.timed_search(vec, top_k)
            candidates = []
            for hit in hits:
                rec = self.store.get_face(hit.face_id)
                if rec is None:
                    continue
                person = self.store.get_person(rec.person_id)
                candidates.append(
                    {
                        "face_id": rec.id,
                        "person_id": rec.person_id,
                        "person_name": person.name if person else "?",
                        "similarity": round(hit.similarity, 4),
                        "matched": hit.similarity >= thr,
                        "added_at": rec.added_at,
                        "meta": rec.meta,
                        "crop_url": f"/api/media/faces/{rec.id}/crop",
                        "image_url": f"/api/media/faces/{rec.id}/image",
                        "stored_bbox": rec.meta.get("bbox"),
                    }
                )
            best = candidates[0] if candidates else None
            results.append(
                QueryFaceResult(
                    face_index=i,
                    bbox=det.bbox_list(),
                    score=round(det.score, 4),
                    crop_thumb_b64=_b64_thumb(crop_face(img, det.box)),
                    matched=bool(best and best["similarity"] >= thr),
                    best_similarity=best["similarity"] if best else None,
                    candidates=candidates,
                )
            )
        timings["search"] = time.perf_counter() - t0
        timings["total"] = sum(timings.values())
        return SearchResult(
            faces=results,
            timings_ms={k: round(v * 1000, 2) for k, v in timings.items()},
            index_stats=self.index.stats(),
            threshold=thr,
            embedder=self.embedder.name,
        )

    # --------------------------------------------------------------- internals
    def _detect_and_embed(self, img: np.ndarray):
        """Detect on a (possibly downscaled) copy, embed there, then map the
        boxes back to full-resolution coordinates for crops/annotations."""
        small, scale = downscale_for_detection(img, self.settings.max_detection_edge)
        t0 = time.perf_counter()
        dets_small = self.embedder.detect(small)
        detect_ms = time.perf_counter() - t0
        t0 = time.perf_counter()
        embeddings = [self.embedder.embed(small, det) for det in dets_small]
        embed_ms = time.perf_counter() - t0
        if scale != 1.0:
            inv = 1.0 / scale
            dets = [
                FaceDetection(det.box.scaled(inv), det.score, [v * inv for v in det.keypoints])
                for det in dets_small
            ]
        else:
            dets = dets_small
        return dets, detect_ms, embeddings, embed_ms

    # ------------------------------------------------------------------- admin
    def ingest_directory(self, path: str, max_images: int = 500) -> dict:
        """Recursively enroll a directory tree: each sub-folder = one person,
        images directly inside ``path`` go to a person named after the file
        stem's parent... (kept simple: sub-folders only)."""
        from pathlib import Path as P

        root = P(path)
        if not root.exists():
            raise FileNotFoundError(path)
        report = {"persons": 0, "faces": 0, "skipped": []}
        person_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()] or [root]
        for pdir in person_dirs:
            person_name = pdir.name.replace("_", " ").title() if pdir != root else root.name
            imgs = sorted(
                [f for f in pdir.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
            )[:max_images]
            for img_path in imgs:
                try:
                    res = self.enroll(img_path.read_bytes(), person_name, source_name=img_path.name)
                    report["faces"] += len(res["faces"])
                    if not res["faces"]:
                        report["skipped"].append(f"{img_path.name}: no face detected")
                except ImageError as exc:
                    report["skipped"].append(f"{img_path.name}: {exc}")
        report["persons"] = len(self.store.list_persons())
        return report

    def annotated_image(self, face_id: str) -> tuple[bytes, list[Box]] | None:
        rec = self.store.get_face(face_id)
        raw = self.store.read_image(face_id, "image")
        if rec is None or raw is None:
            return None
        img = decode_image(raw)
        bbox = rec.meta.get("bbox")
        boxes = [Box(*bbox)] if bbox else []
        person = self.store.get_person(rec.person_id)
        return encode_jpeg(annotate(img, boxes, [person.name if person else ""])), boxes

    def benchmark(self, sizes: list[int] | None = None, iterations: int = 200) -> dict:
        """Synthetic-vector latency probe: proves the search layer scales.

        Real end-to-end latency is reported per request in ``timings_ms``.
        """
        sizes = sizes or [1_000, 10_000, 100_000]
        dim = self.embedder.dim or 128
        rng = np.random.default_rng(7)
        out: dict[str, Any] = {}
        for n in sizes:
            mat = rng.standard_normal((n, dim), dtype=np.float32)
            for i in range(0, n, 65_536):  # chunked in-place normalise (low memory)
                block = mat[i : i + 65_536]
                norms = np.sqrt(np.einsum("ij,ij->i", block, block), out=np.empty(block.shape[0], np.float32))
                norms[norms == 0] = 1
                block /= norms[:, None]
            idx = VectorIndex(dim, prefer=self.settings.index_backend,
                              faiss_min_vectors=self.settings.faiss_min_vectors)
            if n >= 500_000:
                idx.load_matrix([f"syn_{i}" for i in range(n)], mat)
            else:
                idx.add_many([(f"syn_{i}", mat[i]) for i in range(n)])
            queries = rng.standard_normal((iterations, dim), dtype=np.float32)
            queries /= np.linalg.norm(queries, axis=1, keepdims=True)
            lat = []
            for q in queries:
                _, ms = idx.timed_search(q, 5)
                lat.append(ms)
            out[str(n)] = {
                "engine": idx.engine,
                "p50_ms": round(statistics.median(lat), 4),
                "p95_ms": round(sorted(lat)[int(0.95 * len(lat))], 4),
                "mean_ms": round(statistics.mean(lat), 4),
                "queries_per_sec": round(1000.0 / statistics.mean(lat), 1),
            }
        return {"dim": dim, "iterations": iterations, "results": out}


def _b64_thumb(img: np.ndarray, max_edge: int = 190) -> str:
    return base64.b64encode(encode_jpeg(thumbnail(img, max_edge), quality=80)).decode()
