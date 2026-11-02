"""Vector index layer.

Search has to be *fast*: embeddings live in RAM as a contiguous float32
matrix and queries are answered with a single BLAS dot-product.  Past
``faiss_min_vectors`` the same matrix is handed to a FAISS ``IndexFlatIP``
session (exact inner product, SIMD optimised) — accuracy is identical,
constant factors improve on wide matrices.

Vectors from different embedders are incompatible, so the service keeps one
index *per embedder name*.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Hit:
    face_id: str
    similarity: float


class VectorIndex:
    """Append/remove vector store with cosine search (vectors are L2-normalised
    on insert, so cosine == inner product)."""

    def __init__(self, dim: int, prefer: str = "auto", faiss_min_vectors: int = 20_000) -> None:
        self.dim = dim
        self.prefer = prefer
        self.faiss_min_vectors = faiss_min_vectors
        self._ids: list[str] = []
        self._vecs: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None
        self._dirty = True
        self._faiss = None
        self._loaded = False
        self._lock = threading.RLock()

    # ------------------------------------------------------------- mutations
    def add(self, face_id: str, vec: np.ndarray) -> None:
        with self._lock:
            self._materialise_rows()
            self._ids.append(face_id)
            self._vecs.append(np.asarray(vec, dtype=np.float32).ravel())
            self._dirty = True
            self._faiss = None  # invalidate faiss session

    def add_many(self, pairs: list[tuple[str, np.ndarray]]) -> None:
        with self._lock:
            self._materialise_rows()
            for fid, vec in pairs:
                self._ids.append(fid)
                self._vecs.append(np.asarray(vec, dtype=np.float32).ravel())
            self._dirty = True
            self._faiss = None

    def load_matrix(self, ids: list[str], matrix: np.ndarray) -> None:
        """Bulk-load a prebuilt contiguous matrix *without copying* — used by
        the benchmark harness for million-vector probes.  add()/remove() still
        work afterwards (rows are materialised lazily)."""
        with self._lock:
            mat = np.ascontiguousarray(matrix, dtype=np.float32)
            self._ids = list(ids)
            self._vecs = []
            self._matrix = mat
            self._dirty = False
            self._faiss = None
            self._loaded = True

    def _materialise_rows(self) -> None:
        if getattr(self, "_loaded", False) and not self._vecs and self._matrix is not None:
            self._vecs = [self._matrix[i] for i in range(self._matrix.shape[0])]
            self._loaded = False

    def remove(self, face_ids: set[str]) -> int:
        with self._lock:
            self._materialise_rows()
            if not face_ids:
                return 0
            keep = [(i, v) for i, (fid, v) in enumerate(zip(self._ids, self._vecs)) if fid not in face_ids]
            removed = len(self._ids) - len(keep)
            self._ids = [self._ids[i] for i, _ in keep]
            self._vecs = [v for _, v in keep]
            self._dirty = True
            self._faiss = None
            return removed

    def clear(self) -> None:
        with self._lock:
            self._ids.clear()
            self._vecs.clear()
            self._matrix = None
            self._faiss = None
            self._dirty = False

    # ----------------------------------------------------------------- search
    def _ensure_matrix(self) -> np.ndarray | None:
        if self._dirty:
            if self._vecs:
                self._matrix = np.ascontiguousarray(np.stack(self._vecs), dtype=np.float32)
            else:
                self._matrix = None
            self._dirty = False
        return self._matrix

    def _ensure_faiss(self, matrix: np.ndarray):
        if self._faiss is not None:
            return self._faiss
        use = self.prefer == "faiss" or (
            self.prefer == "auto" and matrix.shape[0] >= self.faiss_min_vectors
        )
        if not use:
            return None
        try:
            import faiss
        except Exception:  # pragma: no cover - faiss optional
            return None
        index = faiss.IndexFlatIP(self.dim)
        index.add(matrix)
        self._faiss = index
        return index

    @property
    def engine(self) -> str:
        with self._lock:
            matrix = self._ensure_matrix()
            if matrix is None:
                return "empty"
            return "faiss" if self._ensure_faiss(matrix) is not None else "numpy"

    def search(self, vec: np.ndarray, k: int = 5) -> list[Hit]:
        with self._lock:
            matrix = self._ensure_matrix()
            if matrix is None or k <= 0:
                return []
            q = np.ascontiguousarray(np.asarray(vec, dtype=np.float32).ravel()[None])
            k = min(k, matrix.shape[0])
            faiss_index = self._ensure_faiss(matrix)
            if faiss_index is not None:
                sims, idx = faiss_index.search(q, k)
                sims, idx = sims[0], idx[0]
            else:
                sims = matrix @ q.T
                sims = sims.ravel()
                idx = np.argpartition(-sims, k - 1)[:k] if matrix.shape[0] > k else np.arange(matrix.shape[0])
                order = np.argsort(-sims[idx])
                idx = idx[order]
                sims = sims[idx]
            hits = []
            for i, s in zip(idx, sims):
                if i < 0:
                    continue
                hits.append(Hit(face_id=self._ids[int(i)], similarity=float(s)))
            return hits

    def timed_search(self, vec: np.ndarray, k: int = 5) -> tuple[list[Hit], float]:
        t0 = time.perf_counter()
        hits = self.search(vec, k)
        return hits, (time.perf_counter() - t0) * 1000.0

    # ------------------------------------------------------------------ misc
    def __len__(self) -> int:
        return len(self._ids)

    def stats(self) -> dict:
        with self._lock:
            matrix = self._ensure_matrix()
            return {
                "engine": self.engine,
                "vectors": len(self._ids),
                "dim": self.dim,
                "matrix_mb": round(matrix.nbytes / 1e6, 2) if matrix is not None else 0.0,
            }
