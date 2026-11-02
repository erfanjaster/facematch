"""Vector index behaviour (numpy engine + auto/faiss switch)."""

from __future__ import annotations

import numpy as np

from facematch.index import VectorIndex


def _rand(n, dim, seed):
    mat = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)


def test_add_search_remove():
    idx = VectorIndex(dim=32)
    mat = _rand(50, 32, 0)
    idx.add_many([(f"f{i}", mat[i]) for i in range(50)])
    hits = idx.search(mat[7], 5)
    assert hits[0].face_id == "f7"
    assert hits[0].similarity > 0.999
    idx.remove({"f7"})
    hits = idx.search(mat[7], 5)
    assert all(h.face_id != "f7" for h in hits)
    assert len(idx) == 49


def test_faiss_and_numpy_agree():
    pytest_faiss = np.random.default_rng(1)
    dim = 64
    mat = _rand(500, dim, 2)
    a = VectorIndex(dim=dim, prefer="numpy")
    b = VectorIndex(dim=dim, prefer="faiss")
    for i in range(500):
        a.add(f"f{i}", mat[i])
        b.add(f"f{i}", mat[i])
    if b.engine != "faiss":  # faiss not installed in this env
        return
    q = _rand(1, dim, 3)[0]
    ha, hb = a.search(q, 10), b.search(q, 10)
    assert [h.face_id for h in ha] == [h.face_id for h in hb]
    assert np.allclose([h.similarity for h in ha], [h.similarity for h in hb], atol=1e-4)


def test_empty_index_search():
    idx = VectorIndex(dim=16)
    assert idx.search(np.ones(16, np.float32), 3) == []
    assert idx.engine == "empty"
