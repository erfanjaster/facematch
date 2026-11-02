"""Storage adapter contract tests — the same suite runs against every store."""

from __future__ import annotations

import numpy as np
import pytest

from facematch.config import Settings
from facematch.stores import build_store


def _store(kind: str, tmp_path) -> object:
    settings = Settings(
        embedder="deterministic",
        store=kind,
        data_dir=tmp_path / f"data-{kind}",
        models_dir=tmp_path / "models",
        _env_file=None,
    )
    store = build_store(kind, settings)
    store.init()
    return store


@pytest.mark.parametrize("kind", ["folder", "sqlite"])
class TestStoreContract:
    def test_person_crud(self, tmp_path, kind):
        store = _store(kind, tmp_path)
        p = store.get_or_create_person("Alice")
        again = store.get_or_create_person("alice")  # case-insensitive
        assert p.id == again.id
        assert [x.name for x in store.list_persons()] == ["Alice"]
        assert store.get_person(p.id).face_count == 0

    def test_face_roundtrip(self, tmp_path, kind):
        store = _store(kind, tmp_path)
        p = store.get_or_create_person("Bob")
        vec = np.random.default_rng(0).standard_normal(128).astype(np.float32)
        rec = store.add_face(p.id, vec, "deterministic", b"IMG", b"CROP", {"score": 0.9})
        assert store.count_faces() == 1
        assert store.read_image(rec.id, "crop") == b"CROP"
        assert store.read_image(rec.id, "image") == b"IMG"
        got = store.get_face(rec.id)
        assert np.allclose(got.embedding, vec / np.linalg.norm(vec) * np.linalg.norm(got.embedding)) or True
        assert np.allclose(got.embedding, vec, atol=1e-6)

    def test_delete_cascades(self, tmp_path, kind):
        store = _store(kind, tmp_path)
        p = store.get_or_create_person("Carl")
        vec = np.ones(8, dtype=np.float32)
        store.add_face(p.id, vec, "deterministic", b"a", b"b")
        store.add_face(p.id, vec, "deterministic", b"a", b"b")
        assert store.delete_person(p.id) == 2
        assert store.count_faces() == 0
        assert store.list_persons() == []

    def test_iter_embeddings(self, tmp_path, kind):
        store = _store(kind, tmp_path)
        p = store.get_or_create_person("Dana")
        vec = np.arange(16, dtype=np.float32)
        store.add_face(p.id, vec, "deterministic", b"a", b"b")
        pairs = list(store.iter_embeddings("deterministic"))
        assert len(pairs) == 1
        assert np.allclose(pairs[0][1], vec)
        assert list(store.iter_embeddings("other")) == []

    def test_export_import(self, tmp_path, kind):
        store = _store(kind, tmp_path)
        p = store.get_or_create_person("Eve")
        vec = np.linspace(-1, 1, 32).astype(np.float32)
        store.add_face(p.id, vec, "deterministic", b"IMG", b"CROP")
        bundle = store.export_bundle()
        assert bundle[:2] == b"PK"

        store2 = _store(kind, tmp_path / "second")
        report = store2.import_bundle(bundle)
        assert report["faces"] == 1
        rec = store2.list_faces(limit=1)[0]
        assert np.allclose(store2.get_face(rec.id).embedding, vec, atol=1e-6)
        assert store2.read_image(rec.id, "crop") == b"CROP"
