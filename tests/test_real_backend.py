"""Integration test against the real ONNX backend — skipped automatically when
the models have not been downloaded (CI without network stays green)."""

from __future__ import annotations

from pathlib import Path

import pytest

from facematch.config import Settings
from facematch.service import FaceMatchService

DEMO = Path(__file__).resolve().parents[1] / "demo_images"
QUERIES = Path(__file__).resolve().parents[1] / "queries"


@pytest.fixture()
def real_service(tmp_path, real_models_dir):
    if real_models_dir is None:
        pytest.skip("ONNX models not downloaded (run scripts/download_models.py)")
    if not DEMO.exists():
        pytest.skip("demo images missing")
    settings = Settings(
        embedder="opencv_sface",
        store="folder",
        data_dir=tmp_path / "data",
        models_dir=real_models_dir,
        top_k=4,
        _env_file=None,
    )
    svc = FaceMatchService(settings)
    svc.start()
    return svc


def test_same_person_different_photo_matches(real_service):
    svc = real_service
    svc.enroll((DEMO / "barack_obama" / "01.jpg").read_bytes(), "Obama")
    svc.enroll((DEMO / "man_bw" / "01.jpg").read_bytes(), "Stranger")

    # a *different* photo of the same person must match...
    res = svc.search((DEMO / "barack_obama" / "02.jpg").read_bytes())
    assert res.faces, "no face detected"
    face = res.faces[0]
    assert face.matched, f"expected match, best={face.best_similarity}"
    assert face.candidates[0]["person_name"] == "Obama"

    # ...while an unrelated person must not
    res = svc.search((QUERIES / "stranger.jpg").read_bytes())
    face = res.faces[0]
    assert not face.matched, f"false positive: {face.candidates[0]}"


def test_hard_query_small_face(real_service):
    svc = real_service
    svc.enroll((DEMO / "barack_obama" / "01.jpg").read_bytes(), "Obama")
    res = svc.search((QUERIES / "obama_hard.jpg").read_bytes())
    assert res.faces
    assert res.faces[0].matched
    assert res.faces[0].candidates[0]["person_name"] == "Obama"
