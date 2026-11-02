"""End-to-end matching semantics with the model-free backend."""

from __future__ import annotations

from conftest import make_image


def test_enroll_then_search_same_image_matches(service):
    img = make_image(1)
    res = service.enroll(img, "Alice")
    assert len(res["faces"]) == 1

    out = service.search(img)
    assert len(out.faces) == 1
    face = out.faces[0]
    assert face.matched is True
    assert face.candidates[0]["person_name"] == "Alice"
    assert face.candidates[0]["similarity"] > 0.99


def test_different_image_does_not_match(service):
    service.enroll(make_image(1), "Alice")
    out = service.search(make_image(2))
    face = out.faces[0]
    assert face.matched is False


def test_multiple_persons_ranked(service):
    service.enroll(make_image(10), "Alice")
    service.enroll(make_image(20), "Bob")
    out = service.search(make_image(10))
    assert out.faces[0].candidates[0]["person_name"] == "Alice"
    names = {c["person_name"] for c in out.faces[0].candidates}
    assert names == {"Alice", "Bob"}


def test_timings_reported(service):
    out = service.search(make_image(3))
    for key in ("decode", "detect", "embed", "search", "total"):
        assert key in out.timings_ms


def test_threshold_override(service):
    service.enroll(make_image(1), "Alice")
    out = service.search(make_image(2), threshold=0.0)  # everything matches
    assert out.faces[0].matched is True
