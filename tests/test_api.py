"""HTTP API surface tests."""

from __future__ import annotations

import base64

from conftest import make_image


def test_health_and_config(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok"
    cfg = client.get("/api/config").json()
    assert cfg["embedder"] == "deterministic"
    assert any(b["name"] == "opencv_sface" for b in cfg["backends"])


def test_enroll_search_flow(client):
    img = make_image(7)
    r = client.post(
        "/api/enroll",
        files={"file": ("a.jpg", img, "image/jpeg")},
        data={"person": "Zoe"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["person"]["name"] == "Zoe"
    assert len(body["faces"]) == 1

    r = client.post("/api/search", files={"file": ("q.jpg", img, "image/jpeg")})
    assert r.status_code == 200
    data = r.json()
    assert data["faces"][0]["matched"] is True
    assert data["faces"][0]["candidates"][0]["person_name"] == "Zoe"
    assert data["timings_ms"]["total"] >= 0


def test_search_json_payload(client):
    img = make_image(8)
    client.post("/api/enroll/json", json={"image_b64": base64.b64encode(img).decode(), "person": "Yan"})
    r = client.post("/api/search/json", json={"image_b64": base64.b64encode(img).decode()})
    assert r.json()["faces"][0]["matched"] is True


def test_search_bad_image(client):
    r = client.post("/api/search", files={"file": ("q.bin", b"not an image", "application/octet-stream")})
    assert r.status_code == 422


def test_persons_and_media(client):
    img = make_image(9)
    client.post("/api/enroll", files={"file": ("a.jpg", img, "image/jpeg")}, data={"person": "Max"})
    persons = client.get("/api/persons").json()["persons"]
    pid = persons[0]["id"]
    faces = client.get(f"/api/persons/{pid}/faces").json()["faces"]
    fid = faces[0]["id"]

    crop = client.get(f"/api/media/faces/{fid}/crop")
    assert crop.status_code == 200 and crop.headers["content-type"] == "image/jpeg"
    annotated = client.get(f"/api/media/faces/{fid}/annotated")
    assert annotated.status_code == 200

    assert client.delete(f"/api/faces/{fid}").json()["deleted"] is True
    assert client.get(f"/api/media/faces/{fid}/crop").status_code == 404
    assert client.delete(f"/api/persons/{pid}").json()["deleted_faces"] == 0


def test_raw_face_insert_and_embeddings(client):
    person = client.post("/api/persons", json={"name": "Raw"}).json()
    emb = [0.1] * 128
    r = client.post(
        "/api/faces",
        json={
            "person_id": person["id"],
            "embedding": emb,
            "embedder": "deterministic",
            "image_b64": base64.b64encode(make_image(11)).decode(),
        },
    )
    assert r.status_code == 201
    dump = client.get("/api/embeddings").json()
    assert len(dump["faces"]) == 1


def test_benchmark_endpoint(client):
    r = client.post("/api/admin/benchmark", json={"sizes": [500], "iterations": 20})
    assert r.status_code == 200
    assert "500" in r.json()["results"]


def test_export_import_roundtrip(client):
    img = make_image(12)
    client.post("/api/enroll", files={"file": ("a.jpg", img, "image/jpeg")}, data={"person": "Port"})
    bundle = client.get("/api/admin/export").content
    assert bundle[:2] == b"PK"
    rep = client.post("/api/admin/import", files={"file": ("b.zip", bundle, "application/zip")}).json()
    assert rep["faces"] == 1


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "FaceMatch" in r.text
