"""HTTP store — use *another* FaceMatch instance (or any service speaking the
same REST contract) as the database.  This is the stepping stone to a hosted
tier: the local process keeps the vector index hot in RAM while the records
live somewhere else.

Experimental on purpose: latency-bound operations (read_image) go over the
network, so prefer it for federation experiments rather than production.
"""

from __future__ import annotations

import base64
from typing import Iterator

import httpx
import numpy as np

from .base import FaceRecord, Person, register_store


@register_store("http")
class HttpStore:
    name = "http"
    description = "Remote FaceMatch instance as database (federation / future cloud tier)"

    def __init__(self, settings) -> None:
        if not settings.store_url:
            raise ValueError("store=http requires FACEMATCH_STORE_URL")
        headers = {}
        if settings.store_api_key:
            headers["Authorization"] = f"Bearer {settings.store_api_key}"
        self._client = httpx.Client(base_url=settings.store_url.rstrip("/"), timeout=60.0, headers=headers)

    # ---------------------------------------------------------------- lifecycle
    def init(self) -> None:
        self._client.get("/api/health").raise_for_status()

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict:
        r = self._client.get("/api/stats")
        return {"store": self.name, **r.json()}

    # ----------------------------------------------------------------- persons
    def get_or_create_person(self, name: str, meta: dict | None = None) -> Person:
        r = self._client.post("/api/persons", json={"name": name, "meta": meta or {}})
        r.raise_for_status()
        return Person(**{k: r.json()[k] for k in ("id", "name", "created_at", "meta", "face_count")})

    def list_persons(self) -> list[Person]:
        r = self._client.get("/api/persons")
        r.raise_for_status()
        return [Person(**p) for p in r.json()["persons"]]

    def get_person(self, person_id: str) -> Person | None:
        r = self._client.get(f"/api/persons/{person_id}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return Person(**r.json())

    def delete_person(self, person_id: str) -> int:
        r = self._client.delete(f"/api/persons/{person_id}")
        if r.status_code == 404:
            return 0
        r.raise_for_status()
        return r.json().get("deleted_faces", 0)

    # ------------------------------------------------------------------- faces
    def add_face(self, person_id, embedding, embedder, image_bytes, crop_bytes, meta=None) -> FaceRecord:
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        payload = {
            "person_id": person_id,
            "embedding": vec.tolist(),
            "embedder": embedder,
            "image_b64": base64.b64encode(image_bytes).decode(),
            "crop_b64": base64.b64encode(crop_bytes).decode(),
            "meta": meta or {},
        }
        r = self._client.post("/api/faces", json=payload)
        r.raise_for_status()
        data = r.json()
        return FaceRecord(
            id=data["id"], person_id=person_id, embedder=embedder, dim=int(vec.size),
            added_at=data.get("added_at", 0.0), meta=meta or {}, embedding=vec,
        )

    def get_face(self, face_id: str) -> FaceRecord | None:
        r = self._client.get(f"/api/faces/{face_id}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        d = r.json()
        vec = np.asarray(d.get("embedding", []), dtype=np.float32) if d.get("embedding") else None
        return FaceRecord(d["id"], d["person_id"], d["embedder"], d["dim"], d["added_at"], d.get("meta", {}), vec)

    def list_faces(self, person_id: str | None = None, limit: int = 100, offset: int = 0) -> list[FaceRecord]:
        params = {"limit": limit, "offset": offset}
        if person_id:
            params["person_id"] = person_id
        r = self._client.get("/api/faces", params=params)
        r.raise_for_status()
        return [
            FaceRecord(f["id"], f["person_id"], f["embedder"], f["dim"], f["added_at"], f.get("meta", {}), None)
            for f in r.json()["faces"]
        ]

    def delete_face(self, face_id: str) -> bool:
        r = self._client.delete(f"/api/faces/{face_id}")
        return r.status_code < 300

    def read_image(self, face_id: str, kind: str = "crop") -> bytes | None:
        r = self._client.get(f"/api/media/faces/{face_id}/{kind}")
        return r.content if r.status_code == 200 else None

    # -------------------------------------------------------------------- bulk
    def iter_embeddings(self, embedder: str | None = None) -> Iterator[tuple[str, np.ndarray]]:
        params = {"embedder": embedder} if embedder else {}
        r = self._client.get("/api/embeddings", params=params)
        r.raise_for_status()
        for item in r.json()["faces"]:
            yield item["id"], np.asarray(item["embedding"], dtype=np.float32)

    def count_faces(self) -> int:
        return self.health().get("faces", 0)

    def export_bundle(self) -> bytes:
        r = self._client.get("/api/admin/export")
        r.raise_for_status()
        return r.content

    def import_bundle(self, payload: bytes) -> dict:
        r = self._client.post("/api/admin/import", files={"file": ("bundle.zip", payload)})
        r.raise_for_status()
        return r.json()
