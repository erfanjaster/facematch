"""HTTP routes. Thin wrappers around FaceMatchService."""

from __future__ import annotations

import base64
import dataclasses
import io
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from ..config import Settings
from ..imaging import ImageError, decode_image, encode_jpeg
from ..pipeline import list_backends
from ..service import FaceMatchService
from ..stores import list_stores
from .schemas import (
    BenchmarkRequest,
    EnrollJsonRequest,
    IngestRequest,
    PersonCreate,
    RawFaceAdd,
    SearchRequest,
)

router = APIRouter(prefix="/api")


def get_service(request: Request) -> FaceMatchService:
    return request.app.state.service


def get_settings(request: Request) -> Settings:
    return request.app.state.service.settings


Svc = Annotated[FaceMatchService, Depends(get_service)]


# --------------------------------------------------------------------- helpers
async def _image_bytes(file: UploadFile | None, b64: str | None) -> bytes:
    if file is not None:
        return await file.read()
    if b64:
        try:
            return base64.b64decode(b64)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(422, f"bad base64 payload: {exc}") from exc
    raise HTTPException(422, "provide an image via multipart 'file' or JSON 'image_b64'")


def _face_result_dict(fr) -> dict:
    return dataclasses.asdict(fr)


# ----------------------------------------------------------------------- meta
@router.get("/health")
def health(svc: Svc) -> dict:
    from .. import __version__

    return {
        "status": "ok",
        "version": __version__,
        "embedder": svc.embedder.name,
        "store": svc.store.name,
        "started": svc._started,
    }


@router.get("/stats")
def stats(svc: Svc) -> dict:
    return svc.stats()


@router.get("/config")
def config(settings: Annotated[Settings, Depends(get_settings)], svc: Svc) -> dict:
    return {
        "embedder": settings.embedder,
        "store": settings.store,
        "threshold": settings.resolved_match_threshold,
        "top_k": settings.top_k,
        "detection_score_threshold": settings.detection_score_threshold,
        "index_backend": settings.index_backend,
        "faiss_min_vectors": settings.faiss_min_vectors,
        "max_upload_mb": settings.max_upload_mb,
        "backends": list_backends(),
        "stores": list_stores(),
    }


# --------------------------------------------------------------------- search
@router.post("/search")
async def search(
    svc: Svc,
    file: UploadFile | None = File(default=None),
    image_b64: str | None = Form(default=None),
    top_k: int | None = Form(default=None),
    threshold: float | None = Form(default=None),
) -> dict:
    payload = await _image_bytes(file, image_b64)
    try:
        result = svc.search(payload, top_k=top_k, threshold=threshold)
    except ImageError as exc:
        raise HTTPException(422, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {
        "faces": [_face_result_dict(f) for f in result.faces],
        "timings_ms": result.timings_ms,
        "index": result.index_stats,
        "threshold": result.threshold,
        "embedder": result.embedder,
    }


@router.post("/search/json")
def search_json(svc: Svc, body: SearchRequest) -> dict:
    try:
        payload = base64.b64decode(body.image_b64)
        result = svc.search(payload, top_k=body.top_k, threshold=body.threshold)
    except ImageError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "faces": [_face_result_dict(f) for f in result.faces],
        "timings_ms": result.timings_ms,
        "index": result.index_stats,
        "threshold": result.threshold,
        "embedder": result.embedder,
    }


# --------------------------------------------------------------------- enroll
@router.post("/enroll")
async def enroll(
    svc: Svc,
    file: UploadFile | None = File(default=None),
    image_b64: str | None = Form(default=None),
    person: str = Form(...),
    create_person: bool = Form(default=True),
) -> dict:
    payload = await _image_bytes(file, image_b64)
    try:
        return svc.enroll(payload, person, create_person=create_person, source_name=file.filename if file else None)
    except ImageError as exc:
        raise HTTPException(422, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/enroll/json")
def enroll_json(svc: Svc, body: EnrollJsonRequest) -> dict:
    try:
        return svc.enroll(base64.b64decode(body.image_b64), body.person, create_person=body.create_person)
    except ImageError as exc:
        raise HTTPException(422, str(exc)) from exc


# -------------------------------------------------------------------- persons
@router.get("/persons")
def persons(svc: Svc) -> dict:
    return {"persons": [p.as_dict() for p in svc.store.list_persons()]}


@router.post("/persons", status_code=201)
def create_person(svc: Svc, body: PersonCreate) -> dict:
    return svc.store.get_or_create_person(body.name, body.meta).as_dict()


@router.get("/persons/{person_id}")
def get_person(svc: Svc, person_id: str) -> dict:
    person = svc.store.get_person(person_id)
    if person is None:
        raise HTTPException(404, "person not found")
    return person.as_dict()


@router.delete("/persons/{person_id}")
def delete_person(svc: Svc, person_id: str) -> dict:
    doomed = {f.id for f in svc.store.list_faces(person_id=person_id, limit=10**9)}
    removed = svc.store.delete_person(person_id)
    svc.index.remove(doomed)
    return {"deleted_faces": removed}


@router.get("/persons/{person_id}/faces")
def person_faces(svc: Svc, person_id: str, limit: int = 100, offset: int = 0) -> dict:
    faces = svc.store.list_faces(person_id=person_id, limit=limit, offset=offset)
    return {"faces": [f.as_dict() for f in faces]}


# ---------------------------------------------------------------------- faces
@router.get("/faces")
def faces(svc: Svc, person_id: str | None = None, limit: int = 100, offset: int = 0) -> dict:
    return {"faces": [f.as_dict() for f in svc.store.list_faces(person_id, limit, offset)]}


@router.post("/faces", status_code=201)
def add_raw_face(svc: Svc, body: RawFaceAdd) -> dict:
    """Raw insert (embedding supplied by the caller) — used by the http store
    adapter and external tooling."""
    person = svc.store.get_person(body.person_id)
    if person is None:
        raise HTTPException(404, "person not found")
    vec = np.asarray(body.embedding, dtype=np.float32)
    embedder = body.embedder or svc.embedder.name
    image = base64.b64decode(body.image_b64) if body.image_b64 else encode_jpeg(np.zeros((8, 8, 3), np.uint8))
    crop = base64.b64decode(body.crop_b64) if body.crop_b64 else image
    rec = svc.store.add_face(person.id, vec, embedder, image, crop, body.meta)
    if embedder == svc.embedder.name:
        svc.index.add(rec.id, vec)
    return rec.as_dict()


@router.get("/faces/{face_id}")
def get_face(svc: Svc, face_id: str, include_embedding: bool = False) -> dict:
    rec = svc.store.get_face(face_id)
    if rec is None:
        raise HTTPException(404, "face not found")
    return rec.as_dict(include_embedding=include_embedding)


@router.delete("/faces/{face_id}")
def delete_face(svc: Svc, face_id: str) -> dict:
    ok = svc.store.delete_face(face_id)
    if ok:
        svc.index.remove({face_id})
    return {"deleted": ok}


@router.get("/media/faces/{face_id}/{kind}")
def face_media(svc: Svc, face_id: str, kind: str = "crop") -> Response:
    if kind == "annotated":
        data = svc.annotated_image(face_id)
        if data is None:
            raise HTTPException(404, "face not found")
        payload = data[0]
    else:
        if kind not in ("crop", "image"):
            raise HTTPException(422, "kind must be crop|image|annotated")
        payload = svc.store.read_image(face_id, kind)
        if payload is None:
            raise HTTPException(404, "face not found")
    return Response(content=payload, media_type="image/jpeg")


# ------------------------------------------------------------------ bulk/admin
@router.get("/embeddings")
def embeddings(svc: Svc, embedder: str | None = None) -> dict:
    want = embedder or svc.embedder.name
    return {
        "embedder": want,
        "faces": [
            {"id": fid, "embedding": vec.tolist()}
            for fid, vec in svc.store.iter_embeddings(want)
        ],
    }


@router.post("/admin/reindex")
def reindex(svc: Svc) -> dict:
    return svc.reindex()


@router.get("/admin/export")
def export(svc: Svc) -> Response:
    try:
        payload = svc.store.export_bundle()
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc)) from exc
    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="facematch-db.zip"'},
    )


@router.post("/admin/import")
async def import_bundle(svc: Svc, file: UploadFile = File(...)) -> dict:
    payload = await file.read()
    report = svc.store.import_bundle(payload)
    svc.reindex()
    return report


@router.post("/admin/benchmark")
def benchmark(svc: Svc, body: BenchmarkRequest) -> dict:
    return svc.benchmark(body.sizes, body.iterations)


@router.post("/admin/ingest")
def ingest(svc: Svc, body: IngestRequest) -> dict:
    try:
        return svc.ingest_directory(body.path, body.max_images)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"path not found: {exc}") from exc
