"""Folder store — the v1 default.

The whole database is a plain directory you can browse, copy, zip or put under
git::

    data/faces/
      index.json                      # persons + face metadata (no vectors)
      <person_slug>/
        faces/<face_id>.jpg           # the stored *face crop*
        faces/<face_id>.emb           # raw float32 embedding (dim * 4 bytes)
        originals/<face_id>.jpg       # full source photo

Perfect for tests and early development; swap to ``sqlite`` (or your own
adapter) when you outgrow it — the rest of the system will not notice.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import threading
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np

from .base import FaceRecord, Person, new_id, now_ts, register_store

INDEX_NAME = "index.json"


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "person"


@register_store("folder")
class FolderStore:
    name = "folder"
    description = "Plain-directory database (browsable, zip-able, dev friendly)"

    def __init__(self, settings) -> None:
        self.settings = settings
        self.root = Path(settings.data_dir) / "faces"
        self._lock = threading.RLock()
        self._persons: dict[str, dict] = {}
        self._faces: dict[str, dict] = {}
        self._created_at = now_ts()

    # ---------------------------------------------------------------- lifecycle
    def init(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / INDEX_NAME).exists():
            self._load()
        else:
            self._save()

    def close(self) -> None:
        with self._lock:
            self._save()

    def health(self) -> dict:
        size = sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())
        return {
            "store": self.name,
            "path": str(self.root),
            "persons": len(self._persons),
            "faces": len(self._faces),
            "disk_mb": round(size / 1e6, 2),
        }

    # ------------------------------------------------------------------- io
    def _load(self) -> None:
        data = json.loads((self.root / INDEX_NAME).read_text(encoding="utf-8"))
        self._persons = data.get("persons", {})
        self._faces = data.get("faces", {})
        self._created_at = data.get("created_at", now_ts())

    def _save(self) -> None:
        payload = {
            "version": 1,
            "created_at": self._created_at,
            "persons": self._persons,
            "faces": self._faces,
        }
        tmp = self.root / (INDEX_NAME + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.root / INDEX_NAME)

    def _person_dir(self, person: dict) -> Path:
        return self.root / person["slug"]

    # --------------------------------------------------------------- persons
    def get_or_create_person(self, name: str, meta: dict | None = None) -> Person:
        with self._lock:
            for pid, p in self._persons.items():
                if p["name"].lower() == name.lower():
                    return self._person_obj(pid)
            pid = new_id("person")
            slug = slugify(name)
            candidate = slug
            n = 2
            while any(p["slug"] == candidate for p in self._persons.values()):
                candidate = f"{slug}_{n}"
                n += 1
            self._persons[pid] = {
                "id": pid,
                "name": name,
                "slug": candidate,
                "created_at": now_ts(),
                "meta": meta or {},
            }
            self._save()
            return self._person_obj(pid)

    def _person_obj(self, pid: str) -> Person:
        p = self._persons[pid]
        count = sum(1 for f in self._faces.values() if f["person_id"] == pid)
        return Person(id=pid, name=p["name"], created_at=p["created_at"], meta=p.get("meta", {}), face_count=count)

    def list_persons(self) -> list[Person]:
        with self._lock:
            return sorted((self._person_obj(pid) for pid in self._persons), key=lambda p: p.name.lower())

    def get_person(self, person_id: str) -> Person | None:
        with self._lock:
            return self._person_obj(person_id) if person_id in self._persons else None

    def delete_person(self, person_id: str) -> int:
        with self._lock:
            if person_id not in self._persons:
                return 0
            doomed = [fid for fid, f in self._faces.items() if f["person_id"] == person_id]
            for fid in doomed:
                self._remove_face_files(fid)
                del self._faces[fid]
            person = self._persons.pop(person_id)
            pdir = self._person_dir(person)
            shutil.rmtree(pdir, ignore_errors=True)
            self._save()
            return len(doomed)

    # ----------------------------------------------------------------- faces
    def _remove_face_files(self, face_id: str) -> None:
        f = self._faces.get(face_id)
        if not f:
            return
        pdir = self._person_dir(self._persons[f["person_id"]])
        for rel in (f"faces/{face_id}.jpg", f"faces/{face_id}.emb", f"originals/{face_id}.jpg"):
            (pdir / rel).unlink(missing_ok=True)

    def add_face(
        self,
        person_id: str,
        embedding: np.ndarray,
        embedder: str,
        image_bytes: bytes,
        crop_bytes: bytes,
        meta: dict | None = None,
    ) -> FaceRecord:
        with self._lock:
            person = self._persons[person_id]
            fid = new_id("face")
            pdir = self._person_dir(person)
            (pdir / "faces").mkdir(parents=True, exist_ok=True)
            (pdir / "originals").mkdir(parents=True, exist_ok=True)
            (pdir / "faces" / f"{fid}.jpg").write_bytes(crop_bytes)
            (pdir / "originals" / f"{fid}.jpg").write_bytes(image_bytes)
            vec = np.ascontiguousarray(np.asarray(embedding, dtype=np.float32).ravel())
            (pdir / "faces" / f"{fid}.emb").write_bytes(vec.tobytes())
            self._faces[fid] = {
                "id": fid,
                "person_id": person_id,
                "embedder": embedder,
                "dim": int(vec.size),
                "added_at": now_ts(),
                "meta": meta or {},
            }
            self._save()
            return self._face_obj(fid, vec)

    def _face_obj(self, fid: str, vec: np.ndarray | None = None) -> FaceRecord:
        f = self._faces[fid]
        if vec is None:
            emb_file = self._person_dir(self._persons[f["person_id"]]) / "faces" / f"{fid}.emb"
            if emb_file.exists():
                vec = np.frombuffer(emb_file.read_bytes(), dtype=np.float32).copy()
        return FaceRecord(
            id=fid,
            person_id=f["person_id"],
            embedder=f["embedder"],
            dim=f["dim"],
            added_at=f["added_at"],
            meta=f.get("meta", {}),
            embedding=vec,
        )

    def get_face(self, face_id: str) -> FaceRecord | None:
        with self._lock:
            if face_id not in self._faces:
                return None
            return self._face_obj(face_id)

    def list_faces(self, person_id: str | None = None, limit: int = 100, offset: int = 0) -> list[FaceRecord]:
        with self._lock:
            items = sorted(self._faces.values(), key=lambda f: f["added_at"])
            if person_id:
                items = [f for f in items if f["person_id"] == person_id]
            return [self._face_obj(f["id"]) for f in items[offset : offset + limit]]

    def delete_face(self, face_id: str) -> bool:
        with self._lock:
            if face_id not in self._faces:
                return False
            self._remove_face_files(face_id)
            del self._faces[face_id]
            self._save()
            return True

    def _face_path(self, face_id: str, kind: str) -> Path | None:
        f = self._faces.get(face_id)
        if not f:
            return None
        pdir = self._person_dir(self._persons[f["person_id"]])
        return pdir / ("faces" if kind == "crop" else "originals") / f"{face_id}.jpg"

    def read_image(self, face_id: str, kind: str = "crop") -> bytes | None:
        with self._lock:
            path = self._face_path(face_id, kind)
            return path.read_bytes() if path and path.exists() else None

    # ------------------------------------------------------------------ bulk
    def iter_embeddings(self, embedder: str | None = None) -> Iterator[tuple[str, np.ndarray]]:
        with self._lock:
            snapshot = [(fid, dict(f)) for fid, f in self._faces.items()]
        for fid, f in snapshot:
            if embedder and f["embedder"] != embedder:
                continue
            pdir = self._person_dir(self._persons[f["person_id"]])
            emb_file = pdir / "faces" / f"{fid}.emb"
            if not emb_file.exists():
                continue
            raw = emb_file.read_bytes()
            vec = np.frombuffer(raw, dtype=np.float32).copy()
            yield fid, vec

    def count_faces(self) -> int:
        with self._lock:
            return len(self._faces)

    def export_bundle(self) -> bytes:
        with self._lock:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for path in sorted(self.root.rglob("*")):
                    if path.is_file() and path.name != INDEX_NAME + ".tmp":
                        zf.write(path, path.relative_to(self.root).as_posix())
            return buf.getvalue()

    def import_bundle(self, payload: bytes) -> dict:
        """Merge a bundle exported from another instance (ids are remapped)."""
        added_persons = added_faces = 0
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            index = json.loads(zf.read(INDEX_NAME))
            slug_by_pid = {pid: p["slug"] for pid, p in index.get("persons", {}).items()}
            name_by_pid = {pid: p["name"] for pid, p in index.get("persons", {}).items()}
            for fid, f in index.get("faces", {}).items():
                person = self.get_or_create_person(name_by_pid[f["person_id"]])
                if person.face_count == 0 and added_persons == 0:
                    pass
                crop = zf.read(f"{slug_by_pid[f['person_id']]}/faces/{fid}.jpg")
                orig = zf.read(f"{slug_by_pid[f['person_id']]}/originals/{fid}.jpg")
                vec = np.frombuffer(zf.read(f"{slug_by_pid[f['person_id']]}/faces/{fid}.emb"), dtype=np.float32)
                self.add_face(person.id, vec, f["embedder"], orig, crop, f.get("meta"))
                added_faces += 1
            added_persons = len({f["person_id"] for f in index.get("faces", {}).values()})
        return {"persons": added_persons, "faces": added_faces}
