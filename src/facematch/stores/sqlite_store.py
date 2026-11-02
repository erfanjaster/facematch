"""SQLite store — single-file database that scales past the folder store.

One ``facematch.db`` file holds persons, face metadata, embeddings and the
image bytes.  WAL mode keeps writes cheap; a future Postgres/pgvector adapter
can copy this file's schema almost line for line.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Iterator

import numpy as np

from .base import FaceRecord, Person, new_id, now_ts, register_store

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    created_at REAL NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS faces (
    id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    embedder TEXT NOT NULL,
    dim INTEGER NOT NULL,
    added_at REAL NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    embedding BLOB NOT NULL,
    image BLOB NOT NULL,
    crop BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
CREATE INDEX IF NOT EXISTS idx_faces_embedder ON faces(embedder);
"""


@register_store("sqlite")
class SqliteStore:
    name = "sqlite"
    description = "Single-file SQL database (WAL mode, good to ~100k faces)"

    def __init__(self, settings) -> None:
        self.settings = settings
        self.path = Path(settings.data_dir) / settings.sqlite_filename
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    # ---------------------------------------------------------------- lifecycle
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._conn = conn
        return self._conn

    def init(self) -> None:
        with self._lock:
            self._db().executescript(SCHEMA)
            self._db().commit()

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.commit()
                self._conn.close()
                self._conn = None

    def health(self) -> dict:
        with self._lock:
            db = self._db()
            size = self.path.stat().st_size if self.path.exists() else 0
            return {
                "store": self.name,
                "path": str(self.path),
                "persons": db.execute("SELECT COUNT(*) c FROM persons").fetchone()["c"],
                "faces": db.execute("SELECT COUNT(*) c FROM faces").fetchone()["c"],
                "disk_mb": round(size / 1e6, 2),
            }

    # ----------------------------------------------------------------- persons
    def get_or_create_person(self, name: str, meta: dict | None = None) -> Person:
        with self._lock:
            db = self._db()
            row = db.execute("SELECT * FROM persons WHERE name = ?", (name,)).fetchone()
            if row:
                return self._person(row)
            pid = new_id("person")
            db.execute(
                "INSERT INTO persons(id,name,created_at,meta) VALUES (?,?,?,?)",
                (pid, name, now_ts(), json.dumps(meta or {})),
            )
            db.commit()
            return Person(id=pid, name=name, created_at=now_ts(), meta=meta or {}, face_count=0)

    @staticmethod
    def _person(row: sqlite3.Row, face_count: int = 0) -> Person:
        return Person(
            id=row["id"], name=row["name"], created_at=row["created_at"],
            meta=json.loads(row["meta"]), face_count=face_count,
        )

    def list_persons(self) -> list[Person]:
        with self._lock:
            db = self._db()
            rows = db.execute(
                """SELECT p.*, (SELECT COUNT(*) FROM faces f WHERE f.person_id = p.id) AS fc
                   FROM persons p ORDER BY lower(p.name)"""
            ).fetchall()
            return [self._person(r, r["fc"]) for r in rows]

    def get_person(self, person_id: str) -> Person | None:
        with self._lock:
            db = self._db()
            row = db.execute(
                """SELECT p.*, (SELECT COUNT(*) FROM faces f WHERE f.person_id = p.id) AS fc
                   FROM persons p WHERE p.id = ?""",
                (person_id,),
            ).fetchone()
            return self._person(row, row["fc"]) if row else None

    def delete_person(self, person_id: str) -> int:
        with self._lock:
            db = self._db()
            n = db.execute("SELECT COUNT(*) c FROM faces WHERE person_id = ?", (person_id,)).fetchone()["c"]
            db.execute("DELETE FROM faces WHERE person_id = ?", (person_id,))
            db.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            db.commit()
            return n

    # ------------------------------------------------------------------- faces
    def add_face(self, person_id, embedding, embedder, image_bytes, crop_bytes, meta=None) -> FaceRecord:
        vec = np.ascontiguousarray(np.asarray(embedding, dtype=np.float32).ravel())
        fid = new_id("face")
        with self._lock:
            db = self._db()
            db.execute(
                """INSERT INTO faces(id,person_id,embedder,dim,added_at,meta,embedding,image,crop)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (fid, person_id, embedder, int(vec.size), now_ts(), json.dumps(meta or {}),
                 vec.tobytes(), image_bytes, crop_bytes),
            )
            db.commit()
        return FaceRecord(fid, person_id, embedder, int(vec.size), now_ts(), meta or {}, vec)

    def _face(self, row: sqlite3.Row, with_embedding: bool = True) -> FaceRecord:
        vec = np.frombuffer(row["embedding"], dtype=np.float32).copy() if with_embedding else None
        return FaceRecord(
            id=row["id"], person_id=row["person_id"], embedder=row["embedder"], dim=row["dim"],
            added_at=row["added_at"], meta=json.loads(row["meta"]), embedding=vec,
        )

    def get_face(self, face_id: str) -> FaceRecord | None:
        with self._lock:
            row = self._db().execute("SELECT * FROM faces WHERE id = ?", (face_id,)).fetchone()
            return self._face(row) if row else None

    def list_faces(self, person_id: str | None = None, limit: int = 100, offset: int = 0) -> list[FaceRecord]:
        with self._lock:
            db = self._db()
            if person_id:
                rows = db.execute(
                    "SELECT id,person_id,embedder,dim,added_at,meta FROM faces WHERE person_id=? ORDER BY added_at LIMIT ? OFFSET ?",
                    (person_id, limit, offset),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT id,person_id,embedder,dim,added_at,meta FROM faces ORDER BY added_at LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            return [self._face(r, with_embedding=False) for r in rows]

    def delete_face(self, face_id: str) -> bool:
        with self._lock:
            db = self._db()
            cur = db.execute("DELETE FROM faces WHERE id = ?", (face_id,))
            db.commit()
            return cur.rowcount > 0

    def read_image(self, face_id: str, kind: str = "crop") -> bytes | None:
        col = "crop" if kind == "crop" else "image"
        with self._lock:
            row = self._db().execute(f"SELECT {col} b FROM faces WHERE id = ?", (face_id,)).fetchone()
            return row["b"] if row else None

    # -------------------------------------------------------------------- bulk
    def iter_embeddings(self, embedder: str | None = None) -> Iterator[tuple[str, np.ndarray]]:
        with self._lock:
            db = self._db()
            if embedder:
                rows = db.execute("SELECT id, embedding FROM faces WHERE embedder = ?", (embedder,)).fetchall()
            else:
                rows = db.execute("SELECT id, embedding FROM faces").fetchall()
        for row in rows:
            yield row["id"], np.frombuffer(row["embedding"], dtype=np.float32).copy()

    def count_faces(self) -> int:
        with self._lock:
            return self._db().execute("SELECT COUNT(*) c FROM faces").fetchone()["c"]

    def export_bundle(self) -> bytes:
        import io
        import zipfile

        buf = io.BytesIO()
        with self._lock:
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                persons = [p.as_dict() for p in self.list_persons()]
                zf.writestr("persons.json", json.dumps(persons))
                for rec in self.list_faces(limit=10**9):
                    rec = self.get_face(rec.id)
                    zf.writestr(f"faces/{rec.id}.json", json.dumps(rec.as_dict()))
                    zf.writestr(f"faces/{rec.id}.emb", rec.embedding.tobytes())
                    zf.writestr(f"faces/{rec.id}.crop.jpg", self.read_image(rec.id, "crop"))
                    zf.writestr(f"faces/{rec.id}.orig.jpg", self.read_image(rec.id, "image"))
        return buf.getvalue()

    def import_bundle(self, payload: bytes) -> dict:
        import io
        import zipfile

        added = 0
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            persons = {p["id"]: p for p in json.loads(zf.read("persons.json"))}
            face_files = [n for n in zf.namelist() if n.startswith("faces/") and n.endswith(".json")]
            for name in face_files:
                meta = json.loads(zf.read(name))
                fid = meta["id"]
                person = self.get_or_create_person(persons[meta["person_id"]]["name"])
                vec = np.frombuffer(zf.read(f"faces/{fid}.emb"), dtype=np.float32)
                self.add_face(
                    person.id, vec, meta["embedder"],
                    zf.read(f"faces/{fid}.orig.jpg"), zf.read(f"faces/{fid}.crop.jpg"), meta.get("meta"),
                )
                added += 1
        return {"persons": len(persons), "faces": added}
