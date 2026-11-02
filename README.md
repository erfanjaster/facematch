# FaceMatch 🔎

**Give the service a photo of a person → it tells you in milliseconds whether
that *face* already exists in your database — and if so, hands you back the
stored face crop plus the original photo with the match highlighted.**

Matching is done on **face embeddings**, never on whole-image comparison:
a different photo, angle, lighting or background of the same person still
matches; a different person in an identical setting does not.

Built as a small, honest, *pluggable* system: the embedding backend, the
vector index and the storage adapter each sit behind a tiny protocol, so the
"folder on disk" database you start with today can grow into SQLite, a remote
API or Postgres tomorrow without rewriting anything.

---

## Highlights

| | |
|---|---|
|  **Face-only matching** | detect → align → embed → cosine search. The stored *face crop* is returned, not the whole photo (the original + annotated original are one click away). |
| ⚡ **Fast search** | in-memory float32 matrix + BLAS dot product; exact FAISS `IndexFlatIP` available for big galleries. Sub-millisecond vector search at 10k scale (see benchmarks). |
| 🗄 **Flexible database** | `folder` store (browse/zip/git it) · `sqlite` store (single file, ~100k faces) · `http` store (federate another instance). Add your own in one file. |
| 🔌 **Pluggable backends** | `opencv_sface` (default, ~40 MB ONNX, CPU) · `arcface` (SCRFD + ArcFace 512-d, higher accuracy) · `remote_api` (any cloud face API) · `deterministic` (tests). |
| 👥 **Multi-face queries** | group photos return one verdict per detected face. |
| 🖥 **Zero-build web UI** | search / enroll / gallery / database admin / benchmarks, no CDN, no npm. |
| 🐳 **Deployable** | Dockerfile + compose, healthcheck, env-driven config. |

---

## Quickstart

```bash
./bootstrap.sh                 # venv + deps + ONNX models (~40 MB)
source .venv/bin/activate

python scripts/seed_demo_db.py # optional: enroll demo_images/ and run sample queries
python -m facematch            # UI + API on http://localhost:8000
```

Open <http://localhost:8000> — drop a photo in **Search**, watch the verdict.

> Models are downloaded by `scripts/download_models.py` into `models/`.
> For the high-accuracy backend also run
> `python scripts/download_models.py --pack buffalo_l` (~190 MB) and set
> `FACEMATCH_EMBEDDER=arcface`.

### Docker

```bash
docker compose up --build      # serves :8000 with a named volume for data/
```

---

## How it works

```
            upload / paste
                 │
                 ▼
        ┌─────────────────┐   YuNet / SCRFD      ┌──────────────┐
        │  decode + EXIF  │ ───────────────────► │ face boxes   │
        └─────────────────┘   (detector)        └──────┬───────┘
                                                       │ per face
                                                       ▼
                                      align (5 keypoints) → 112×112
                                                       │
                                                       ▼
                                    embedder (SFace / ArcFace / API)
                                                       │ 128-d or 512-d float32
                                                       ▼
                            ┌────────────────────────────────────────┐
                            │  VectorIndex (cosine = inner product)  │
                            │  numpy BLAS  ──auto──►  FAISS FlatIP   │
                            └───────────────┬────────────────────────┘
                                            │ top-k hits + similarity
                                            ▼
                       Store (folder | sqlite | http) ──► person, crop,
                                                          original, metadata
```

1. **Detect** – YuNet (default) or SCRFD locates faces with a confidence floor
   (`FACEMATCH_DETECTION_SCORE_THRESHOLD`).
2. **Align & embed** – 5 keypoints warp the face to a canonical 112×112 patch;
   the recognizer maps it to an L2-normalised vector.
3. **Search** – cosine similarity against every stored vector of the *same
   embedder* (vectors from different backends are incompatible and are kept in
   separate namespaces).
4. **Verdict** – `similarity ≥ threshold` ⇒ MATCH. Defaults: **0.363** for
   SFace (OpenCV model-zoo recommendation), **0.40** for ArcFace. Tune live
   with the slider in the UI or `FACEMATCH_MATCH_THRESHOLD`.
5. **Return** – stored face crop, person name, similarity, plus URLs for the
   original photo and an *annotated* copy with the matched face boxed.

---

## Benchmarks (measured)

Vector-search latency, 128-d, exact top-5, on a 2-core CI container
(`scripts/benchmark.py`, 300 queries per size):

| vectors | engine | p50 | mean | queries/s |
|---:|---|---:|---:|---:|
| 1 000 | numpy | **0.04 ms** | 0.15 ms | ~6 600 |
| 10 000 | numpy | **0.18 ms** | 0.30 ms | ~3 300 |
| 100 000 | numpy | **3.5 ms** | 7.7 ms | ~130 |
| 1 000 000 | numpy | **87 ms** | 76 ms | ~13 |

End-to-end `/api/search` on real photos (demo gallery): **35–90 ms** total, of
which vector search is **< 2 ms** — the rest is JPEG decode + detection +
embedding. On real servers (more cores, `faiss`, or an ANN index — roadmap)
the 100k+ rows improve dramatically; at gallery sizes typical of v1 use
(< 50k faces) search is effectively free.

Match quality on the bundled demo set (`opencv_sface`):

| query | result |
|---|---|
| different event, full-body, small face | ✅ MATCH, sim **0.74** |
| group photo, 4 faces | ✅ only the enrolled person matches (0.42); the 3 strangers correctly don't (≤ 0.17) |
| unrelated person | ❌ no match, closest sim **0.19** (threshold 0.363) |

With `arcface`: same-person 0.84 / 0.66 vs different-person −0.03 / −0.09.

---

## The flexible database

### v1 default: folder store

```
data/faces/
├── index.json                    # persons + face metadata (human readable)
├── barack_obama/
│   ├── faces/face_8f3k… .jpg     # the stored FACE CROP
│   ├── faces/face_8f3k… .emb     # raw float32 embedding
│   └── originals/face_8f3k… .jpg # full source photo
└── …
```

Perfect for tests & early development: inspect it, `zip` it, put it in git,
copy it between machines, import/export it from the UI.

### Scale up: sqlite store

`FACEMATCH_STORE=sqlite` → one `data/facematch.db` (WAL mode) holding persons,
faces, embeddings and image blobs. Same API, same UI.

### Federate: http store

`FACEMATCH_STORE=http` + `FACEMATCH_STORE_URL=http://other-host:8000` → this
process keeps a hot in-RAM index while records live on another instance.

### Write your own adapter

Implement the `Store` protocol (`src/facematch/stores/base.py`, ~15 methods)
and decorate with `@register_store("postgres")`. Same for embedders
(`@register("my_backend")`) — the registry powers `/api/config` and the UI's
adapter panel.

---

## API tour

Full schema at `/docs` (Swagger). Essentials:

| method & path | purpose |
|---|---|
| `POST /api/search` | multipart `file` (or `image_b64`) → per-face verdicts, candidates, timings |
| `POST /api/enroll` | `file` + `person` (+`create_person`) → store face(s) |
| `GET /api/persons` · `POST /api/persons` · `DELETE /api/persons/{id}` | manage people |
| `GET /api/persons/{id}/faces` · `DELETE /api/faces/{id}` | manage stored faces |
| `GET /api/media/faces/{id}/crop\|image\|annotated` | face crop / original / original-with-box |
| `GET /api/stats` · `GET /api/config` · `GET /api/health` | observability |
| `GET /api/embeddings` · `POST /api/faces` | raw vector access (external tools, http store) |
| `POST /api/admin/reindex` · `GET /api/admin/export` · `POST /api/admin/import` | maintenance |
| `POST /api/admin/benchmark` · `POST /api/admin/ingest` | latency probe, bulk folder ingest |

Example:

```bash
curl -s -F file=@queries/obama_hard.jpg -F top_k=3 http://localhost:8000/api/search | jq
```

```json
{
  "faces": [{
    "bbox": [277.0, 33.0, 62.0, 78.0],
    "score": 0.87,
    "matched": true,
    "best_similarity": 0.744,
    "candidates": [{
      "person_name": "Barack Obama",
      "similarity": 0.744,
      "crop_url": "/api/media/faces/face_…/crop",
      "image_url": "/api/media/faces/face_…/image"
    }]
  }],
  "timings_ms": {"decode": 51.9, "detect": 15.8, "embed": 18.9, "search": 0.8, "total": 87.4}
}
```

---

## Configuration

Everything via env (or `.env`, see `.env.example`): `FACEMATCH_EMBEDDER`,
`FACEMATCH_STORE`, `FACEMATCH_DATA_DIR`, `FACEMATCH_MODELS_DIR`,
`FACEMATCH_MATCH_THRESHOLD`, `FACEMATCH_TOP_K`,
`FACEMATCH_DETECTION_SCORE_THRESHOLD`, `FACEMATCH_INDEX_BACKEND`,
`FACEMATCH_FAISS_MIN_VECTORS`, `FACEMATCH_MAX_UPLOAD_MB`,
`FACEMATCH_REMOTE_EMBED_URL`, `FACEMATCH_STORE_URL`, `FACEMATCH_PORT`, …

---

## Project layout

```
src/facematch/
├── config.py            # env-driven settings (pydantic-settings)
├── imaging.py           # decode/EXIF, crops, thumbnails, annotations
├── pipeline.py          # Embedder protocol + backend registry
├── backends/
│   ├── opencv_sface.py  #   default: YuNet + SFace (128-d)
│   ├── arcface.py       #   SCRFD + ArcFace w600k_r50 (512-d, aligned)
│   ├── remote_api.py    #   any HTTP face API
│   └── deterministic.py #   model-free, for tests/offline dev
├── index/vector_index.py#   numpy BLAS ⇄ FAISS exact search
├── stores/
│   ├── folder_store.py  #   v1 default: plain directory DB
│   ├── sqlite_store.py  #   single-file SQL DB
│   └── http_store.py    #   remote FaceMatch as DB
├── service.py           # orchestration: enroll / search / ingest / bench
└── api/                 # FastAPI app + routes (+ /docs)
web/                     # dependency-free SPA (search, enroll, gallery, admin)
scripts/                 # download_models · seed_demo_db · benchmark
tests/                   # 29 tests, model-free core + real-backend integration
```

## Tests

```bash
python -m pytest tests -q          # offline core (deterministic backend)
FACEMATCH_MODELS_DIR=models python -m pytest tests -q   # + real ONNX integration
```

## Roadmap (ideas already architected for)

- **Postgres + pgvector** store adapter (schema mirrors `sqlite_store`)
- **ANN indexes** (FAISS IVF/HNSW, or hnswlib) past ~1M vectors
- **AuthN/Z** (API keys per tenant, signed upload URLs)
- **Batch ingest API** + dedupe pass (near-duplicate face detection)
- **Quality gates**: blur/exposure scoring before enrollment
- **GPU / ONNX quantised models** for the embedders
- **Webhooks** on match / no-match for access-control integrations
- **Redis** hot-cache for repeated queries

## Ethics & privacy note

Face templates are biometric data. Store only what you may lawfully store,
tell your users, and delete on request (`DELETE /api/faces/{id}` or the whole
person). The bundled demo photos are sample data only — swap in your own.

## License

MIT — see `LICENSE` (demo imagery excluded, see note therein).
