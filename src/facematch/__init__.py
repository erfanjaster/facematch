"""FaceMatch — a fast, pluggable face search engine.

Give the service a photo of a person and it will tell you, in milliseconds,
whether that *face* (not the photo) already lives in your database — and if
so, it returns the stored face crop plus the original image with the matched
face highlighted.

The project is intentionally small and dependency-light, but every moving
part (embedding backend, vector index, storage adapter) sits behind a
protocol so it can be swapped for something bigger later (SQL/Postgres,
Redis, cloud face APIs, ...).
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
