"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import Settings
from ..service import FaceMatchService
from .routes import router

WEB_DIR = Path(__file__).resolve().parents[3] / "web"

log = logging.getLogger("facematch")


def create_app(settings: Settings | None = None) -> FastAPI:
    from ..config import get_settings

    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    service = FaceMatchService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.start()
        log.info(
            "FaceMatch %s ready | embedder=%s store=%s vectors=%d",
            __version__, service.embedder.name, service.store.name, len(service.index),
        )
        yield
        service.store.close()

    app = FastAPI(
        title="FaceMatch",
        version=__version__,
        description="Fast, pluggable face search: give a photo, find the *face* in your database.",
        lifespan=lifespan,
    )
    app.state.service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return app
