"""``python -m facematch`` entry point."""

from __future__ import annotations

import uvicorn

from .api import create_app
from .config import get_settings


def main() -> None:
    settings = get_settings()
    settings.ensure_dirs()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
