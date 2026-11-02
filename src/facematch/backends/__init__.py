"""Backend registry imports (importing this module registers everything)."""

from . import arcface, deterministic, opencv_sface, remote_api  # noqa: F401

__all__ = ["arcface", "deterministic", "opencv_sface", "remote_api"]
