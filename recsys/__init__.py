import contextlib
import sys

if sys.platform == "darwin":
    # On macOS torch (which feast imports when it is installed), faiss and LightGBM each ship
    # their own libomp, and LightGBM segfaults if another copy initialised first. Loading it
    # here, before anything else, avoids that. Linux images use one libgomp and are unaffected.
    with contextlib.suppress(ImportError):
        import lightgbm  # noqa: F401
