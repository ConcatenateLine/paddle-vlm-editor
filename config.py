"""
Centralised configuration for the PaddleOCR-VL workspace.

All path constants, pipeline definitions, supported extensions, and
process-spawn settings live here so every other module can import them
without circular dependencies.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_STRING_DIR: str = "/mnt/output"
OUTPUT_DIR: Path = Path("/mnt/output")
DATA_DIR: Path = Path("/mnt/data")
FILES_DIR: Path = OUTPUT_DIR / "files"
INDEX_PATH: Path = OUTPUT_DIR / "index.json"

# Ensure directories exist at import time (safe inside Docker).
# Wrapped in try/except so the module can be imported on the host for
# linting and testing even when /mnt/* paths don't exist.
try:
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
PIPELINES: dict[str, str] = {
    "Document Parser": "doc_parser",
    "OCR": "ocr",
    "Table Recognition": "table_recognition_v2",
    "Formula Recognition": "formula_recognition",
    "Chart Parsing": "chart_parsing",
}

# ---------------------------------------------------------------------------
# File extensions
# ---------------------------------------------------------------------------
IMAGE_EXTS: set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
SUPPORTED_EXTS: set[str] = IMAGE_EXTS | {".pdf"}

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
PREVIEW_SIZES: dict[str, int] = {"Small": 300, "Normal (Actual)": 560}

# ---------------------------------------------------------------------------
# Multiprocessing
# ---------------------------------------------------------------------------
# "spawn" (not the Linux default "fork") is required: forking would inherit
# the parent's already-initialised CUDA/paddle context, which is exactly the
# corrupted state we're trying to escape by using a subprocess.
_MP_CTX: mp.context.BaseContext = mp.get_context("spawn")
