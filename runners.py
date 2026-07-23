"""
Per-pipeline result runners.

Each runner takes an instantiated pipeline, an input path, and a working
directory, runs ``predict()``, writes per-page output files, and returns
a combined JSON string.

All pipelines except ``doc_parser`` are handled here.  ``doc_parser`` runs
in a dedicated subprocess via ``DocParserWorker`` (see
``pipeline_cache.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from config import IMAGE_EXTS

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _read_first_matching(directory: Path, suffix: str) -> str:
    """Return the contents of the first file matching ``*<suffix>`` in *directory*."""
    matches = sorted(directory.glob(f"*{suffix}"))
    if not matches:
        return ""
    return matches[0].read_text(encoding="utf-8", errors="replace")


def _extract_res(data: dict) -> dict:
    """Unwrap the ``res`` key if present (PaddleX wrapping convention).

    Pipeline JSON is sometimes wrapped under a top-level ``res`` key and
    sometimes flat.  Try the wrapped form first, fall back to the raw dict.
    """
    res = data.get("res")
    if isinstance(res, dict) and res:
        return res
    return data


# ---------------------------------------------------------------------------
# Generic runner
# ---------------------------------------------------------------------------

def _run_generic(
    pipeline,
    input_path: str,
    work_dir: Path,
    predict_fn: Callable,
    save_fn: Callable[[object, Path], None],
) -> tuple[str, str]:
    """Shared runner: predict -> save per page -> combine into JSON.

    *predict_fn(pipeline, input_path)* must return the iterable of result
    objects.  *save_fn(result_object, page_dir)* writes the desired output
    files (JSON, HTML, etc.) into *page_dir*.
    """
    outputs = predict_fn(pipeline, input_path)
    json_parts: list[str] = []
    for i, res in enumerate(outputs):
        page_dir = work_dir / f"page_{i:03d}"
        page_dir.mkdir(exist_ok=True)
        save_fn(res, page_dir)
        raw_json = _read_first_matching(page_dir, ".json")
        json_parts.append(raw_json)
    combined_json = "[\n" + ",\n".join(p for p in json_parts if p) + "\n]"
    return combined_json, "json"


# ---------------------------------------------------------------------------
# Predict helpers (one per pipeline flavour)
# ---------------------------------------------------------------------------

def _predict_default(pipeline, input_path: str):
    return pipeline.predict(input_path)


def _predict_chart(pipeline, input_path: str):
    return pipeline.predict(input={"image": input_path}, batch_size=1)


# ---------------------------------------------------------------------------
# Save helpers (one per pipeline flavour)
# ---------------------------------------------------------------------------

def _save_json(res, page_dir: Path) -> None:
    res.save_to_json(save_path=str(page_dir))


def _save_html_and_json(res, page_dir: Path) -> None:
    res.save_to_html(save_path=str(page_dir))
    res.save_to_json(save_path=str(page_dir))


# ---------------------------------------------------------------------------
# Public runners
# ---------------------------------------------------------------------------

def run_ocr(pipeline, input_path: str, work_dir: Path) -> tuple[str, str]:
    return _run_generic(pipeline, input_path, work_dir, _predict_default, _save_json)


def run_table(pipeline, input_path: str, work_dir: Path) -> tuple[str, str]:
    return _run_generic(pipeline, input_path, work_dir, _predict_default, _save_html_and_json)


def run_formula(pipeline, input_path: str, work_dir: Path) -> tuple[str, str]:
    return _run_generic(pipeline, input_path, work_dir, _predict_default, _save_json)


def run_chart(pipeline, input_path: str, work_dir: Path) -> tuple[str, str]:
    ext = Path(input_path).suffix.lower()
    if ext not in IMAGE_EXTS:
        raise ValueError(
            "Chart Parsing only accepts single chart images (png/jpg/etc), not PDFs. "
            "Crop the chart out of the PDF first, or use Document Parser on the whole page."
        )
    return _run_generic(pipeline, input_path, work_dir, _predict_chart, _save_json)


# Registry -- used by pipeline_cache.run_pipeline_with_recovery()
RUNNERS: dict[str, Callable] = {
    "ocr": run_ocr,
    "table_recognition_v2": run_table,
    "formula_recognition": run_formula,
    "chart_parsing": run_chart,
}
