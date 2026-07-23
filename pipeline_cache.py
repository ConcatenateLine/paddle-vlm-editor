"""
Pipeline instantiation, caching, and dispatch.

* **In-process cache** -- OCR, table-recognition, formula-recognition, and
  chart-parsing pipelines are lazily built and reused across calls via
  ``get_pipeline()``.
* **Subprocess worker** -- the ``doc_parser`` (PaddleOCR-VL) pipeline runs
  in a dedicated subprocess because of a known static-graph crash bug that
  corrupts paddle's global device state.  See ``DocParserWorker`` below.

The single entry point ``run_pipeline_with_recovery()`` decides which
path to take.
"""

from __future__ import annotations

import queue as queue_module
import threading
import time
from pathlib import Path
from typing import Optional

import doc_parser_worker
from config import _MP_CTX
from runners import RUNNERS

# ---------------------------------------------------------------------------
# doc_parser subprocess worker
# ---------------------------------------------------------------------------

class _StaticGraphBug(RuntimeError):
    """Raised when the doc_parser worker reports the known static-graph crash.

    The worker process has already been torn down by the time this is
    raised, so callers can retry immediately against a fresh one.
    """


class DocParserWorker:
    """Owns a single persistent subprocess running ``PaddleOCRVL.predict()``.

    Started lazily on first use and kept alive across calls so model
    weights are only loaded once per subprocess lifetime.  Torn down and
    respawned on any failure -- see module docstring for why in-process
    recovery doesn't work for this particular pipeline.
    """

    STARTUP_TIMEOUT: int = 600   # model weight loading can be slow on first boot
    PREDICT_TIMEOUT: int = 900   # generous ceiling for a single large PDF

    def __init__(self, device: str) -> None:
        self.device = device
        self._process: Optional[object] = None
        self._task_q: Optional[object] = None
        self._result_q: Optional[object] = None
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def _start(self) -> None:
        self._task_q = _MP_CTX.Queue()
        self._result_q = _MP_CTX.Queue()
        self._process = _MP_CTX.Process(
            target=doc_parser_worker.worker_main,
            args=(self.device, self._task_q, self._result_q),
            daemon=True,
        )
        self._process.start()

        try:
            status, payload = self._result_q.get(timeout=self.STARTUP_TIMEOUT)
        except queue_module.Empty:
            self._stop()
            raise RuntimeError(
                f"doc_parser worker did not become ready within "
                f"{self.STARTUP_TIMEOUT}s during startup."
            )

        if status != "ready":
            self._stop()
            raise RuntimeError(f"doc_parser worker failed to start: {payload}")

    def _stop(self) -> None:
        if self._process is not None and self._process.is_alive():
            try:
                self._task_q.put_nowait(None)  # ask nicely first
            except Exception:
                pass
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
        self._process = None
        self._task_q = None
        self._result_q = None

    # -- public API --------------------------------------------------------

    def predict(self, input_path: str, work_dir: Path) -> tuple[str, str, str]:
        with self._lock:
            if self._process is None or not self._process.is_alive():
                self._start()

            self._task_q.put((input_path, str(work_dir)))

            deadline = time.time() + self.PREDICT_TIMEOUT
            while True:
                try:
                    status, payload = self._result_q.get(timeout=2)
                    break
                except queue_module.Empty:
                    if not self._process.is_alive():
                        self._stop()
                        raise RuntimeError(
                            "doc_parser worker process died unexpectedly "
                            "(likely an out-of-memory kill or crash outside "
                            "the caught exception path). It has been "
                            "restarted -- please retry."
                        )
                    if time.time() > deadline:
                        self._stop()
                        raise RuntimeError(
                            f"doc_parser worker timed out after "
                            f"{self.PREDICT_TIMEOUT}s and was restarted -- "
                            f"please retry."
                        )

            if status == "ok":
                return payload  # (combined_md, combined_json, output_format)

            # status == "error": payload is (error_str, is_static_graph_bug)
            error_str, is_recoverable = payload
            self._stop()
            if is_recoverable:
                raise _StaticGraphBug(error_str)
            raise RuntimeError(error_str)


# ---------------------------------------------------------------------------
# Worker singleton
# ---------------------------------------------------------------------------

_DOC_PARSER_WORKER: Optional[DocParserWorker] = None
_DOC_PARSER_WORKER_LOCK = threading.Lock()


def get_doc_parser_worker(device: str) -> DocParserWorker:
    global _DOC_PARSER_WORKER
    with _DOC_PARSER_WORKER_LOCK:
        if _DOC_PARSER_WORKER is None or _DOC_PARSER_WORKER.device != device:
            _DOC_PARSER_WORKER = DocParserWorker(device)
        return _DOC_PARSER_WORKER


# ---------------------------------------------------------------------------
# In-process pipeline cache (everything except doc_parser)
# ---------------------------------------------------------------------------

_PIPELINE_CACHE: dict[str, object] = {}


def _build_pipeline(pipeline_key: str, device: str):
    if pipeline_key == "ocr":
        from paddleocr import PaddleOCR
        return PaddleOCR(
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )

    elif pipeline_key == "table_recognition_v2":
        from paddleocr import TableRecognitionPipelineV2
        return TableRecognitionPipelineV2(device=device)

    elif pipeline_key == "formula_recognition":
        from paddleocr import FormulaRecognitionPipeline
        return FormulaRecognitionPipeline(device=device)

    elif pipeline_key == "chart_parsing":
        from paddleocr import ChartParsing
        return ChartParsing(model_name="PP-Chart2Table", device=device)

    else:
        raise ValueError(
            f"{pipeline_key} is not handled by the in-process cache "
            f"(doc_parser runs via DocParserWorker instead)."
        )


def get_pipeline(pipeline_key: str, device: str):
    cache_key = f"{pipeline_key}:{device}"
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    pipeline = _build_pipeline(pipeline_key, device)
    _PIPELINE_CACHE[cache_key] = pipeline
    return pipeline


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

def run_pipeline_with_recovery(
    pipeline_key: str,
    device: str,
    input_path: str,
    work_dir: Path,
) -> tuple[str, str]:
    """Dispatch to the doc_parser subprocess worker, or to the normal
    in-process cached pipeline for everything else."""
    if pipeline_key == "doc_parser":
        worker = get_doc_parser_worker(device)
        try:
            return worker.predict(input_path, work_dir)
        except _StaticGraphBug:
            # The worker already tore itself down; predict() will lazily
            # spawn a brand new subprocess on this call, which is what
            # actually clears the corrupted paddle state.
            worker = get_doc_parser_worker(device)
            return worker.predict(input_path, work_dir)

    pipeline = get_pipeline(pipeline_key, device)
    runner = RUNNERS[pipeline_key]
    return runner(pipeline, input_path, work_dir)
