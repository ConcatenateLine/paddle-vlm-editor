"""
Subprocess worker for the doc_parser (PaddleOCR-VL) pipeline.

Kept in its own module, deliberately free of any Gradio/UI imports, so
that when web_client.py spawns this as a subprocess (via multiprocessing's
"spawn" start method -- required because paddle's CUDA context cannot be
safely forked) the child process only has to import this lightweight
module, not re-import and re-build the whole Gradio app.

Why a subprocess at all
------------------------
PaddleOCR-VL's local VLM worker has a known bug where reusing the same
PaddleOCRVL instance for a second .predict() call can crash with:

    int(Tensor) is not supported in static graph mode

The first fix attempt -- catching that error and rebuilding a fresh
PaddleOCRVL instance *in the same process* -- does not reliably recover.
paddle's global device/runtime state is left corrupted after the crash,
so the rebuild attempt itself can fail differently, e.g.:

    is_bfloat16_supported(): incompatible function arguments
    Invoked with: Place(undefined:0)

A brand new OS process is the only thing that reliably gives paddle a
clean slate. So doc_parser runs in a dedicated subprocess that gets torn
down and respawned whenever it hits a failure, instead of trying to
recover in place.
"""

import json
import traceback
from pathlib import Path


def _read_first_matching(directory: Path, suffix: str) -> str:
    matches = sorted(directory.glob(f"*{suffix}"))
    if not matches:
        return ""
    return matches[0].read_text(encoding="utf-8", errors="replace")


def run_doc_parser(pipeline, input_path: str, work_dir: Path):
    outputs = pipeline.predict(input_path)
    json_parts = []
    for i, res in enumerate(outputs):
        page_dir = work_dir / f"page_{i:03d}"
        page_dir.mkdir(exist_ok=True)
        res.save_to_json(save_path=str(page_dir))
        json_parts.append(_read_first_matching(page_dir, ".json"))
    combined_json = "[\n" + ",\n".join(p for p in json_parts if p) + "\n]"
    return combined_json, "json"


def is_static_graph_bug(exc: Exception) -> bool:
    return "not supported in static graph mode" in str(exc)


def worker_main(device: str, task_q, result_q):
    """Entry point run inside the dedicated subprocess.

    Builds PaddleOCRVL exactly once, signals readiness, then serves
    predict requests from task_q until the parent sends the shutdown
    sentinel (None) or kills the process outright.
    """
    try:
        from paddleocr import PaddleOCRVL
        pipeline = PaddleOCRVL(device=device)
    except Exception as e:
        # Model failed to load at all -- tell the parent and exit.
        result_q.put(("startup_error", f"{e}\n{traceback.format_exc()}"))
        return

    result_q.put(("ready", None))

    while True:
        item = task_q.get()
        if item is None:  # shutdown sentinel
            return

        input_path, work_dir_str = item
        try:
            result = run_doc_parser(pipeline, input_path, Path(work_dir_str))
            result_q.put(("ok", result))
        except Exception as e:
            result_q.put((
                "error",
                (f"{e}\n{traceback.format_exc()}", is_static_graph_bug(e)),
            ))

