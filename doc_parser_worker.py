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

IMAGE_BLOCK_LABELS = {"image", "header_image"}


def _extract_images_from_pdf(input_path: str, work_dir: Path, pages_data: list):
    """Render each page and crop image blocks using bbox coordinates."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return

    ext = Path(input_path).suffix.lower()
    if ext not in (".pdf",):
        return

    try:
        pdf = pdfium.PdfDocument(input_path)
    except Exception:
        return

    images_dir = work_dir / "images"
    images_dir.mkdir(exist_ok=True)

    for page_idx, page_data in enumerate(pages_data):
        if not isinstance(page_data, dict):
            continue
        parsing = page_data.get("parsing_res_list", [])
        has_images = any(
            b.get("block_label") in IMAGE_BLOCK_LABELS for b in parsing
        )
        if not has_images:
            continue

        try:
            page = pdf[page_idx]
            bitmap = page.render(scale=2.0)
            pil_image = bitmap.to_pil()
        except Exception:
            continue

        for block in parsing:
            if block.get("block_label") not in IMAGE_BLOCK_LABELS:
                continue
            bbox = block.get("block_bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            try:
                cropped = pil_image.crop((x1 * 2, y1 * 2, x2 * 2, y2 * 2))
                img_name = f"page{page_idx}_block{block.get('block_id', 0)}.png"
                img_path = images_dir / img_name
                cropped.save(str(img_path))
                block["block_content"] = str(img_path)
            except Exception:
                continue

    try:
        pdf.close()
    except Exception:
        pass


def _read_first_matching(directory: Path, suffix: str) -> str:
    matches = sorted(directory.glob(f"*{suffix}"))
    if not matches:
        return ""
    return matches[0].read_text(encoding="utf-8", errors="replace")


def run_doc_parser(pipeline, input_path: str, work_dir: Path):
    outputs = pipeline.predict(input_path)
    json_parts = []
    pages_data = []
    for i, res in enumerate(outputs):
        page_dir = work_dir / f"page_{i:03d}"
        page_dir.mkdir(exist_ok=True)
        res.save_to_json(save_path=str(page_dir))
        raw = _read_first_matching(page_dir, ".json")
        json_parts.append(raw)
        if raw:
            try:
                pages_data.append(json.loads(raw))
            except Exception:
                pages_data.append({})

    _extract_images_from_pdf(input_path, work_dir, pages_data)

    updated_parts = []
    for page_obj in pages_data:
        updated_parts.append(json.dumps(page_obj, ensure_ascii=False))
    combined_json = "[\n" + ",\n".join(p for p in updated_parts if p) + "\n]"
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

