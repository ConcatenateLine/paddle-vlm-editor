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


def _iou(box_a, box_b):
    """Intersection over Union for two [x1, y1, x2, y2] boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def _extract_images_from_blocks(input_path: str, work_dir: Path, pages_data: list):
    """Extract image blocks from the document.

    For PDFs: uses PyMuPDF to extract embedded images directly, matching
    them to PaddleOCR-VL blocks by bbox overlap.  Falls back to rendering
    the page region if no embedded image matches.

    For images (PNG/JPG/etc.): crops directly from the source image using
    PaddleOCR-VL's bbox coordinates (which are in image pixel space).
    """
    images_dir = work_dir / "images"
    images_dir.mkdir(exist_ok=True)

    ext = Path(input_path).suffix.lower()
    if ext == ".pdf":
        _extract_from_pdf(input_path, images_dir, pages_data)
    else:
        _extract_from_image(input_path, images_dir, pages_data)


def _extract_from_pdf(input_path, images_dir, pages_data):
    """Extract embedded images from a PDF using PyMuPDF.

    PaddleOCR-VL renders PDF pages at 144 DPI (2× the 72-DPI PDF coordinate
    system), so its bbox coordinates are in a 2×-scaled pixel space.  We scale
    bboxes down by 0.5 before using them with PyMuPDF, which expects PDF
    point coordinates.
    """
    try:
        import pymupdf
    except ImportError:
        return

    try:
        doc = pymupdf.open(input_path)
    except Exception:
        return

    try:
        for page_idx, page_data in enumerate(pages_data):
            if not isinstance(page_data, dict):
                continue
            parsing = page_data.get("parsing_res_list", [])
            image_blocks = [
                b for b in parsing
                if b.get("block_label") in IMAGE_BLOCK_LABELS
                and b.get("block_bbox") and len(b["block_bbox"]) == 4
            ]
            if not image_blocks:
                continue

            try:
                page = doc[page_idx]
            except Exception:
                continue

            page_rect = page.rect

            embedded_images = []
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    rects = page.get_image_rects(xref)
                    for rect in rects:
                        embedded_images.append({
                            "xref": xref,
                            "rect": [rect.x0, rect.y0, rect.x1, rect.y1],
                        })
                except Exception:
                    continue

            matched_embedded = set()
            for block in image_blocks:
                raw_bbox = block["block_bbox"]
                pdf_bbox = [c * 0.5 for c in raw_bbox]

                best_iou = 0
                best_idx = None
                for i, emb in enumerate(embedded_images):
                    if i in matched_embedded:
                        continue
                    iou_val = _iou(pdf_bbox, emb["rect"])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_idx = i

                if best_idx is not None and best_iou > 0.3:
                    matched_embedded.add(best_idx)
                    emb = embedded_images[best_idx]
                    try:
                        pix = pymupdf.Pixmap(doc, emb["xref"])
                        if pix.n > 4:
                            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                        img_name = (
                            f"page{page_idx}_block"
                            f"{block.get('block_id', 0)}.png"
                        )
                        img_path = images_dir / img_name
                        pix.save(str(img_path))
                        block["block_content"] = str(img_path)
                    except Exception:
                        continue

            unmatched = [
                b for b in image_blocks
                if not b.get("block_content")
            ]
            if unmatched:
                for block in unmatched:
                    raw_bbox = block["block_bbox"]
                    x1, y1, x2, y2 = [c * 0.5 for c in raw_bbox]
                    try:
                        rect = pymupdf.Rect(x1, y1, x2, y2)
                        rect = rect & page_rect
                        if rect.is_empty or rect.is_infinite:
                            continue
                        pix = page.get_pixmap(clip=rect)
                        if pix.width == 0 or pix.height == 0:
                            continue
                        img_name = (
                            f"page{page_idx}_block"
                            f"{block.get('block_id', 0)}.png"
                        )
                        img_path = images_dir / img_name
                        pix.save(str(img_path))
                        block["block_content"] = str(img_path)
                    except Exception:
                        continue
    finally:
        try:
            doc.close()
        except Exception:
            pass


def _extract_from_image(input_path, images_dir, pages_data):
    """Crop image blocks directly from an image source."""
    try:
        from PIL import Image
    except ImportError:
        return

    try:
        pil_image = Image.open(input_path)
    except Exception:
        return

    try:
        for page_idx, page_data in enumerate(pages_data):
            if not isinstance(page_data, dict):
                continue
            parsing = page_data.get("parsing_res_list", [])
            for block in parsing:
                if block.get("block_label") not in IMAGE_BLOCK_LABELS:
                    continue
                bbox = block.get("block_bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                try:
                    cropped = pil_image.crop((x1, y1, x2, y2))
                    img_name = (
                        f"page{page_idx}_block"
                        f"{block.get('block_id', 0)}.png"
                    )
                    img_path = images_dir / img_name
                    cropped.save(str(img_path))
                    block["block_content"] = str(img_path)
                except Exception:
                    continue
    finally:
        try:
            pil_image.close()
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

    _extract_images_from_blocks(input_path, work_dir, pages_data)

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

