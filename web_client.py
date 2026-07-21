"""
PaddleOCR-VL Local Workspace
============================
A Gradio workspace for the official `paddleocr-vl` Docker image, styled
after the AI Studio "文档解析与智能文字识别" experience: pick a file,
run a pipeline, preview/edit the structured output, and keep a history
of everything you've processed.

Intended to run as web_client.py inside the paddleocr-vl container, per
docker-compose.yml:

    services:
      paddleocr-vl:
        image: ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu
        runtime: nvidia
        volumes:
          - ./data:/mnt/data
          - ./output:/mnt/output
          - ./web_client.py:/workspace/web_client.py
          - ./doc_parser_worker.py:/workspace/doc_parser_worker.py
        shm_size: "32g"
        ports:
          - "7860:7860"
        command: bash -c "pip install gradio && python /workspace/web_client.py"

The image ships PaddlePaddle + PaddleOCR preinstalled and does NOT bundle
extra inference engines (vLLM/FastDeploy/Transformers) -- it runs models
via the default paddle inference engine, which is all this app needs.

Recommended compose addition: persist the model cache so weights aren't
re-downloaded every time you recreate the container:

      volumes:
        - ./paddlex_cache:/root/.paddlex

Notes
-----
- Pipelines are lazily instantiated and cached in memory (_PIPELINE_CACHE)
  so switching files does NOT reload model weights every time -- only the
  first run of a given pipeline pays that cost. This applies to OCR,
  table, formula, and chart pipelines.

- KNOWN LIBRARY BUG, doc_parser only: the `doc_parser` (PaddleOCRVL)
  pipeline's VLM worker runs fine on the first `.predict()` call made
  against a given instance, but a second call against that *same*
  instance can crash with `int(Tensor) is not supported in static graph
  mode` -- some part of the VLM model appears to get traced/compiled
  into a static graph after its first forward pass.

  Rebuilding a fresh PaddleOCRVL instance in the same process after that
  crash does NOT reliably recover -- paddle's global device/runtime
  state is left corrupted, and the rebuild itself can then fail
  differently (e.g. `is_bfloat16_supported()` being handed an undefined
  Place). The only thing that reliably clears this is a brand new OS
  process, so doc_parser runs inside a dedicated, persistent subprocess
  (see doc_parser_worker.py + DocParserWorker below) that gets torn down
  and respawned whenever it hits a failure, rather than trying to
  recover in place. Other pipelines have not shown this behavior and
  stay in the normal in-process cache.

- Every result (original file + extracted output + raw JSON) is written
  under /mnt/output and indexed in index.json, so the file list on the
  left survives container restarts (as long as ./output is bind-mounted).
- Files dropped into ./data on the host (-> /mnt/data in the container)
  show up in the "Pick from /mnt/data" dropdown, so you don't have to
  re-upload large PDFs through the browser each time.

- EDITOR: the "Edit" tab uses Editor.js loaded from a CDN for
  a block-based editor with JSON output. Gradio has no native rich-text 
  component, so a plain <div id="editorjs"> is injected via gr.HTML, Editor.js 
  mounts onto it client-side, and a hidden gr.Textbox (#editorjs_hidden_content) 
  is used as the bridge back to Python: JS snippets attached to existing events
  push/pull the editor's JSON into/out of that hidden textbox. See the
  EDITOR_HEAD / _push_into_editorjs_js / _pull_from_editorjs_js constants below.

  Editor.js works with block-based JSON, not HTML. What you save from the 
  Edit tab is Editor.js JSON format, which preserves the structure better 
  than HTML. The original pipeline JSON is still available in result_raw.json.
"""

import json
import multiprocessing as mp
import queue as queue_module
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

import gradio as gr

import doc_parser_worker

OUTPUT_STRING_DIR = "/mnt/output"
OUTPUT_DIR = Path("/mnt/output")
DATA_DIR = Path("/mnt/data")
FILES_DIR = OUTPUT_DIR / "files"
INDEX_PATH = OUTPUT_DIR / "index.json"
FILES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

PIPELINES = {
    "Document Parser": "doc_parser",       # PaddleOCR-VL: full page -> Markdown + JSON
    "OCR": "ocr",                          # PaddleOCR: plain text lines
    "Table Recognition": "table_recognition_v2",  # TableRecognitionPipelineV2: HTML table
    "Formula Recognition": "formula_recognition",  # FormulaRecognitionPipeline: LaTeX
    "Chart Parsing": "chart_parsing",      # ChartParsing model: chart -> data table
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

# "spawn" (not the Linux default "fork") is required here: forking would
# inherit the parent's already-initialized CUDA/paddle context, which is
# exactly the corrupted state we're trying to escape by using a subprocess
# in the first place.
_MP_CTX = mp.get_context("spawn")


# ---------------------------------------------------------------------------
# doc_parser subprocess worker
# ---------------------------------------------------------------------------
class _StaticGraphBug(RuntimeError):
    """Raised when the doc_parser worker reports the known static-graph
    crash. The worker process has already been torn down by the time this
    is raised, so callers can retry immediately against a fresh one."""


class DocParserWorker:
    """Owns a single persistent subprocess running PaddleOCRVL.predict().

    Started lazily on first use and kept alive across calls so model
    weights are only loaded once per subprocess lifetime. Torn down and
    respawned on any failure -- see module docstring for why in-process
    recovery doesn't work for this particular pipeline.
    """

    STARTUP_TIMEOUT = 600   # model weight loading can be slow on first boot
    PREDICT_TIMEOUT = 900   # generous ceiling for a single large PDF

    def __init__(self, device: str):
        self.device = device
        self._process = None
        self._task_q = None
        self._result_q = None
        self._lock = threading.Lock()

    def _start(self):
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

    def _stop(self):
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

    def predict(self, input_path: str, work_dir: Path):
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


_DOC_PARSER_WORKER = None
_DOC_PARSER_WORKER_LOCK = threading.Lock()


def get_doc_parser_worker(device: str) -> DocParserWorker:
    global _DOC_PARSER_WORKER
    with _DOC_PARSER_WORKER_LOCK:
        if _DOC_PARSER_WORKER is None or _DOC_PARSER_WORKER.device != device:
            _DOC_PARSER_WORKER = DocParserWorker(device)
        return _DOC_PARSER_WORKER


# ---------------------------------------------------------------------------
# In-process pipeline cache -- for every pipeline EXCEPT doc_parser, which
# is handled by DocParserWorker above.
# ---------------------------------------------------------------------------
_PIPELINE_CACHE = {}


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


def run_pipeline_with_recovery(pipeline_key: str, device: str, input_path: str, work_dir: Path):
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


# ---------------------------------------------------------------------------
# Workspace index (persisted file history)
# ---------------------------------------------------------------------------
def load_index():
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return []


def save_index(records):
    INDEX_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def add_record(record):
    records = load_index()
    records.insert(0, record)  # newest first
    save_index(records)
    return records


def update_record(record_id, **fields):
    records = load_index()
    for r in records:
        if r["id"] == record_id:
            r.update(fields)
            break
    save_index(records)
    return records


SUPPORTED_EXTS = IMAGE_EXTS | {".pdf"}


def list_data_dir():
    """Scan /mnt/data for supported files (recursively) for the picker dropdown."""
    if not DATA_DIR.exists():
        return []
    paths = sorted(
        p for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    return [str(p.relative_to(DATA_DIR)) for p in paths]


def library_choices(records):
    """Build (label, value) pairs for the workspace dropdown."""
    return [
        (f"{r['original_name']}  ·  {r['pipeline']}  ·  {r['timestamp']}", r["id"])
        for r in records
    ]


def find_record(record_id, records=None):
    records = records if records is not None else load_index()
    for r in records:
        if r["id"] == record_id:
            return r
    return None


# ---------------------------------------------------------------------------
# Per-pipeline result extraction
# All pipelines here save via the library's own save_to_* methods, then we
# read the saved files back so we're never re-implementing their formatting.
# (run_doc_parser lives in doc_parser_worker.py, since it executes inside
# the subprocess rather than here.)
# ---------------------------------------------------------------------------
def _read_first_matching(directory: Path, suffix: str) -> str:
    matches = sorted(directory.glob(f"*{suffix}"))
    if not matches:
        return ""
    return matches[0].read_text(encoding="utf-8", errors="replace")


def _extract_res(data: dict) -> dict:
    """PaddleX/PaddleOCR pipeline JSON is wrapped under a top-level 'res'
    key in some versions, and flat (fields directly at the top level) in
    others. Try the wrapped form first, and fall back to the raw dict
    itself so field lookups (rec_texts, rec_formula, result, ...) work
    either way instead of silently returning nothing."""
    res = data.get("res")
    if isinstance(res, dict) and res:
        return res
    return data


def run_ocr(pipeline, input_path: str, work_dir: Path):
    outputs = pipeline.predict(input_path)
    json_parts = []
    for i, res in enumerate(outputs):
        page_dir = work_dir / f"page_{i:03d}"
        page_dir.mkdir(exist_ok=True)
        res.save_to_json(save_path=str(page_dir))
        raw_json = _read_first_matching(page_dir, ".json")
        json_parts.append(raw_json)
    combined_json = "[\n" + ",\n".join(p for p in json_parts if p) + "\n]"
    return combined_json, "json"


def run_table(pipeline, input_path: str, work_dir: Path):
    outputs = pipeline.predict(input_path)
    json_parts = []
    for i, res in enumerate(outputs):
        page_dir = work_dir / f"page_{i:03d}"
        page_dir.mkdir(exist_ok=True)
        res.save_to_html(save_path=str(page_dir))
        res.save_to_json(save_path=str(page_dir))
        json_parts.append(_read_first_matching(page_dir, ".json"))
    combined_json = "[\n" + ",\n".join(p for p in json_parts if p) + "\n]"
    return combined_json, "json"


def run_formula(pipeline, input_path: str, work_dir: Path):
    outputs = pipeline.predict(input_path)
    json_parts = []
    for i, res in enumerate(outputs):
        page_dir = work_dir / f"page_{i:03d}"
        page_dir.mkdir(exist_ok=True)
        res.save_to_json(save_path=str(page_dir))
        raw_json = _read_first_matching(page_dir, ".json")
        json_parts.append(raw_json)
    combined_json = "[\n" + ",\n".join(p for p in json_parts if p) + "\n]"
    return combined_json, "json"


def run_chart(pipeline, input_path: str, work_dir: Path):
    ext = Path(input_path).suffix.lower()
    if ext not in IMAGE_EXTS:
        raise ValueError(
            "Chart Parsing only accepts single chart images (png/jpg/etc), not PDFs. "
            "Crop the chart out of the PDF first, or use Document Parser on the whole page."
        )
    outputs = pipeline.predict(input={"image": input_path}, batch_size=1)
    json_parts = []
    for i, res in enumerate(outputs):
        page_dir = work_dir / f"page_{i:03d}"
        page_dir.mkdir(exist_ok=True)
        res.save_to_json(save_path=str(page_dir / "res.json"))
        raw_json = _read_first_matching(page_dir, ".json")
        json_parts.append(raw_json)
    combined_json = "[\n" + ",\n".join(p for p in json_parts if p) + "\n]"
    return combined_json, "json"


# doc_parser intentionally omitted: it's dispatched to DocParserWorker
# by run_pipeline_with_recovery, not looked up here.
RUNNERS = {
    "ocr": run_ocr,
    "table_recognition_v2": run_table,
    "formula_recognition": run_formula,
    "chart_parsing": run_chart,
}


# ---------------------------------------------------------------------------
# Main processing entry point
# ---------------------------------------------------------------------------
def process_file(file, data_choice, pipeline_label, device):
    # An uploaded file takes priority; otherwise fall back to whatever was
    # picked from the /mnt/data dropdown.
    if file is not None:
        source_path = Path(file.name)
    elif data_choice:
        source_path = DATA_DIR / data_choice
    else:
        return "Upload a file or pick one from /mnt/data first.", gr.update(), gr.update(), gr.update(), None

    if not source_path.exists():
        return f"File not found: {source_path}", gr.update(), gr.update(), gr.update(), None

    pipeline_key = PIPELINES[pipeline_label]
    record_id = uuid.uuid4().hex[:12]
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    original_name = source_path.name

    work_dir = FILES_DIR / record_id
    work_dir.mkdir(parents=True, exist_ok=True)
    stored_original = work_dir / original_name
    shutil.copy(source_path, stored_original)

    status_lines = [f"Running {pipeline_label} on {original_name}..."]

    try:
        output_json, output_format = run_pipeline_with_recovery(
            pipeline_key, device, str(stored_original), work_dir
        )

        output_path = work_dir / "result.json"
        output_path.write_text(output_json, encoding="utf-8")
        json_path = work_dir / "result_raw.json"
        json_path.write_text(output_json, encoding="utf-8")

        record = {
            "id": record_id,
            "original_name": original_name,
            "pipeline": pipeline_label,
            "timestamp": timestamp,
            "status": "done",
            "original_path": str(stored_original),
            "output_path": str(output_path),
            "json_path": str(json_path),
            "output_format": output_format,
        }
        records = add_record(record)
        status_lines.append("Done.")

    except Exception as e:
        record = {
            "id": record_id,
            "original_name": original_name,
            "pipeline": pipeline_label,
            "timestamp": timestamp,
            "status": "error",
            "original_path": str(stored_original),
            "output_path": None,
            "json_path": None,
            "output_format": None,
            "error": f"{e}\n{traceback.format_exc()}",
        }
        records = add_record(record)
        status_lines.append(f"Error: {e}")
        output_json = ""

    choices = library_choices(records)
    is_image = Path(stored_original).suffix.lower() in IMAGE_EXTS
    preview_update = gr.update(value=str(stored_original) if is_image else None, visible=is_image)
    file_update = gr.update(value=str(stored_original) if not is_image else None, visible=not is_image)

    return (
        "\n".join(status_lines),
        gr.update(choices=choices, value=record_id),
        preview_update,
        file_update,
        output_json,
    )


def load_from_library(record_id):
    if not record_id:
        return "", gr.update(value=None, visible=False), gr.update(value=None, visible=True)

    record = find_record(record_id)
    if record is None:
        return "Record not found.", gr.update(value=None, visible=False), gr.update(value=None, visible=True)

    if record["status"] != "done":
        text = f"This run failed:\n\n{record.get('error', 'unknown error')}"
        original = record["original_path"]
        is_image = Path(original).suffix.lower() in IMAGE_EXTS
        return (
            text,
            gr.update(value=original if is_image else None, visible=is_image),
            gr.update(value=original if not is_image else None, visible=not is_image),
        )

    output_path = Path(record["output_path"])
     
    # Handle backward compatibility: if .json doesn't exist, try .md
    if not output_path.exists():
        md_path = output_path.parent / "result.md"
        if md_path.exists():
            # Convert old markdown records to JSON format
            output_path = md_path
            # Update record to point to new format
            record["output_path"] = str(md_path)
            update_record(record_id, output_path=str(md_path))
    
    if not output_path.exists():
        return f"Output file not found: {output_path}", gr.update(value=None, visible=False), gr.update(value=None, visible=True)
    
    content = output_path.read_text(encoding="utf-8", errors="replace")
    
    # If it's an old .md file, wrap it in a simple JSON structure for the editor
    if output_path.suffix == ".md":
        content = json.dumps({"markdown": content, "legacy_format": True})
    original = record["original_path"]
    is_image = Path(original).suffix.lower() in IMAGE_EXTS

    return (
        content,
        gr.update(value=original if is_image else None, visible=is_image),
        gr.update(value=original if not is_image else None, visible=not is_image),
    )


def save_edit(record_id, edited_json):
    if not record_id:
        return "Nothing selected to save.", edited_json
    record = find_record(record_id)
    if record is None or record.get("output_path") is None:
        return "This entry has no output file to save to.", edited_json
    
    # Validate JSON before saving
    try:
        json.loads(edited_json)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}", edited_json
    
    # Editor.js saves data in its own JSON format, which we preserve
    # The original pipeline JSON is still available in result_raw.json
    Path(record["output_path"]).write_text(edited_json, encoding="utf-8")
    return f"Saved changes to {Path(record['output_path']).name}.", edited_json


def refresh_library():
    records = load_index()
    return gr.update(choices=library_choices(records))


# ---------------------------------------------------------------------------
# Editor.js wiring
# ---------------------------------------------------------------------------
# Loaded once into <head> so the library is available before any of our
# JS snippets run. Editor.js uses block-based JSON output instead of HTML.
EDITOR_HEAD = """
<script src="https://cdn.jsdelivr.net/npm/@editorjs/editorjs@latest" defer></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/table@latest/dist/table.umd.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/header@latest/dist/bundle.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/list@latest/dist/bundle.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/code@latest/dist/bundle.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/quote@latest/dist/bundle.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/delimiter@latest/dist/bundle.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/bold@latest/dist/bundle.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/italic@latest/dist/bundle.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/underline@latest/dist/bundle.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/link@latest/dist/bundle.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/marker@latest/dist/bundle.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/simple-image@latest/dist/bundle.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/checklist@latest/dist/bundle.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/warning@latest/dist/bundle.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/embed@latest/dist/bundle.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/attaches@latest/dist/bundle.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/@editorjs/raw@latest/dist/bundle.js" defer></script>
<style>
  /* ================================== */
  /* Editor.js dark mode variables */
  --editorjs-dark-background: #52525b;
  --editorjs-dark-toolbar-blockmenu-btn-hover: #52525b;
  --editorjs-dark-block-selected-background: #896755;
  /* ================================== */

  #editorjs-editor-wrap { border: 1px solid var(--border-color-primary, #444); border-radius: 8px; }
  /* editorjs_hidden_content is a bridge component only -- it must stay
     mounted in the DOM for the JS push/pull snippets to find it, so we
     hide it with CSS rather than Gradio's visible=False (which can
     conditionally unmount the component instead of just hiding it,
     silently breaking the bridge). */
  #editorjs_hidden_content { display: none !important; }
  #editorjs { min-height: 560px; background: var(--background-fill-primary, #fff); padding: 20px; }
  .ce-block__content { font-size: 15px; line-height: 1.6; }
  .ce-toolbar__content { max-width: 100%; }
  
  /* ================================== */
  /* Manual dark mode overrides for Editor.js components */

  .ce-stub {
    background: var(--editorjs-dark-background, #52525b);
  }
  .ce-toolbar__settings-btn:hover {
    background: var(--editorjs-dark-toolbar-blockmenu-btn-hover, #52525b);
  }
  .ce-toolbar__plus:hover {
    background-color: var(--editorjs-dark-toolbar-blockmenu-btn-hover, #52525b);
  }
  .ce-block--selected .ce-block__content {
    background: var(--editorjs-dark-block-selected-background, #896755);
    box-shadow: var(--editorjs-dark-block-selected-background, #896755) 0px 1px 4px, var(--editorjs-dark-block-selected-background, #896755) 0px 0px 0px 3px;
  }
  .ce-popover__container {
    background: var(--editorjs-dark-toolbar-blockmenu-btn-hover, #52525b);
  }
  .cdx-search-field {
    background: #27272a;
  }
  .ce-popover-item:hover:not(.ce-popover-item--no-hover) {
    background-color: #80808f;
  }
  .codex-editor ::selection {
    background-color: var(--editorjs-dark-block-selected-background, #896755);
  }
  .tc-add-column:hover, .tc-add-row:hover {
    background-color: #52525b !important;
  }
  .tc-add-column svg, .tc-add-row svg {
    background-color: #52525b !important;
  }
  .tc-table--heading .tc-row:first-child {
    background: #27272a !important;
  }
  .tc-popover {
    background: #52525b !important;
  }
  .tc-popover__item-icon {
    background: #27272a !important;
  }
  .tc-cell--selected,
  .tc-row--selected {
    background: #896755 !important;
  }
  .tc-toolbox--showed {
    z-index: 3 !important;
  }
  #editorjs-undo-bar {
    display: flex;
    gap: 4px;
    padding: 6px 10px;
    border-bottom: 1px solid var(--border-color-primary, #444);
  }
  #editorjs-undo-bar button {
    background: var(--editorjs-dark-background, #52525b);
    color: var(--text-color-primary, #fff);
    border: 1px solid var(--border-color-primary, #444);
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 13px;
    cursor: pointer;
    transition: background 0.15s, opacity 0.15s;
  }
  #editorjs-undo-bar button:hover:not(:disabled) {
    background: #80808f;
  }
  #editorjs-undo-bar button:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }
  #editorjs-undo-bar button:active:not(:disabled) {
    background: var(--editorjs-dark-block-selected-background, #896755);
  }
  /* ================================== */
  
  /* Alignment tune styles */
  .alignment-tune {
    width: 100%;
  }

  /* Inline toolbar styles */
  .ce-inline-toolbar {
    background: #27272a;
    border: 1px solid #444;
  }
  .ce-inline-tool:hover {
    background: #80808f;
  }
  .ce-inline-tool--active {
    background: #896755;
  }

  /* Table styles */
  .tc-table {
    border-collapse: collapse;
    width: 100%;
  }
  .tc-cell {
    border: 1px solid #444;
    padding: 8px;
    min-width: 50px;
  }
  .tc-cell--selected {
    background: #896755;
  }

  /* Warning styles */
  .cdx-warning {
    background: #27272a;
    border-left: 4px solid #f59e0b;
    padding: 16px;
  }

  /* Checklist styles */
  .cdx-checklist__item {
    padding: 8px 0;
  }
  .cdx-checklist__item--checked .cdx-checklist__item-content {
    text-decoration: line-through;
    opacity: 0.6;
  }

  /* Embed styles */
  .embed-tool {
    width: 100%;
  }
  .embed-tool__content {
    width: 100%;
    aspect-ratio: 16/9;
  }

  /* Attaches styles */
  .cdx-attaches {
    background: #27272a;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 12px;
  }

  /* Raw tool styles */
  .raw-tool {
    background: #1a1a1a;
    padding: 16px;
    border-radius: 8px;
    font-family: monospace;
  }

  /* Marker/highlight styles */
  .cdx-marker {
    background: #f59e0b;
    color: #000;
    padding: 2px 4px;
    border-radius: 2px;
  }

  /* Link styles */
  .cdx-link {
    color: #60a5fa;
    text-decoration: underline;
  }
  .cdx-link:hover {
    color: #93c5fd;
  }
</style>
<script>
window.escapeHtml = function(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
};

window.jsonToEditorJs = function(jsonData) {
  try {
    const data = typeof jsonData === 'string' ? JSON.parse(jsonData) : jsonData;
    const blocks = [];

    // Check if data is already in Editor.js format (has blocks array)
    if (data.blocks && Array.isArray(data.blocks)) {
      // Already Editor.js format - return as-is to preserve all block types
      return JSON.stringify(data);
    }

    if (data.markdown && data.legacy_format) {
      blocks.push({
        type: 'paragraph',
        data: {
          text: data.markdown
        }
      });
      return JSON.stringify({ blocks: blocks, version: '2.28.0' });
    }

    if (Array.isArray(data)) {
      data.forEach(function(page) {
        if (page.parsing_res_list && Array.isArray(page.parsing_res_list)) {
          page.parsing_res_list.forEach(function(block) {
            window.convertBlockToEditorJs(block, blocks);
          });
        } else {
          window.convertPageToEditorJs(page, blocks);
        }
        blocks.push({ type: 'delimiter' });
      });
    } else if (data.parsing_res_list && Array.isArray(data.parsing_res_list)) {
      data.parsing_res_list.forEach(function(block) {
        window.convertBlockToEditorJs(block, blocks);
      });
    } else {
      window.convertPageToEditorJs(data, blocks);
    }

    return JSON.stringify({ blocks: blocks, version: '2.28.0' });
  } catch (e) {
    console.error('[jsonToEditorJs] Error:', e);
    return JSON.stringify({
      blocks: [{
        type: 'paragraph',
        data: { text: 'Error parsing JSON: ' + e.message }
      }],
      version: '2.28.0'
    });
  }
};

window.convertBlockToEditorJs = function(block, blocks) {
  if (block.text) {
    blocks.push({
      type: 'paragraph',
      data: { text: block.text }
    });
    return;
  }
  
  if (block.rec_texts && Array.isArray(block.rec_texts)) {
    block.rec_texts.forEach(function(t) {
      blocks.push({
        type: 'paragraph',
        data: { text: t }
      });
    });
    return;
  }
  
  if (block.content) {
    blocks.push({
      type: 'paragraph',
      data: { text: block.content }
    });
    return;
  }
  
  if (block.block_content) {
    const label = block.block_label || '';
    const content = block.block_content;
    
    if (label === 'title' || label === 'header' || label === 'paragraph_title') {
      blocks.push({
        type: 'header',
        data: { text: content, level: 2 }
      });
    } else if (label === 'text' || label === 'paragraph') {
      blocks.push({
        type: 'paragraph',
        data: { text: content }
      });
    } else if (label === 'table' || label === 'table_body') {
      if (content.indexOf('<table') >= 0 || content.indexOf('<tr') >= 0 || content.indexOf('<td') >= 0) {
        blocks.push({
          type: 'paragraph',
          data: { text: content }
        });
      } else {
        blocks.push({
          type: 'paragraph',
          data: { text: content }
        });
      }
    } else if (label === 'list') {
      blocks.push({
        type: 'list',
        data: { style: 'unordered', items: [content] }
      });
    } else if (label === 'number') {
      blocks.push({
        type: 'list',
        data: { style: 'ordered', items: [content] }
      });
    } else if (label === 'header_image') {
      if (content && content.trim()) {
        blocks.push({
          type: 'paragraph',
          data: { text: '[Image: ' + content + ']' }
        });
      }
    } else if (label === 'footer') {
      blocks.push({
        type: 'paragraph',
        data: { text: content }
      });
    } else if (label === 'formula') {
      blocks.push({
        type: 'paragraph',
        data: { text: '$$' + content + '$$' }
      });
    } else if (label === 'code') {
      blocks.push({
        type: 'code',
        data: { code: content }
      });
    } else if (label === 'blockquote') {
      blocks.push({
        type: 'quote',
        data: { text: content, caption: '', alignment: 'left' }
      });
    } else {
      blocks.push({
        type: 'paragraph',
        data: { text: content }
      });
    }
    return;
  }
  
  if (block.html) {
    blocks.push({
      type: 'paragraph',
      data: { text: block.html }
    });
    return;
  }
  
  blocks.push({
    type: 'code',
    data: { code: JSON.stringify(block, null, 2) }
  });
};

window.convertPageToEditorJs = function(pageData, blocks) {
  const res = pageData.res || pageData;
  
  if (res.rec_texts && Array.isArray(res.rec_texts)) {
    res.rec_texts.forEach(function(t) {
      blocks.push({
        type: 'paragraph',
        data: { text: t }
      });
    });
  }
  
  if (res.html) {
    blocks.push({
      type: 'paragraph',
      data: { text: res.html }
    });
  }
  
  if (res.rec_formula) {
    const formulas = Array.isArray(res.rec_formula) ? res.rec_formula : [res.rec_formula];
    formulas.forEach(function(f) {
      blocks.push({
        type: 'paragraph',
        data: { text: '$$' + f + '$$' }
      });
    });
  }
  
  if (res.result && typeof res.result === 'string') {
    blocks.push({
      type: 'code',
      data: { code: res.result }
    });
  }

  if (res.parsing_res_list && Array.isArray(res.parsing_res_list)) {
    res.parsing_res_list.forEach(function(block) {
      window.convertBlockToEditorJs(block, blocks);
    });
  }
  
  if (blocks.length === 0) {
    blocks.push({
      type: 'code',
      data: { code: JSON.stringify(res, null, 2) }
    });
  }
};

window.AlignmentTune = class AlignmentTune {
  static get isTune() {
    return true;
  }

  static get sanitize() {
    return {
      div: {
        'data-alignment': true
      }
    };
  }

  constructor({ api, data, config, block }) {
    this.api = api;
    this.block = block;
    this.config = config;
    this.data = data || { alignment: 'left' };
    this.wrapper = null;
  }

  render() {
    const alignments = ['left', 'center', 'right'];

    return alignments.map(alignment => ({
      icon: this.getAlignIcon(alignment),
      label: 'Align ' + alignment,
      isActive: this.data.alignment === alignment,
      closeOnActivate: true,
      onActivate: () => {
        this.data.alignment = alignment;
        this.updateAlignment();
      }
    }));
  }

  wrap(blockContent) {
    this.wrapper = document.createElement('div');
    this.wrapper.classList.add('alignment-tune');
    this.wrapper.dataset.alignment = this.data.alignment;
    this.wrapper.style.textAlign = this.data.alignment;
    this.wrapper.appendChild(blockContent);

    return this.wrapper;
  }

  updateAlignment() {
    if (this.wrapper) {
      this.wrapper.dataset.alignment = this.data.alignment;
      this.wrapper.style.textAlign = this.data.alignment;
    }
  }

  save() {
    return this.data;
  }

  getAlignIcon(alignment) {
    const icons = {
      left: '<svg width="18" height="18" viewBox="0 0 24 24"><path d="M3 21h18v-2H3v2zm0-4h12v-2H3v2zm0-4h18v-2H3v2zm0-4h12V7H3v2zm0-6v2h18V3H3z"/></svg>',
      center: '<svg width="18" height="18" viewBox="0 0 24 24"><path d="M7 21h10v-2H7v2zm-4-4h18v-2H3v2zm4-4h10v-2H7v2zm-4-4h18V7H3v2zm4-6v2h10V3H7z"/></svg>',
      right: '<svg width="18" height="18" viewBox="0 0 24 24"><path d="M3 21h18v-2H3v2zm6-4h12v-2H9v2zm-6-4h18v-2H3v2zm6-4h12V7H9v2zm-6-6v2h18V3H3z"/></svg>'
    };
    return icons[alignment];
  }
};
</script>
"""

# Mounts Editor.js on page load. Retries until the CDN script (and the
# gr.HTML div it targets) actually exist in the DOM, since Gradio renders
# client-side and there's no guaranteed ordering against the CDN <script>.
_EDITORJS_INIT_JS = """
() => {
  function initEditorJs() {
    const target = document.getElementById('editorjs');
    console.log('[EDITORJS_INIT] Looking for editorjs element:', target);
    console.log('[EDITORJS_INIT] EditorJS available:', typeof EditorJS !== 'undefined');
    if (!target || typeof EditorJS === 'undefined') {
      console.log('[EDITORJS_INIT] Retrying in 200ms...');
      setTimeout(initEditorJs, 200);
      return;
    }
    
    if (window.editorjsEditor) {
      console.log('[EDITORJS_INIT] Editor.js already initialized');
      return;
    }
 
    console.log('[EDITORJS_INIT] Initializing Editor.js');
    console.log('[EDITORJS_INIT] Available plugins:', {
      Header: typeof window.Header,
      List: typeof window.List,
      CodeTool: typeof window.CodeTool,
      Quote: typeof window.Quote,
      Delimiter: typeof window.Delimiter,
      Bold: typeof window.Bold,
      Italic: typeof window.Italic,
      Underline: typeof window.Underline,
      LinkTool: typeof window.LinkTool,
      Marker: typeof window.Marker,
      Table: typeof window.Table,
      EditorjsTable: typeof window.EditorjsTable,
      ImageTool: typeof window.ImageTool,
      SimpleImage: typeof window.SimpleImage,
      Checklist: typeof window.Checklist,
      Warning: typeof window.Warning,
      Embed: typeof window.Embed,
      AttachesTool: typeof window.AttachesTool,
      RawTool: typeof window.RawTool
    });

    const tools = {};
    if (typeof window.Header !== 'undefined') {
      tools.header = {
        class: window.Header,
        config: {
          levels: [1, 2, 3],
          defaultLevel: 2
        },
        tunes: ['alignment']
      };
    }
    if (typeof window.List !== 'undefined') {
      tools.list = {
        class: window.List,
        inlineToolbar: true,
        tunes: ['alignment']
      };
    }
    if (typeof window.CodeTool !== 'undefined') {
      tools.code = window.CodeTool;
    }
    if (typeof window.Quote !== 'undefined') {
      tools.quote = {
        class: window.Quote,
        inlineToolbar: true,
        tunes: ['alignment']
      };
    }
    if (typeof window.Delimiter !== 'undefined') {
      tools.delimiter = window.Delimiter;
    }
    if (typeof window.Bold !== 'undefined') {
      tools.bold = window.Bold;
    }
    if (typeof window.Italic !== 'undefined') {
      tools.italic = window.Italic;
    }
    if (typeof window.Underline !== 'undefined') {
      tools.underline = window.Underline;
    }
    if (typeof window.LinkTool !== 'undefined') {
      tools.link = window.LinkTool;
    }
    if (typeof window.Marker !== 'undefined') {
      tools.marker = window.Marker;
    }
    if (typeof window.Table !== 'undefined') {
      tools.table = {
        class: window.Table,
        inlineToolbar: true
      };
    } else if (typeof window.EditorjsTable !== 'undefined') {
      tools.table = {
        class: window.EditorjsTable,
        inlineToolbar: true
      };
    }
    if (typeof window.ImageTool !== 'undefined') {
      tools.image = {
        class: window.ImageTool,
        inlineToolbar: true
      };
    } else if (typeof window.SimpleImage !== 'undefined') {
      tools.image = {
        class: window.SimpleImage,
        inlineToolbar: true
      };
    }
    if (typeof window.Checklist !== 'undefined') {
      tools.checklist = {
        class: window.Checklist,
        inlineToolbar: true
      };
    }
    if (typeof window.Warning !== 'undefined') {
      tools.warning = window.Warning;
    }
    if (typeof window.Embed !== 'undefined') {
      tools.embed = window.Embed;
    }
    if (typeof window.AttachesTool !== 'undefined') {
      tools.attaches = window.AttachesTool;
    }
    if (typeof window.RawTool !== 'undefined') {
      tools.raw = window.RawTool;
    }
    if (typeof window.AlignmentTune !== 'undefined') {
      tools.alignment = window.AlignmentTune;
    }

    window.editorjsEditor = new EditorJS({
      holder: 'editorjs',
      placeholder: 'Run a pipeline or pick a file from the workspace history to edit its output...',
      tools: tools,
      data: {
        blocks: []
      },
      onChange: () => {
        if (window.editorjsHistory && !window.editorjsHistory._restoring) {
          window.editorjsHistory._debouncedSave();
        }
        window.editorjsEditor.save().then((outputData) => {
          const hidden = document.querySelector('#editorjs_hidden_content textarea');
          if (hidden) {
            hidden.value = JSON.stringify(outputData);
            hidden.dispatchEvent(new Event('input', { bubbles: true }));
          }
        }).catch((error) => {
          console.error('Saving failed: ', error);
        });
      }
    });
    class EditorHistory {
      constructor(editor, maxStack) {
        this.editor = editor;
        this.undoStack = [];
        this.redoStack = [];
        this.maxStack = maxStack || 100;
        this._restoring = false;
        this._debounceTimer = null;
        this.undoBtn = document.getElementById('editorjs-undo-btn');
        this.redoBtn = document.getElementById('editorjs-redo-btn');
        this._bindButtons();
        this._bindKeyboard();
      }
      _bindButtons() {
        const self = this;
        if (this.undoBtn) {
          this.undoBtn.addEventListener('click', () => { self.undo(); });
        }
        if (this.redoBtn) {
          this.redoBtn.addEventListener('click', () => { self.redo(); });
        }
      }
      _bindKeyboard() {
        const self = this;
        document.addEventListener('keydown', (e) => {
          const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
          const mod = isMac ? e.metaKey : e.ctrlKey;
          if (mod && e.key === 'z' && !e.shiftKey) {
            e.preventDefault();
            self.undo();
          } else if (mod && e.key === 'z' && e.shiftKey) {
            e.preventDefault();
            self.redo();
          } else if (mod && e.key === 'y') {
            e.preventDefault();
            self.redo();
          }
        });
      }
      _updateButtons() {
        if (this.undoBtn) this.undoBtn.disabled = this.undoStack.length === 0;
        if (this.redoBtn) this.redoBtn.disabled = this.redoStack.length === 0;
      }
      _debouncedSave() {
        if (this._restoring) return;
        clearTimeout(this._debounceTimer);
        const self = this;
        this._debounceTimer = setTimeout(() => {
          self._snapshot();
        }, 300);
      }
      async _snapshot() {
        if (this._restoring) return;
        try {
          const data = await this.editor.save();
          const last = this.undoStack[this.undoStack.length - 1];
          if (last && JSON.stringify(last) === JSON.stringify(data)) return;
          this.undoStack.push(data);
          if (this.undoStack.length > this.maxStack) {
            this.undoStack.shift();
          }
          this.redoStack = [];
          this._updateButtons();
        } catch (err) {
          console.error('[EditorHistory] snapshot failed:', err);
        }
      }
      async undo() {
        if (this.undoStack.length === 0) return;
        this._restoring = true;
        try {
          const current = await this.editor.save();
          this.redoStack.push(current);
          const prev = this.undoStack.pop();
          await this.editor.render(prev);
        } catch (err) {
          console.error('[EditorHistory] undo failed:', err);
        }
        this._restoring = false;
        this._updateButtons();
        this._syncHidden();
      }
      async redo() {
        if (this.redoStack.length === 0) return;
        this._restoring = true;
        try {
          const current = await this.editor.save();
          this.undoStack.push(current);
          const next = this.redoStack.pop();
          await this.editor.render(next);
        } catch (err) {
          console.error('[EditorHistory] redo failed:', err);
        }
        this._restoring = false;
        this._updateButtons();
        this._syncHidden();
      }
      async initialize(data) {
        this.undoStack = [];
        this.redoStack = [];
        this._restoring = false;
        try {
          const current = await this.editor.save();
          this.undoStack.push(current);
          this._updateButtons();
        } catch (err) {
          console.error('[EditorHistory] initialize failed:', err);
        }
      }
      _syncHidden() {
        this.editor.save().then((outputData) => {
          const hidden = document.querySelector('#editorjs_hidden_content textarea');
          if (hidden) {
            hidden.value = JSON.stringify(outputData);
            hidden.dispatchEvent(new Event('input', { bubbles: true }));
          }
        });
      }
    }
    window.editorjsHistory = new EditorHistory(window.editorjsEditor, 100);
    console.log('[EDITORJS_INIT] Custom undo/redo manager enabled');
    console.log('[EDITORJS_INIT] Editor.js initialized successfully');
  }
  setTimeout(initEditorJs, 300);
}
"""

# Pulls whatever Python just wrote into the hidden textbox and pushes it
# INTO the Editor.js editor, running it through jsonToEditorJs to convert JSON
# to Editor.js format for display. Chain this with .then() right after any event that
# updates editorjs_hidden from the backend (process_file, load_from_library).
_PUSH_INTO_EDITORJS_JS = """
() => {
  const hidden = document.querySelector('#editorjs_hidden_content textarea');
  if (!hidden) {
    console.warn('[editorjs bridge] hidden textarea (#editorjs_hidden_content) not found in DOM');
    return;
  }
  if (!window.editorjsEditor) {
    console.warn('[editorjs bridge] editorjsEditor not initialized yet');
    return;
  }
  const raw = hidden.value || '';
  console.log('[PUSH_INTO_EDITORJS] Hidden textarea found:', hidden);
  console.log('[PUSH_INTO_EDITORJS] Hidden textarea value length:', raw.length);
  console.log('[PUSH_INTO_EDITORJS] Hidden textarea value preview:', raw.substring(0, 200));
  console.log('[PUSH_INTO_EDITORJS] Hidden textarea value empty:', raw.length === 0);
   
   if (raw.length === 0) {
      console.warn('[PUSH_INTO_EDITORJS] Hidden textarea is empty - nothing to push');
      return;
   }
   
  const editorJsData = window.jsonToEditorJs ? window.jsonToEditorJs(raw) : raw;
  console.log('[PUSH_INTO_EDITORJS] Generated Editor.js data length:', editorJsData.length);
  console.log('[PUSH_INTO_EDITORJS] Generated Editor.js data preview:', editorJsData.substring(0, 200));
  console.log('[PUSH_INTO_EDITORJS] Rendering data into editor');
  
  try {
    const data = typeof editorJsData === 'string' ? JSON.parse(editorJsData) : editorJsData;
    window.editorjsEditor.render(data).then(() => {
      console.log('[PUSH_INTO_EDITORJS] Data rendered successfully');
      if (window.editorjsHistory) {
        window.editorjsHistory.initialize(data);
        console.log('[PUSH_INTO_EDITORJS] Undo history re-initialized with new data');
      }
      const undoBtn = document.getElementById('editorjs-undo-btn');
      const redoBtn = document.getElementById('editorjs-redo-btn');
      if (undoBtn) undoBtn.disabled = true;
      if (redoBtn) redoBtn.disabled = true;
    }).catch((error) => {
      console.error('[PUSH_INTO_EDITORJS] Render failed:', error);
    });
  } catch (e) {
    console.error('[PUSH_INTO_EDITORJS] JSON parse error:', e);
  }
}
"""

# Pulls the CURRENT Editor.js data out into the hidden textbox. Run this
# before any event that needs to read the latest edited content on the
# Python side (e.g. Save).
_PULL_FROM_EDITORJS_JS = """
() => {
  const hidden = document.querySelector('#editorjs_hidden_content textarea');
  if (hidden && window.editorjsEditor) {
    window.editorjsEditor.save().then((outputData) => {
      hidden.value = JSON.stringify(outputData);
      hidden.dispatchEvent(new Event('input', { bubbles: true }));
    }).catch((error) => {
      console.error('Saving failed: ', error);
    });
  }
}
"""

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_demo():
    with gr.Blocks(title="PaddleOCR-VL Workspace", head=EDITOR_HEAD) as demo:
        gr.Markdown(
            "## PaddleOCR-VL Local Workspace\n"
            "Upload a PDF or image, pick a pipeline, and edit the extracted result. "
            "Files and results persist in `/mnt/output` between sessions."
        )

        current_record = gr.State(value=None)

        # --- Controls, all up top: input + pipeline + run on one line,
        # status + workspace history on the next. ---
        with gr.Row():
            file_input = gr.File(label="Upload PDF or Image", scale=2)
            data_dir_dropdown = gr.Dropdown(
                choices=list_data_dir(), label="...or pick a file already in /mnt/data", scale=2
            )
            refresh_data_button = gr.Button("↻", scale=0, min_width=40)
            pipeline_selector = gr.Dropdown(
                choices=list(PIPELINES.keys()), value="Document Parser", label="Pipeline", scale=1
            )
            device_selector = gr.Radio(
                choices=["gpu", "cpu"], value="gpu", label="Device", scale=1,
                info="gpu is the default; switch to cpu only for debugging."
            )
            run_button = gr.Button("Run", variant="primary", scale=1)

        with gr.Row():
            status_box = gr.Textbox(label="Status", lines=2, interactive=False, scale=2)
            library_dropdown = gr.Dropdown(
                choices=library_choices(load_index()), label="Workspace history", interactive=True, scale=2
            )

        # --- Below: original on the left, combined preview+edit on the right. ---
        with gr.Row():
            with gr.Column(scale=1):
                image_preview = gr.Image(label="Original", visible=False, height=560)
                file_preview = gr.File(label="Original file", visible=True)

            with gr.Column(scale=2):
                # Editor.js renders JSON blocks directly, so this one
                # pane replaces the separate Preview + Edit tabs -- what
                # you see is what you can immediately click into and edit.
                gr.HTML('<div id="editorjs-editor-wrap"><div id="editorjs-undo-bar"><button id="editorjs-undo-btn" disabled title="Undo (Ctrl+Z)">↩ Undo</button><button id="editorjs-redo-btn" disabled title="Redo (Ctrl+Shift+Z)">↪ Redo</button></div><div id="editorjs"></div></div>')
                # Bridge only -- never shown to the user. Editor.js's JSON
                # lives here so Python can read/write it.
                editorjs_hidden = gr.Textbox(elem_id="editorjs_hidden_content", visible=True)
                save_button = gr.Button("Save edits")

        # Mount Editor.js once, as soon as the page loads.
        demo.load(fn=None, js=_EDITORJS_INIT_JS)

        refresh_data_button.click(
            lambda: gr.update(choices=list_data_dir()), inputs=[], outputs=[data_dir_dropdown]
        )

        run_button.click(
            process_file,
            inputs=[file_input, data_dir_dropdown, pipeline_selector, device_selector],
            outputs=[status_box, library_dropdown, image_preview, file_preview, editorjs_hidden],
        ).then(
            # New content just landed in the hidden textbox -- render it into Editor.js.
            fn=None, js=_PUSH_INTO_EDITORJS_JS
        )

        # Any change to the dropdown (from the user picking a past file, OR
        # from process_file programmatically selecting the new record)
        # reloads that record's content and marks it as the "current"
        # record for saving.
        library_dropdown.change(
            load_from_library,
            inputs=[library_dropdown],
            outputs=[editorjs_hidden, image_preview, file_preview],
        ).then(
            fn=None, js=_PUSH_INTO_EDITORJS_JS
        )
        library_dropdown.change(lambda rid: rid, inputs=[library_dropdown], outputs=[current_record])

        save_button.click(
            # Grab the latest Editor.js data before running the Python save.
            fn=None, js=_PULL_FROM_EDITORJS_JS
        ).then(
            save_edit,
            inputs=[current_record, editorjs_hidden],
            outputs=[status_box, editorjs_hidden],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[OUTPUT_STRING_DIR])
    