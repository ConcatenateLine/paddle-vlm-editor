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
  first run of a given pipeline pays that cost.
- Every result (original file + extracted output + raw JSON) is written
  under /mnt/output and indexed in index.json, so the file list on the
  left survives container restarts (as long as ./output is bind-mounted).
- Files dropped into ./data on the host (-> /mnt/data in the container)
  show up in the "Pick from /mnt/data" dropdown, so you don't have to
  re-upload large PDFs through the browser each time.
"""

import gradio as gr
import json
import shutil
import time
import traceback
import uuid
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Pipeline cache -- models are only loaded once per process, on first use.
# ---------------------------------------------------------------------------
_PIPELINE_CACHE = {}


def get_pipeline(pipeline_key: str, device: str):
    cache_key = f"{pipeline_key}:{device}"
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    if pipeline_key == "doc_parser":
        from paddleocr import PaddleOCRVL
        pipeline = PaddleOCRVL(device=device)

    elif pipeline_key == "ocr":
        from paddleocr import PaddleOCR
        pipeline = PaddleOCR(
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )

    elif pipeline_key == "table_recognition_v2":
        from paddleocr import TableRecognitionPipelineV2
        pipeline = TableRecognitionPipelineV2(device=device)

    elif pipeline_key == "formula_recognition":
        from paddleocr import FormulaRecognitionPipeline
        pipeline = FormulaRecognitionPipeline(device=device)

    elif pipeline_key == "chart_parsing":
        from paddleocr import ChartParsing
        pipeline = ChartParsing(model_name="PP-Chart2Table", device=device)

    else:
        raise ValueError(f"Unknown pipeline: {pipeline_key}")

    _PIPELINE_CACHE[cache_key] = pipeline
    return pipeline


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
# ---------------------------------------------------------------------------
def _read_first_matching(directory: Path, suffix: str) -> str:
    matches = sorted(directory.glob(f"*{suffix}"))
    if not matches:
        return ""
    return matches[0].read_text(encoding="utf-8", errors="replace")


def run_doc_parser(pipeline, input_path: str, work_dir: Path):
    outputs = pipeline.predict(input_path)
    md_parts, json_parts = [], []
    for i, res in enumerate(outputs):
        page_dir = work_dir / f"page_{i:03d}"
        page_dir.mkdir(exist_ok=True)
        res.save_to_markdown(save_path=str(page_dir))
        res.save_to_json(save_path=str(page_dir))
        md_parts.append(_read_first_matching(page_dir, ".md"))
        json_parts.append(_read_first_matching(page_dir, ".json"))
    combined_md = "\n\n---\n\n".join(p for p in md_parts if p)
    combined_json = "[\n" + ",\n".join(p for p in json_parts if p) + "\n]"
    return combined_md, combined_json, "markdown"


def run_ocr(pipeline, input_path: str, work_dir: Path):
    outputs = pipeline.predict(input_path)
    text_parts, json_parts = [], []
    for i, res in enumerate(outputs):
        page_dir = work_dir / f"page_{i:03d}"
        page_dir.mkdir(exist_ok=True)
        res.save_to_json(save_path=str(page_dir))
        raw_json = _read_first_matching(page_dir, ".json")
        json_parts.append(raw_json)
        try:
            data = json.loads(raw_json)
            texts = data.get("res", {}).get("rec_texts", [])
            text_parts.append("\n".join(texts))
        except Exception:
            pass
    combined_text = "\n\n".join(text_parts)
    combined_json = "[\n" + ",\n".join(p for p in json_parts if p) + "\n]"
    return combined_text, combined_json, "text"


def run_table(pipeline, input_path: str, work_dir: Path):
    outputs = pipeline.predict(input_path)
    html_parts, json_parts = [], []
    for i, res in enumerate(outputs):
        page_dir = work_dir / f"page_{i:03d}"
        page_dir.mkdir(exist_ok=True)
        res.save_to_html(save_path=str(page_dir))
        res.save_to_json(save_path=str(page_dir))
        html_parts.append(_read_first_matching(page_dir, ".html"))
        json_parts.append(_read_first_matching(page_dir, ".json"))
    combined_html = "\n\n".join(h for h in html_parts if h)
    combined_json = "[\n" + ",\n".join(p for p in json_parts if p) + "\n]"
    return combined_html, combined_json, "html"


def run_formula(pipeline, input_path: str, work_dir: Path):
    outputs = pipeline.predict(input_path)
    text_parts, json_parts = [], []
    for i, res in enumerate(outputs):
        page_dir = work_dir / f"page_{i:03d}"
        page_dir.mkdir(exist_ok=True)
        res.save_to_json(save_path=str(page_dir))
        raw_json = _read_first_matching(page_dir, ".json")
        json_parts.append(raw_json)
        try:
            data = json.loads(raw_json)
            formulas = data.get("res", {}).get("rec_formula", [])
            if isinstance(formulas, str):
                formulas = [formulas]
            text_parts.append("\n\n".join(f"$$\n{f}\n$$" for f in formulas))
        except Exception:
            pass
    combined_text = "\n\n---\n\n".join(text_parts)
    combined_json = "[\n" + ",\n".join(p for p in json_parts if p) + "\n]"
    return combined_text, combined_json, "markdown"


def run_chart(pipeline, input_path: str, work_dir: Path):
    ext = Path(input_path).suffix.lower()
    if ext not in IMAGE_EXTS:
        raise ValueError(
            "Chart Parsing only accepts single chart images (png/jpg/etc), not PDFs. "
            "Crop the chart out of the PDF first, or use Document Parser on the whole page."
        )
    outputs = pipeline.predict(input={"image": input_path}, batch_size=1)
    text_parts, json_parts = [], []
    for i, res in enumerate(outputs):
        page_dir = work_dir / f"page_{i:03d}"
        page_dir.mkdir(exist_ok=True)
        res.save_to_json(save_path=str(page_dir / "res.json"))
        raw_json = _read_first_matching(page_dir, ".json")
        json_parts.append(raw_json)
        try:
            data = json.loads(raw_json)
            text_parts.append(data.get("res", {}).get("result", ""))
        except Exception:
            pass
    combined_text = "\n\n".join(text_parts)
    combined_json = "[\n" + ",\n".join(p for p in json_parts if p) + "\n]"
    return combined_text, combined_json, "text"


RUNNERS = {
    "doc_parser": run_doc_parser,
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
        return "Upload a file or pick one from /mnt/data first.", gr.update(), gr.update(), gr.update(), None, None

    if not source_path.exists():
        return f"File not found: {source_path}", gr.update(), gr.update(), gr.update(), None, None

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
        pipeline = get_pipeline(pipeline_key, device)
        runner = RUNNERS[pipeline_key]
        output_text, output_json, output_format = runner(pipeline, str(stored_original), work_dir)

        output_path = work_dir / "result.md"
        output_path.write_text(output_text, encoding="utf-8")
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
        output_text = ""

    choices = library_choices(records)
    is_image = Path(stored_original).suffix.lower() in IMAGE_EXTS
    preview_update = gr.update(value=str(stored_original) if is_image else None, visible=is_image)
    file_update = gr.update(value=str(stored_original) if not is_image else None, visible=not is_image)

    return (
        "\n".join(status_lines),
        gr.update(choices=choices, value=record_id),
        preview_update,
        file_update,
        output_text,
        output_text,
    )


def load_from_library(record_id):
    if not record_id:
        return "", "", None, gr.update(value=None, visible=False), gr.update(value=None, visible=True)

    record = find_record(record_id)
    if record is None:
        return "Record not found.", "", None, gr.update(value=None, visible=False), gr.update(value=None, visible=True)

    if record["status"] != "done":
        text = f"This run failed:\n\n{record.get('error', 'unknown error')}"
        original = record["original_path"]
        is_image = Path(original).suffix.lower() in IMAGE_EXTS
        return (
            text,
            text,
            gr.update(value=original if is_image else None, visible=is_image),
            gr.update(value=original if not is_image else None, visible=not is_image),
        )

    output_text = Path(record["output_path"]).read_text(encoding="utf-8", errors="replace")
    original = record["original_path"]
    is_image = Path(original).suffix.lower() in IMAGE_EXTS

    return (
        output_text,
        output_text,
        gr.update(value=original if is_image else None, visible=is_image),
        gr.update(value=original if not is_image else None, visible=not is_image),
    )


def save_edit(record_id, edited_text):
    if not record_id:
        return "Nothing selected to save.", edited_text
    record = find_record(record_id)
    if record is None or record.get("output_path") is None:
        return "This entry has no output file to save to.", edited_text
    Path(record["output_path"]).write_text(edited_text, encoding="utf-8")
    return f"Saved changes to {Path(record['output_path']).name}.", edited_text


def refresh_library():
    records = load_index()
    return gr.update(choices=library_choices(records))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="PaddleOCR-VL Workspace") as demo:
    gr.Markdown(
        "## PaddleOCR-VL Local Workspace\n"
        "Upload a PDF or image, pick a pipeline, and edit the extracted result. "
        "Files and results persist in `/mnt/output` between sessions."
    )

    current_record = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload PDF or Image")
            with gr.Row():
                data_dir_dropdown = gr.Dropdown(
                    choices=list_data_dir(), label="...or pick a file already in /mnt/data",
                    scale=4,
                )
                refresh_data_button = gr.Button("↻", scale=1)
            pipeline_selector = gr.Dropdown(
                choices=list(PIPELINES.keys()), value="Document Parser", label="Pipeline"
            )
            device_selector = gr.Radio(
                choices=["gpu", "cpu"], value="gpu", label="Device",
                info="This container reserves an NVIDIA GPU, so gpu is the default. Switch to cpu only for debugging."
            )
            run_button = gr.Button("Run", variant="primary")
            status_box = gr.Textbox(label="Status", lines=3, interactive=False)

            gr.Markdown("### Workspace history")
            library_dropdown = gr.Dropdown(
                choices=library_choices(load_index()), label="Previous files", interactive=True
            )

        with gr.Column(scale=2):
            with gr.Row():
                image_preview = gr.Image(label="Original", visible=False, height=280)
                file_preview = gr.File(label="Original file", visible=True)

            with gr.Tabs():
                with gr.Tab("Preview"):
                    output_preview = gr.Markdown(label="Rendered output")
                with gr.Tab("Edit"):
                    output_edit = gr.Textbox(
                        label="Editable output", lines=22, buttons=["copy"]
                    )
                    save_button = gr.Button("Save edits")

    refresh_data_button.click(
        lambda: gr.update(choices=list_data_dir()), inputs=[], outputs=[data_dir_dropdown]
    )

    run_button.click(
        process_file,
        inputs=[file_input, data_dir_dropdown, pipeline_selector, device_selector],
        outputs=[status_box, library_dropdown, image_preview, file_preview, output_preview, output_edit],
    )

    # Any change to the dropdown (from the user picking a past file, OR from
    # process_file programmatically selecting the new record) reloads that
    # record's content and marks it as the "current" record for saving.
    library_dropdown.change(
        load_from_library,
        inputs=[library_dropdown],
        outputs=[output_preview, output_edit, image_preview, file_preview],
    )
    library_dropdown.change(lambda rid: rid, inputs=[library_dropdown], outputs=[current_record])

    save_button.click(
        save_edit,
        inputs=[current_record, output_edit],
        outputs=[status_box, output_edit],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[OUTPUT_STRING_DIR])
    

