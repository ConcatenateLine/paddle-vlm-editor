"""
PaddleOCR-VL Local Workspace
============================
A Gradio workspace for the official ``paddleocr-vl`` Docker image, styled
after the AI Studio document-parsing experience: pick a file, run a
pipeline, preview/edit the structured output, and keep a history of
everything you've processed.

Intended to run as ``web_client.py`` inside the paddleocr-vl container, per
``docker-compose.yml``.

Module layout
-------------
``web_client.py`` is the thin Gradio entry-point.  Business logic is
split across:

- ``config.py``        -- paths, constants, pipeline definitions
- ``persistence.py``   -- workspace-index CRUD and file-listing helpers
- ``runners.py``       -- per-pipeline result extraction (OCR, table, ...)
- ``pipeline_cache.py``-- pipeline factory, caching, and dispatch
- ``editor.py``        -- Editor.js asset loaders
- ``static/``          -- raw CSS / JS fragments loaded by ``editor.py``
- ``doc_parser_worker.py`` -- subprocess worker (unchanged)

Notes
-----
- Pipelines are lazily instantiated and cached in memory so switching
  files does NOT reload model weights every time.

- ``doc_parser`` runs in a dedicated subprocess to work around a known
  paddle static-graph crash bug.  See ``pipeline_cache.DocParserWorker``.

- Every result (original file + extracted output + raw JSON) is written
  under ``/mnt/output`` and indexed in ``index.json``.

- The Edit tab uses Editor.js loaded from a CDN for a block-based editor
  with JSON output.  A hidden ``gr.Textbox`` bridges JS <-> Python.
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
import uuid
from pathlib import Path

import gradio as gr

from config import (
    FILES_DIR,
    IMAGE_EXTS,
    PIPELINES,
    PREVIEW_SIZES,
    OUTPUT_STRING_DIR,
    DATA_DIR,
)
from editor import (
    load_editor_head,
    load_editor_init_js,
    load_push_js,
    load_pull_js,
)
from persistence import (
    add_record,
    find_record,
    library_choices,
    list_data_dir,
    load_index,
    update_record,
)
from pipeline_cache import run_pipeline_with_recovery

# ---------------------------------------------------------------------------
# Editor.js assets (loaded once at import time)
# ---------------------------------------------------------------------------
EDITOR_HEAD: str = load_editor_head()
_EDITORJS_INIT_JS: str = load_editor_init_js()
_PUSH_INTO_EDITORJS_JS: str = load_push_js()
_PULL_FROM_EDITORJS_JS: str = load_pull_js()


# ---------------------------------------------------------------------------
# Main processing entry point
# ---------------------------------------------------------------------------

def process_file(file, data_choice, pipeline_label, device, preview_size):
    # An uploaded file takes priority; otherwise fall back to whatever was
    # picked from the /mnt/data dropdown.
    if file is not None:
        source_path = Path(file.name)
    elif data_choice:
        source_path = DATA_DIR / data_choice
    else:
        return (
            "Upload a file or pick one from /mnt/data first.",
            gr.update(), gr.update(), gr.update(), None,
        )

    if not source_path.exists():
        return (
            f"File not found: {source_path}",
            gr.update(), gr.update(), gr.update(), None,
        )

    pipeline_key = PIPELINES[pipeline_label]
    record_id = uuid.uuid4().hex[:12]
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    original_name = source_path.name

    work_dir = FILES_DIR / record_id
    work_dir.mkdir(parents=True, exist_ok=True)
    stored_original = work_dir / original_name
    shutil.copy(source_path, stored_original)

    status_lines = [f"Running {pipeline_label} on {original_name}..."]
    height = PREVIEW_SIZES.get(preview_size, 560)

    try:
        output_json, output_format = run_pipeline_with_recovery(
            pipeline_key, device, str(stored_original), work_dir,
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
    preview_update = gr.update(
        value=str(stored_original) if is_image else None,
        visible=is_image,
        height=height,
    )

    # Generate HTML iframe for PDF preview
    if not is_image:
        file_url = f"/gradio_api/file={stored_original}"
        file_html = (
            f'<iframe src="{file_url}" width="100%" height="{height}" '
            f'style="border: none;"></iframe>'
        )
        file_update = gr.update(value=file_html, visible=True)
    else:
        file_update = gr.update(value=None, visible=False)

    return (
        "\n".join(status_lines),
        gr.update(choices=choices, value=record_id),
        preview_update,
        file_update,
        output_json,
    )


# ---------------------------------------------------------------------------
# Library / workspace history
# ---------------------------------------------------------------------------

def load_from_library(record_id, preview_size):
    height = PREVIEW_SIZES.get(preview_size, 560)
    if not record_id:
        return (
            "",
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=True),
        )

    record = find_record(record_id)
    if record is None:
        return (
            "Record not found.",
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=True),
        )

    if record["status"] != "done":
        text = f"This run failed:\n\n{record.get('error', 'unknown error')}"
        original = record["original_path"]
        is_image = Path(original).suffix.lower() in IMAGE_EXTS
        if is_image:
            return (
                text,
                gr.update(value=original if is_image else None, visible=is_image, height=height),
                gr.update(value=None, visible=False),
            )
        else:
            file_url = f"/gradio_api/file={original}"
            file_html = (
                f'<iframe src="{file_url}" width="100%" height="{height}" '
                f'style="border: none;"></iframe>'
            )
            return (
                text,
                gr.update(value=None, visible=False),
                gr.update(value=file_html, visible=True),
            )

    output_path = Path(record["output_path"])

    # Handle backward compatibility: if .json doesn't exist, try .md
    if not output_path.exists():
        md_path = output_path.parent / "result.md"
        if md_path.exists():
            output_path = md_path
            record["output_path"] = str(md_path)
            update_record(record_id, output_path=str(md_path))

    if not output_path.exists():
        return (
            f"Output file not found: {output_path}",
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=True),
        )

    content = output_path.read_text(encoding="utf-8", errors="replace")

    # If it's an old .md file, wrap it in a simple JSON structure for the editor
    if output_path.suffix == ".md":
        content = json.dumps({"markdown": content, "legacy_format": True})

    original = record["original_path"]
    is_image = Path(original).suffix.lower() in IMAGE_EXTS

    if is_image:
        return (
            content,
            gr.update(value=original if is_image else None, visible=is_image, height=height),
            gr.update(value=None, visible=False),
        )
    else:
        file_url = f"/gradio_api/file={original}"
        file_html = (
            f'<iframe src="{file_url}" width="100%" height="{height}" '
            f'style="border: none;"></iframe>'
        )
        return (
            content,
            gr.update(value=None, visible=False),
            gr.update(value=file_html, visible=True),
        )


# ---------------------------------------------------------------------------
# Preview size
# ---------------------------------------------------------------------------

def change_preview_size(size_label, current_record_id):
    height = PREVIEW_SIZES.get(size_label, 560)
    is_small = size_label == "Small"
    img_update = gr.update(height=height)

    file_update = gr.update()
    if current_record_id:
        record = find_record(current_record_id)
        if record and record["status"] == "done":
            original = record["original_path"]
            is_image = Path(original).suffix.lower() in IMAGE_EXTS
            if not is_image:
                file_url = f"/gradio_api/file={original}"
                file_html = (
                    f'<iframe src="{file_url}" width="100%" height="{height}" '
                    f'style="border: none;"></iframe>'
                )
                file_update = gr.update(value=file_html, visible=True)
            else:
                file_update = gr.update(value=None, visible=False)

    preview_col_update = gr.update(scale=1)
    editor_col_update = gr.update(scale=2 if is_small else 1)

    return height, img_update, file_update, preview_col_update, editor_col_update


# ---------------------------------------------------------------------------
# Editor bridge helpers
# ---------------------------------------------------------------------------

def pull_from_editorjs(edited_json):
    """Python wrapper that triggers the JavaScript pull and returns the updated JSON."""
    return edited_json


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

    Path(record["output_path"]).write_text(edited_json, encoding="utf-8")
    return f"Saved changes to {Path(record['output_path']).name}.", edited_json


def refresh_library():
    records = load_index()
    return gr.update(choices=library_choices(records))


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_demo():
    with gr.Blocks(title="PaddleOCR-VL Workspace", head=EDITOR_HEAD) as demo:
        gr.Markdown(
            "## PaddleOCR-VL Local Workspace\n"
            "Upload a PDF or image, pick a pipeline, and edit the extracted result. "
            "Files and results persist in `/mnt/output` between sessions."
        )

        current_record = gr.State(value=None)
        preview_size = gr.State(value=560)

        # --- Controls, all up top: input + pipeline + run on one line,
        # status + workspace history on the next. ---
        with gr.Row(equal_height=True, min_height=242):
            file_input = gr.File(label="Upload PDF or Image", scale=2)
            data_dir_dropdown = gr.Dropdown(
                choices=list_data_dir(),
                label="...or pick a file already in /mnt/data",
                scale=2,
            )
            refresh_data_button = gr.Button("\u21bb", scale=0, min_width=40)
            pipeline_selector = gr.Dropdown(
                choices=list(PIPELINES.keys()),
                value="Document Parser",
                label="Pipeline",
                scale=1,
            )
            device_selector = gr.Radio(
                choices=["gpu", "cpu"],
                value="gpu",
                label="Device",
                scale=1,
                info="gpu is the default; switch to cpu only for debugging.",
            )

        with gr.Row():
            status_box = gr.Textbox(label="Status", lines=2, interactive=False, scale=2)
            run_button = gr.Button("Run", variant="primary", scale=1)

        with gr.Sidebar(position="left"):
            gr.Markdown("# \U0001f43e Workspace history")
            gr.Markdown("Select a workspace to load")

            library_dropdown = gr.Dropdown(
                choices=library_choices(load_index()),
                value="",
                label="Workspace history",
                interactive=True,
                scale=1,
            )

            with gr.Row():
                size_selector = gr.Radio(
                    choices=["Small", "Normal (Actual)"],
                    value="Normal (Actual)",
                    label="Preview Size",
                    scale=0,
                    min_width=200,
                )

        # --- Below: original on the left, combined preview+edit on the right. ---
        with gr.Row():
            with gr.Column(elem_id="preview-column") as preview_column:
                save_button = gr.Button("Save edits")
                gr.HTML("<div class='column-divider'></div>")
                image_preview = gr.Image(label="Original", visible=False, height=560)
                file_preview = gr.HTML(label="Original file", visible=True)

            with gr.Column(elem_id="editorjs-column") as editor_column:
                gr.HTML(
                    '<div id="editorjs-editor-wrap">'
                    '<div id="editorjs-undo-bar">'
                    '<button id="editorjs-undo-btn" disabled title="Undo (Ctrl+Z)">\u21a9 Undo</button>'
                    '<button id="editorjs-redo-btn" disabled title="Redo (Ctrl+Shift+Z)">\u21aa Redo</button>'
                    '</div>'
                    '<div id="editorjs"></div>'
                    '</div>'
                )
                editorjs_hidden = gr.Textbox(elem_id="editorjs_hidden_content", visible=True)

        # Mount Editor.js once, as soon as the page loads.
        demo.load(fn=None, js=_EDITORJS_INIT_JS)

        refresh_data_button.click(
            lambda: gr.update(choices=list_data_dir()),
            inputs=[],
            outputs=[data_dir_dropdown],
        )

        run_button.click(
            process_file,
            inputs=[file_input, data_dir_dropdown, pipeline_selector, device_selector, preview_size],
            outputs=[status_box, library_dropdown, image_preview, file_preview, editorjs_hidden],
        ).then(
            fn=None, js=_PUSH_INTO_EDITORJS_JS,
        )

        library_dropdown.change(
            load_from_library,
            inputs=[library_dropdown, preview_size],
            outputs=[editorjs_hidden, image_preview, file_preview],
        ).then(
            fn=None, js=_PUSH_INTO_EDITORJS_JS,
        )
        library_dropdown.change(
            lambda rid: rid,
            inputs=[library_dropdown],
            outputs=[current_record],
        )

        size_selector.change(
            change_preview_size,
            inputs=[size_selector, current_record],
            outputs=[preview_size, image_preview, file_preview, preview_column, editor_column],
        )

        save_button.click(
            pull_from_editorjs,
            inputs=[editorjs_hidden],
            outputs=[editorjs_hidden],
            js=_PULL_FROM_EDITORJS_JS,
        ).then(
            save_edit,
            inputs=[current_record, editorjs_hidden],
            outputs=[status_box, editorjs_hidden],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[OUTPUT_STRING_DIR],
    )
