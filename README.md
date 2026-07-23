# PaddleVLM Editor

> Self-hosted document parsing, OCR, and rich-text editing workspace
> powered by PaddleOCR-VL and Gradio.

PaddleVLM Editor is a local, Docker-based application that extracts structured content from PDFs and images using PaddleOCR-VL's Vision-Language Model, then lets you review and edit the results in a rich-text block editor — all without sending data to any cloud service.

---

## Overview

- **5 processing pipelines** in a single UI: Document Parser, OCR, Table Recognition, Formula Recognition, and Chart Parsing
- **Editor.js block editor** for reviewing and editing extracted content (headers, paragraphs, tables, images, code, lists, formulas)
- **Persistent workspace** — all results survive container restarts
- **PDF image extraction** via PyMuPDF (embedded images matched by IoU, fallback to page rendering)
- **GPU-accelerated** inference with automatic CPU fallback for debugging
- **Subprocess isolation** for PaddleOCR-VL to recover from known PaddlePaddle static-graph crashes

---

## Prerequisites

| Requirement | Details |
|---|---|
| **NVIDIA GPU** | Required for GPU mode. The VLM model needs ~2-3 GB VRAM minimum. |
| **NVIDIA Container Toolkit** | Must be installed so Docker can access the GPU (`nvidia-container-toolkit` package). |
| **Docker** | Version 20.10 or later. |
| **Docker Compose** | V2 (the `docker compose` command, not the legacy `docker-compose`). |
| **Disk space** | ~5 GB for the PaddleOCR-VL model cache on first run. Grows with processed output. |
| **Network access** | One-time pull from Baidu's container registry. Runtime access to `cdn.jsdelivr.net` for Editor.js plugins (graceful degradation if blocked). |

> **CPU-only mode** is available in the UI for debugging, but inference will be significantly slower.

---

## Quick Start

```bash
# 1. Clone the repository
git clone <your-repo-url> paddle-vlm-editor
cd paddle-vlm-editor

# 2. Create the required directories
mkdir -p data output paddlex_cache

# 3. Start the application
docker compose up -d

# 4. Open in your browser
#    http://localhost:7860
```

On first launch, the container pulls the PaddleOCR-VL model weights (~2 GB) into `./paddlex_cache`. Subsequent starts reuse the cached weights.

---

## Usage

### Uploading Files

There are two ways to provide input files:

1. **Upload via the UI** — Click "Upload PDF or Image" and select a file.
2. **Drop files in `./data`** — Files placed in the `./data` directory appear in the "pick a file already in /mnt/data" dropdown. Use the refresh button after adding files.

### Supported File Types

| Type | Extensions |
|---|---|
| Images | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.tif`, `.tiff` |
| Documents | `.pdf` |

### Running a Pipeline

1. Select a pipeline from the **Pipeline** dropdown.
2. Select **gpu** or **cpu** as the device.
3. Click **Run**.
4. The extracted content appears in the Editor.js block editor below.

### Pipelines

| Pipeline | Input | Output | Description |
|---|---|---|---|
| **Document Parser** | PDF, Image | Structured JSON with typed blocks | Full-page intelligent parsing via PaddleOCR-VL. Extracts text, headers, tables, images, formulas, lists, code blocks, and more — each with bounding boxes and block labels. |
| **OCR** | PDF, Image | JSON with text lines and detection polygons | Plain text line extraction. Best for simple text-only documents. |
| **Table Recognition** | PDF, Image | JSON + HTML tables | Extracts tables from documents and outputs structured HTML. |
| **Formula Recognition** | PDF, Image | JSON with LaTeX strings | Extracts mathematical formulas as LaTeX. |
| **Chart Parsing** | Image only | JSON with chart data | Converts chart images into data tables. **Does not accept PDFs.** |

### Editing Results

After processing, the Editor.js block editor displays the extracted content. You can:

- Edit text in any block (click to type)
- Add new blocks via the `+` button
- Rearrange blocks by dragging
- Use inline formatting: **bold**, *italic*, underline, links, highlight
- Undo / Redo with `Ctrl+Z` / `Ctrl+Shift+Z`
- Click **Save Edits** to persist your changes to `result.json`

### Workspace History

All processed files appear in the **Workspace** dropdown, sorted newest-first. Selecting a past entry reloads its content into the editor. The workspace index (`/mnt/output/index.json`) persists across container restarts.

---

## Architecture

### Project Structure

```
paddle-vlm-editor/
├── web_client.py           # Gradio entry-point (thin UI wiring)
├── config.py               # Constants, paths, pipeline definitions
├── persistence.py          # Workspace-index CRUD and file-listing
├── runners.py              # Per-pipeline result extraction
├── pipeline_cache.py       # Pipeline factory, caching, subprocess dispatch
├── editor.py               # Editor.js asset loaders
├── static/                 # CSS / JS fragments loaded by editor.py
│   ├── editor_head.html    # CDN scripts + styles for <head>
│   ├── editorjs_init.js    # Editor.js mount logic
│   ├── push_into.js        # JS bridge: hidden textbox -> Editor.js
│   └── pull_from.js        # JS bridge: Editor.js -> hidden textbox
├── doc_parser_worker.py    # Subprocess worker for PaddleOCR-VL pipeline
├── docker-compose.yml      # Docker Compose configuration
├── data/                   # Input files (mounted as /mnt/data)
├── output/                 # Persistent results (mounted as /mnt/output)
│   ├── index.json          # Workspace history index
│   └── files/
│       └── <record_id>/
│           ├── <original_file>
│           ├── result.json        # Editor-compatible output
│           ├── result_raw.json    # Unmodified pipeline output
│           ├── page_NNN/          # Per-page raw results
│           └── images/            # Extracted image blocks
└── paddlex_cache/          # Model weight cache (mounted as /home/paddleocr/.paddlex)
```

### Volume Mounts

| Host Path | Container Path | Purpose |
|---|---|---|
| `./data` | `/mnt/data` | Input files directory |
| `./output` | `/mnt/output` | Persistent results storage |
| `./web_client.py` | `/workspace/web_client.py` | Gradio entry-point |
| `./config.py` | `/workspace/config.py` | Constants and paths |
| `./persistence.py` | `/workspace/persistence.py` | Workspace index CRUD |
| `./runners.py` | `/workspace/runners.py` | Pipeline runners |
| `./pipeline_cache.py` | `/workspace/pipeline_cache.py` | Pipeline cache and dispatch |
| `./editor.py` | `/workspace/editor.py` | Editor.js asset loaders |
| `./doc_parser_worker.py` | `/workspace/doc_parser_worker.py` | Subprocess worker script |
| `./static` | `/workspace/static` | CSS / JS assets |
| `./paddlex_cache` | `/home/paddleocr/.paddlex` | Model weight cache |

### Document Parser Subprocess

The Document Parser pipeline runs PaddleOCR-VL in a dedicated OS subprocess (`doc_parser_worker.py`) rather than in-process. This is necessary because PaddlePaddle has a known bug:

> Reusing a `PaddleOCRVL` instance for a second `.predict()` call can crash with:
> `int(Tensor) is not supported in static graph mode`

After this crash, PaddlePaddle's global GPU state is corrupted and cannot be recovered in-process. A fresh OS process is the only reliable way to give PaddlePaddle a clean slate. The worker is **lazily spawned** on first use and **torn down and respawned** on any failure.

All other pipelines (OCR, Table, Formula, Chart) run in-process with model caching.

### Image Extraction (Document Parser)

When processing documents, the pipeline extracts images in two ways:

**For PDFs:**
1. Uses PyMuPDF to enumerate embedded images and their placement rects
2. Matches PaddleOCR-VL's detected image blocks to embedded images by IoU (Intersection over Union) with a threshold of 0.3
3. Matched images are extracted directly from the PDF's embedded image data
4. Unmatched blocks fall back to rendering the corresponding page region

> **Note:** PaddleOCR-VL renders PDF pages at 144 DPI (2x the 72-DPI PDF coordinate system). The extraction code automatically scales bounding box coordinates by 0.5x to convert from OCR pixel space to PDF point space.

**For images (PNG/JPG/etc.):**
- Crops directly from the source image using PaddleOCR-VL's bounding box coordinates (which are in image pixel space)

---

## Configuration

### Paths

| Path | Value | Description |
|---|---|---|
| Input directory | `/mnt/data` (host: `./data`) | Drop files here for the dropdown picker |
| Output directory | `/mnt/output` (host: `./output`) | All results stored here |
| Workspace index | `/mnt/output/index.json` | Master index of processed files |
| Model cache | `/home/paddleocr/.paddlex` (host: `./paddlex_cache`) | PaddleOCR-VL model weights |

### Timeouts

| Timeout | Value | Description |
|---|---|---|
| `STARTUP_TIMEOUT` | 600 seconds | Max wait for the subprocess to load the model |
| `PREDICT_TIMEOUT` | 900 seconds | Max wait for a single prediction |

### Port

| Port | Description |
|---|---|
| `7860` | Gradio web UI (HTTP, bound to all interfaces) |

---

## Known Issues & Troubleshooting

### GPU Out of Memory

If the GPU runs out of memory during inference, you'll see a `ResourceExhaustedError`. Solutions:

- Close other GPU-using processes
- Reduce input file size
- Switch to **cpu** mode in the UI (slower but works)

### Static-Graph Crash (Auto-Recovered)

If PaddleOCR-VL hits the `int(Tensor) is not supported in static graph mode` error, the application automatically:

1. Detects the failure
2. Kills the corrupted subprocess
3. Spawns a fresh one on the next request

No manual intervention is needed — just retry the request.

### Editor.js Not Loading

Editor.js plugins are loaded from `cdn.jsdelivr.net` at runtime. If this CDN is blocked or unreachable:

- The processing pipelines still work (output is saved as JSON)
- The block editor simply won't render
- You can still access results directly in `/mnt/output/files/<record_id>/result.json`

### CPU Mode

Available in the UI as a debugging option. Inference is significantly slower (minutes vs seconds per page). Only recommended when GPU is unavailable or for debugging pipeline behavior.

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | Application code |
| ML Framework | PaddlePaddle | GPU-accelerated inference engine |
| OCR / VLM | PaddleOCR-VL 1.6 (0.9B) | Vision-Language Model for document parsing |
| Model Management | PaddleX | Pipeline loading and model caching |
| Web UI | Gradio 6.20.0 | Interactive web interface |
| Rich-Text Editor | Editor.js | Block-based content editing |
| PDF Processing | PyMuPDF (fitz) | PDF image extraction and page rendering |
| Image Processing | Pillow | Image cropping for non-PDF sources |
| Containerization | Docker + NVIDIA Container Toolkit | GPU-enabled deployment |

---

## License

See `LICENSE` file in the PaddleOCR-VL base image for model licensing terms.
