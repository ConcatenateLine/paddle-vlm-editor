"""
Editor.js asset loaders.

Reads the CSS/JS fragments from the ``static/`` directory and returns
them as plain strings suitable for embedding in Gradio's ``head=``
parameter or ``js=`` callback arguments.
"""

from __future__ import annotations

from pathlib import Path

_STATIC_DIR = Path(__file__).parent / "static"


def load_editor_head() -> str:
    """Return the full ``<head>`` content (CDN scripts + CSS)."""
    return (_STATIC_DIR / "editor_head.html").read_text(encoding="utf-8")


def load_editor_init_js() -> str:
    """Return the JS IIFE that mounts Editor.js on page load."""
    return (_STATIC_DIR / "editorjs_init.js").read_text(encoding="utf-8")


def load_push_js() -> str:
    """Return the JS that pushes hidden-textbox JSON into Editor.js."""
    return (_STATIC_DIR / "push_into.js").read_text(encoding="utf-8")


def load_pull_js() -> str:
    """Return the JS that pulls Editor.js data into the hidden textbox."""
    return (_STATIC_DIR / "pull_from.js").read_text(encoding="utf-8")
