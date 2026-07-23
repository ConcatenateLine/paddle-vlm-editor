"""
Workspace index persistence.

Every processed file is recorded in ``index.json`` under ``/mnt/output``.
This module provides the CRUD helpers and the dropdown-list builder that
the Gradio UI calls into.
"""

from __future__ import annotations

import json
from typing import Optional

from config import DATA_DIR, INDEX_PATH, SUPPORTED_EXTS

# ---------------------------------------------------------------------------
# Index CRUD
# ---------------------------------------------------------------------------

def load_index() -> list[dict]:
    """Return the full record list (newest first), or an empty list."""
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return []


def save_index(records: list[dict]) -> None:
    """Write *records* back to ``index.json``."""
    INDEX_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def add_record(record: dict) -> list[dict]:
    """Insert *record* at the front of the index and persist."""
    records = load_index()
    records.insert(0, record)
    save_index(records)
    return records


def update_record(record_id: str, **fields) -> list[dict]:
    """Merge *fields* into the record with the given *record_id*."""
    records = load_index()
    for r in records:
        if r["id"] == record_id:
            r.update(fields)
            break
    save_index(records)
    return records


def find_record(
    record_id: str,
    records: Optional[list[dict]] = None,
) -> Optional[dict]:
    """Look up a single record by id."""
    records = records if records is not None else load_index()
    for r in records:
        if r["id"] == record_id:
            return r
    return None


# ---------------------------------------------------------------------------
# Data directory scanning
# ---------------------------------------------------------------------------

def list_data_dir() -> list[str]:
    """Return relative paths of supported files inside ``/mnt/data``."""
    if not DATA_DIR.exists():
        return []
    paths = sorted(
        p for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    return [str(p.relative_to(DATA_DIR)) for p in paths]


# ---------------------------------------------------------------------------
# Dropdown helpers
# ---------------------------------------------------------------------------

def library_choices(records: list[dict]) -> list[tuple[str, str]]:
    """Build ``(label, value)`` pairs for the workspace dropdown."""
    return [
        ("\u2014 Select a workspace \u2014", ""),
    ] + [
        (
            f"{r['original_name']}  \u00b7  {r['pipeline']}  \u00b7  {r['timestamp']}",
            r["id"],
        )
        for r in records
    ]
