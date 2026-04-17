"""JSONL append/read helpers with idempotent done-set indexing."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Iterable, Iterator

_LOCK = threading.Lock()


def jsonl_append(path: Path, record: dict) -> None:
    """Append one JSON record as a single line. Thread-safe within process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def jsonl_append_many(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                n += 1
            f.flush()
            os.fsync(f.fileno())
    return n


def jsonl_read(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln} not valid JSON: {e}") from e


def done_set(path: Path, key: str = "record_id") -> set[str]:
    """Build a set of completed record ids (skips records with `error` set)."""
    out: set[str] = set()
    for rec in jsonl_read(path):
        if rec.get("error"):
            continue
        rid = rec.get(key)
        if rid:
            out.add(rid)
    return out
