"""Structured logging + per-call cost ledger."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .io_utils import jsonl_append

_CONFIGURED: set[str] = set()


def get_logger(name: str, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    _CONFIGURED.add(name)
    return logger


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_usage(
    usage_path: Path,
    *,
    stage: str,
    model_label: str,
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: Optional[float],
    extra: Optional[dict] = None,
) -> None:
    rec = {
        "ts": utc_now_iso(),
        "stage": stage,
        "model_label": model_label,
        "model_id": model_id,
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "cost_usd": cost_usd,
    }
    if extra:
        rec.update(extra)
    jsonl_append(usage_path, rec)
