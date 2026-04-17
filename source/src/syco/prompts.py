"""Load and validate prompts.csv."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .schemas import DOMAINS, SEVERITIES

REQUIRED_COLS = ("id", "severity", "domain", "source", "prompt")


@dataclass(frozen=True)
class PromptRow:
    id: str
    severity: int
    domain: str
    source: str
    prompt: str


def load_prompts(path: Path) -> list[PromptRow]:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    rows: list[PromptRow] = []
    for _, r in df.iterrows():
        sev = int(r["severity"])
        if sev not in SEVERITIES:
            raise ValueError(f"{r['id']}: severity {sev} not in {SEVERITIES}")
        domain = str(r["domain"]).strip()
        if domain not in DOMAINS:
            raise ValueError(f"{r['id']}: domain '{domain}' not in {DOMAINS}")
        rows.append(
            PromptRow(
                id=str(r["id"]).strip(),
                severity=sev,
                domain=domain,
                source=str(r["source"]).strip(),
                prompt=str(r["prompt"]),
            )
        )
    if len({p.id for p in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate prompt ids")
    return rows


def load_prompts_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df
