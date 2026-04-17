"""Stage 3b — load human labels and join back to judgments for κ."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config
from .io_utils import jsonl_read


def load_labels(cfg: Config, labeled_csv: Path) -> pd.DataFrame:
    """Read the hand-labeled CSV, validate, and join to judgments + responses."""
    labels = pd.read_csv(labeled_csv)
    required = {"response_record_id", "human_category", "human_appropriateness"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"{labeled_csv} missing columns: {missing}")

    labels["human_category"] = labels["human_category"].astype(str).str.strip().str.upper()
    bad_cat = ~labels["human_category"].isin(list("ABCDEF"))
    if bad_cat.any():
        rows = labels.loc[bad_cat, "response_record_id"].tolist()
        raise ValueError(f"unlabeled or invalid human_category for: {rows}")

    labels["human_appropriateness"] = pd.to_numeric(labels["human_appropriateness"], errors="coerce")
    if labels["human_appropriateness"].isna().any() or not labels["human_appropriateness"].between(1, 5).all():
        rows = labels.loc[
            labels["human_appropriateness"].isna() | ~labels["human_appropriateness"].between(1, 5),
            "response_record_id",
        ].tolist()
        raise ValueError(f"invalid human_appropriateness for: {rows}")
    labels["human_appropriateness"] = labels["human_appropriateness"].astype(int)

    judgs = pd.DataFrame(
        r
        for r in jsonl_read(cfg.paths.judgments_jsonl)
        if not r.get("error") and r.get("category")
    )
    judgs = judgs.rename(
        columns={"category": "judge_category", "appropriateness": "judge_appropriateness"}
    )
    keep = [
        "response_record_id",
        "model_label",
        "judge_category",
        "judge_appropriateness",
        "reasoning",
        "judge_model",
        "judge_prompt_version",
    ]
    judgs = judgs[[c for c in keep if c in judgs.columns]]

    merged = labels.merge(judgs, on="response_record_id", how="left")
    if merged["judge_category"].isna().any():
        rows = merged.loc[merged["judge_category"].isna(), "response_record_id"].tolist()
        raise ValueError(f"could not find judgments for: {rows}")
    return merged
