"""Stage 3a — stratified sampling for human validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config
from .io_utils import jsonl_read
from .schemas import CATEGORIES, SEVERITIES

HUMAN_LABEL_COLS = (
    "response_record_id",
    "severity",
    "domain",
    "prompt",
    "response",
    "human_category",
    "human_appropriateness",
    "human_notes",
    "annotator_id",
)


def load_joined(cfg: Config) -> pd.DataFrame:
    """Inner-join responses and judgments on record_id; drop errored rows."""
    resps = pd.DataFrame(
        r for r in jsonl_read(cfg.paths.responses_jsonl) if not r.get("error")
    )
    judgs = pd.DataFrame(
        r
        for r in jsonl_read(cfg.paths.judgments_jsonl)
        if not r.get("error") and r.get("category")
    )
    if resps.empty or judgs.empty:
        raise RuntimeError(
            f"No data to sample from. resps={len(resps)} judgs={len(judgs)}"
        )
    df = judgs.merge(
        resps[["record_id", "prompt", "response"]],
        left_on="response_record_id",
        right_on="record_id",
        suffixes=("_j", "_r"),
        how="inner",
    )
    df["severity"] = df["severity"].astype(int)
    return df


@dataclass
class SampleReport:
    n_total: int
    n_returned: int
    cells_filled: int
    cells_short: int
    out_path: Path


def draw_stratified(
    df: pd.DataFrame,
    *,
    n: int = 60,
    rng_seed: int = 7,
    balance_col: str = "model_label",
) -> pd.DataFrame:
    """Stratify on (severity, category); secondary balance on `balance_col`."""
    rng = np.random.default_rng(rng_seed)
    cells: list[tuple[int, str]] = [(s, c) for s in SEVERITIES for c in CATEGORIES]
    base = max(n // len(cells), 1)
    quota: dict[tuple[int, str], int] = {cell: base for cell in cells}
    remainder = n - base * len(cells)

    # Hand out remainder to the largest cells first.
    counts = {
        cell: int(((df["severity"] == cell[0]) & (df["category"] == cell[1])).sum())
        for cell in cells
    }
    order = sorted(cells, key=lambda c: -counts[c])
    for cell in order[:remainder]:
        quota[cell] += 1

    picks: list[pd.DataFrame] = []
    short = 0
    for cell, q in quota.items():
        sub = df[(df["severity"] == cell[0]) & (df["category"] == cell[1])]
        if len(sub) == 0:
            short += 1
            continue
        take = min(q, len(sub))
        if take < q:
            short += 1
        sample = sub.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))
        picks.append(sample)

    drawn = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()

    # Secondary balance: if any model is over-represented by > 20%, swap with under-represented within same stratum.
    if not drawn.empty and balance_col in drawn.columns:
        target = max(len(drawn) // drawn[balance_col].nunique(), 1)
        for _ in range(20):  # bounded swap rounds
            counts_m = drawn[balance_col].value_counts()
            over = counts_m[counts_m > target * 1.2].index.tolist()
            under = counts_m[counts_m < target * 0.8].index.tolist()
            if not over or not under:
                break
            swapped_any = False
            for o in over:
                for u in under:
                    drawn_o = drawn[drawn[balance_col] == o]
                    if drawn_o.empty:
                        continue
                    cell = (int(drawn_o.iloc[0]["severity"]), drawn_o.iloc[0]["category"])
                    candidates = df[
                        (df["severity"] == cell[0])
                        & (df["category"] == cell[1])
                        & (df[balance_col] == u)
                        & (~df["response_record_id"].isin(drawn["response_record_id"]))
                    ]
                    if candidates.empty:
                        continue
                    swap_in = candidates.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1)))
                    swap_out_idx = drawn_o.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).index
                    drawn = drawn.drop(swap_out_idx)
                    drawn = pd.concat([drawn, swap_in], ignore_index=True)
                    swapped_any = True
                    break
                if swapped_any:
                    break
            if not swapped_any:
                break

    # Final shuffle to anonymize neighbor order.
    drawn = drawn.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
    return drawn


def write_blinded_csv(drawn: pd.DataFrame, out_path: Path) -> None:
    out = pd.DataFrame()
    out["response_record_id"] = drawn["response_record_id"]
    out["severity"] = drawn["severity"]
    out["domain"] = drawn["domain"]
    out["prompt"] = drawn["prompt"]
    out["response"] = drawn["response"]
    out["human_category"] = ""
    out["human_appropriateness"] = ""
    out["human_notes"] = ""
    out["annotator_id"] = ""
    out = out[list(HUMAN_LABEL_COLS)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def run_sample(
    cfg: Config,
    *,
    n: Optional[int] = None,
    out_path: Optional[Path] = None,
) -> SampleReport:
    cfg.ensure_dirs()
    n = n or cfg.sampling.n_human_label
    df = load_joined(cfg)
    drawn = draw_stratified(df, n=n, rng_seed=cfg.sampling.rng_seed)
    out_path = out_path or (cfg.paths.human_dir / "sample_60.csv")
    write_blinded_csv(drawn, out_path)
    cells_total = len(SEVERITIES) * len(CATEGORIES)
    cells_seen = drawn.groupby(["severity", "category"]).ngroups
    return SampleReport(
        n_total=n,
        n_returned=len(drawn),
        cells_filled=cells_seen,
        cells_short=cells_total - cells_seen,
        out_path=out_path,
    )
