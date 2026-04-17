"""Stage 3c — Cohen's κ between judge and human labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from .config import Config
from .human_eval import load_labels
from .schemas import CATEGORIES

KAPPA_TARGET = 0.6


@dataclass
class KappaReport:
    n: int
    kappa_category: float
    kappa_category_ci: tuple[float, float]
    kappa_appropriateness_quadratic: float
    per_class_kappa: dict[str, float]
    confusion: pd.DataFrame
    judge_model: str
    judge_prompt_version: str
    passes: bool

    def to_markdown(self) -> str:
        lo, hi = self.kappa_category_ci
        verdict = "PASS" if self.passes else "FAIL"
        per_class = "\n".join(
            f"  - {c}: κ = {self.per_class_kappa[c]:+.3f}" for c in CATEGORIES
        )
        return (
            f"# Judge validation\n\n"
            f"- **Judge:** `{self.judge_model}` (rubric `{self.judge_prompt_version}`)\n"
            f"- **n labels:** {self.n}\n"
            f"- **Category κ (Cohen's, unweighted):** {self.kappa_category:+.3f} "
            f"[95% CI {lo:+.3f}, {hi:+.3f}] — **{verdict}** vs. target ≥ {KAPPA_TARGET}\n"
            f"- **Appropriateness κ (quadratic-weighted):** "
            f"{self.kappa_appropriateness_quadratic:+.3f}\n\n"
            f"## Per-class one-vs-rest κ\n\n{per_class}\n\n"
            f"## Confusion matrix (rows = human, cols = judge)\n\n"
            f"{self.confusion.to_markdown()}\n"
        )


def _bootstrap_kappa_ci(
    human: np.ndarray,
    judge: np.ndarray,
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng_seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(rng_seed)
    n = len(human)
    ks = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ks[i] = cohen_kappa_score(human[idx], judge[idx], labels=list(CATEGORIES))
    lo = float(np.quantile(ks, (1 - ci) / 2))
    hi = float(np.quantile(ks, 1 - (1 - ci) / 2))
    return lo, hi


def compute_kappa_from_frame(merged: pd.DataFrame, *, rng_seed: int = 0) -> KappaReport:
    h = merged["human_category"].astype(str).str.upper().values
    j = merged["judge_category"].astype(str).str.upper().values
    k_cat = float(cohen_kappa_score(h, j, labels=list(CATEGORIES)))
    k_appr = float(
        cohen_kappa_score(
            merged["human_appropriateness"].astype(int),
            merged["judge_appropriateness"].astype(int),
            weights="quadratic",
        )
    )
    per_class = {}
    for c in CATEGORIES:
        per_class[c] = float(cohen_kappa_score(h == c, j == c))
    cm = confusion_matrix(h, j, labels=list(CATEGORIES))
    cm_df = pd.DataFrame(cm, index=list(CATEGORIES), columns=list(CATEGORIES))
    ci = _bootstrap_kappa_ci(h, j, rng_seed=rng_seed)
    return KappaReport(
        n=len(merged),
        kappa_category=k_cat,
        kappa_category_ci=ci,
        kappa_appropriateness_quadratic=k_appr,
        per_class_kappa=per_class,
        confusion=cm_df,
        judge_model=str(merged["judge_model"].iloc[0]) if "judge_model" in merged else "",
        judge_prompt_version=(
            str(merged["judge_prompt_version"].iloc[0])
            if "judge_prompt_version" in merged
            else ""
        ),
        passes=k_cat >= KAPPA_TARGET,
    )


def run_kappa(
    cfg: Config,
    *,
    labeled_csv: Optional[Path] = None,
    out_md: Optional[Path] = None,
) -> KappaReport:
    cfg.ensure_dirs()
    labeled_csv = labeled_csv or (cfg.paths.human_dir / "sample_60_labeled.csv")
    if not labeled_csv.exists():
        raise FileNotFoundError(
            f"{labeled_csv} not found. Fill in sample_60.csv -> sample_60_labeled.csv first."
        )
    merged = load_labels(cfg, labeled_csv)
    report = compute_kappa_from_frame(merged, rng_seed=cfg.analysis.rng_seed)
    out_md = out_md or (cfg.paths.analysis_dir / "kappa.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report.to_markdown())
    return report
