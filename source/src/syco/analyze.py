"""Stage 4 — distributions, bootstrap CIs, tables and plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .io_utils import jsonl_read
from .schemas import CATEGORIES, INAPPROPRIATE_AT, SEVERITIES


@dataclass
class AnalyzeReport:
    n_judgments: int
    summary_csv: Path
    summary_md: Path
    figures: list[Path]


def load_judgments_df(cfg: Config) -> pd.DataFrame:
    rows = [
        r
        for r in jsonl_read(cfg.paths.judgments_jsonl)
        if not r.get("error") and r.get("category")
    ]
    if not rows:
        raise RuntimeError(f"No clean judgments at {cfg.paths.judgments_jsonl}")
    df = pd.DataFrame(rows)
    df["severity"] = df["severity"].astype(int)
    df["appropriateness"] = df["appropriateness"].astype(int)
    df["category"] = df["category"].astype(str).str.upper()

    # Pull `source` (seed/generated) from responses for subgroup analysis.
    resps = pd.DataFrame(jsonl_read(cfg.paths.responses_jsonl))
    if not resps.empty and "source" in resps.columns:
        df = df.merge(
            resps[["record_id", "source"]],
            left_on="response_record_id",
            right_on="record_id",
            how="left",
            suffixes=("", "_r"),
        )
    return df


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int, ci: float, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        v = float(values[0])
        return (v, v)
    means = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(values[idx]))
    lo = float(np.quantile(means, (1 - ci) / 2))
    hi = float(np.quantile(means, 1 - (1 - ci) / 2))
    return lo, hi


def category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Rows: model, cols: A..F percentages."""
    ct = pd.crosstab(df["model_label"], df["category"])
    for c in CATEGORIES:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[list(CATEGORIES)]
    pct = ct.div(ct.sum(axis=1), axis=0) * 100.0
    return pct


def mean_appropriateness_by_cell(
    df: pd.DataFrame,
    *,
    bootstrap_n: int,
    ci: float,
    rng_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    rows = []
    for (model, sev), grp in df.groupby(["model_label", "severity"]):
        vals = grp["appropriateness"].astype(float).to_numpy()
        mean = float(np.mean(vals)) if len(vals) else float("nan")
        lo, hi = _bootstrap_mean_ci(vals, bootstrap_n, ci, rng)
        rows.append({
            "model_label": model,
            "severity": int(sev),
            "n": int(len(vals)),
            "mean_appr": mean,
            "ci_lo": lo,
            "ci_hi": hi,
        })
    return pd.DataFrame(rows).sort_values(["model_label", "severity"]).reset_index(drop=True)


def harmful_rate_by_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of responses whose category is in INAPPROPRIATE_AT[severity]."""
    rows = []
    for (model, sev), grp in df.groupby(["model_label", "severity"]):
        bad = INAPPROPRIATE_AT.get(int(sev), set())
        n = len(grp)
        n_bad = int(grp["category"].isin(bad).sum())
        rows.append({
            "model_label": model,
            "severity": int(sev),
            "n": n,
            "n_inappropriate": n_bad,
            "inappropriate_rate": (n_bad / n) if n else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["model_label", "severity"]).reset_index(drop=True)


def _plot_category_distribution(pct: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottoms = np.zeros(len(pct))
    cmap = plt.get_cmap("RdYlGn_r")  # A=red, F=red — but ramp gives a clean axis
    colors = [cmap(i / 5) for i in range(6)]
    for c, color in zip(CATEGORIES, colors):
        ax.bar(pct.index, pct[c], bottom=bottoms, label=c, color=color, edgecolor="white")
        bottoms = bottoms + pct[c].values
    ax.set_ylabel("Response category share (%)")
    ax.set_xlabel("Model")
    ax.set_title("Response category distribution by model")
    ax.set_ylim(0, 100)
    ax.legend(title="Category", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_appropriateness_by_severity(appr: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, grp in appr.groupby("model_label"):
        grp = grp.sort_values("severity")
        ax.plot(grp["severity"], grp["mean_appr"], marker="o", label=model)
        ax.fill_between(grp["severity"], grp["ci_lo"], grp["ci_hi"], alpha=0.15)
    ax.set_xticks(list(SEVERITIES))
    ax.set_xlabel("Severity")
    ax.set_ylabel("Mean appropriateness (1–5)")
    ax.set_ylim(1, 5)
    ax.set_title("Appropriateness by severity (mean ± 95% CI)")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_harmful_rate(harm: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, grp in harm.groupby("model_label"):
        grp = grp.sort_values("severity")
        ax.plot(grp["severity"], grp["inappropriate_rate"], marker="o", label=model)
    ax.set_xticks(list(SEVERITIES))
    ax.set_xlabel("Severity")
    ax.set_ylabel("Inappropriate rate")
    ax.set_ylim(0, 1)
    ax.set_title("Severity-Conditioned Harm Rate")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _build_summary(
    pct: pd.DataFrame, appr: pd.DataFrame, harm: pd.DataFrame
) -> pd.DataFrame:
    """Long-format summary keyed by (model, severity)."""
    base = appr.merge(harm, on=["model_label", "severity", "n"], how="outer")
    pct_long = pct.reset_index().rename(columns={"index": "model_label"})
    pct_long.columns.name = None
    pct_long = pct_long.rename(columns={c: f"category_{c}_pct" for c in CATEGORIES})
    return base.merge(pct_long, on="model_label", how="left")


def run_analyze(cfg: Config) -> AnalyzeReport:
    cfg.ensure_dirs()
    df = load_judgments_df(cfg)
    pct = category_distribution(df)
    appr = mean_appropriateness_by_cell(
        df,
        bootstrap_n=cfg.analysis.bootstrap_n,
        ci=cfg.analysis.ci,
        rng_seed=cfg.analysis.rng_seed,
    )
    harm = harmful_rate_by_cell(df)
    summary = _build_summary(pct, appr, harm)

    out_csv = cfg.paths.analysis_dir / "summary.csv"
    summary.to_csv(out_csv, index=False)

    md_lines: list[str] = []
    md_lines.append("# Sycophancy benchmark — results\n")
    md_lines.append(f"_n judgments: {len(df)}_  _judge: {df['judge_model'].iloc[0]}_  _rubric: {df['judge_prompt_version'].iloc[0]}_\n")
    md_lines.append("## Response category share by model (%)\n")
    md_lines.append(pct.round(1).to_markdown())
    md_lines.append("\n\n## Mean appropriateness by (model, severity) with bootstrap 95% CI\n")
    md_lines.append(appr.round(3).to_markdown(index=False))
    md_lines.append("\n\n## Severity-Conditioned Harm Rate\n")
    md_lines.append(harm.round(3).to_markdown(index=False))
    if "source" in df.columns and df["source"].notna().any():
        md_lines.append("\n\n## Subgroup: by `source` (seed vs generated)\n")
        sub = df.groupby(["source", "model_label"]).agg(
            n=("appropriateness", "size"),
            mean_appr=("appropriateness", "mean"),
        ).reset_index()
        md_lines.append(sub.round(3).to_markdown(index=False))
    out_md = cfg.paths.analysis_dir / "summary.md"
    out_md.write_text("\n".join(md_lines) + "\n")

    figs: list[Path] = []
    fpath = cfg.paths.figures_dir / "category_dist.png"
    _plot_category_distribution(pct, fpath)
    figs.append(fpath)
    fpath = cfg.paths.figures_dir / "appropriateness_by_severity.png"
    _plot_appropriateness_by_severity(appr, fpath)
    figs.append(fpath)
    fpath = cfg.paths.figures_dir / "inappropriate_rate_by_severity.png"
    _plot_harmful_rate(harm, fpath)
    figs.append(fpath)

    return AnalyzeReport(
        n_judgments=len(df), summary_csv=out_csv, summary_md=out_md, figures=figs
    )
