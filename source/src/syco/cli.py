"""Unified CLI: `python -m syco.cli <subcommand>`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyze import run_analyze
from .config import DEFAULT_CONFIG_PATH, load_config
from .generate import run_generate_sync
from .judge import run_judge_sync
from .kappa import run_kappa
from .sample import run_sample


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml"
    )


def _cmd_generate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    report = run_generate_sync(
        cfg,
        model_labels=args.models,
        prompt_ids=args.prompt_ids,
        retry_errors=args.retry_errors,
    )
    print(
        f"[generate] attempted={report.attempted} succ={report.succeeded} "
        f"fail={report.failed} skipped={report.skipped} cost=${report.total_cost_usd:.4f}"
    )
    return 0 if report.failed == 0 else 1


def _cmd_judge(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    report = run_judge_sync(
        cfg,
        prompt_ids=args.prompt_ids,
        model_labels=args.models,
        judge_model_override=args.judge_model,
    )
    print(
        f"[judge] attempted={report.attempted} succ={report.succeeded} "
        f"parse_fail={report.parse_failures} skipped={report.skipped} "
        f"cost=${report.total_cost_usd:.4f}"
    )
    return 0 if report.parse_failures == 0 else 1


def _cmd_sample(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    rep = run_sample(cfg, n=args.n, out_path=args.out)
    print(
        f"[sample] wrote {rep.n_returned}/{rep.n_total} rows -> {rep.out_path} "
        f"(cells_filled={rep.cells_filled} cells_short={rep.cells_short})"
    )
    return 0


def _cmd_kappa(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    rep = run_kappa(cfg, labeled_csv=args.labels, out_md=args.out)
    lo, hi = rep.kappa_category_ci
    verdict = "PASS" if rep.passes else "FAIL"
    print(
        f"[kappa] n={rep.n} κ_cat={rep.kappa_category:+.3f} "
        f"[CI {lo:+.3f},{hi:+.3f}] κ_appr_q={rep.kappa_appropriateness_quadratic:+.3f} "
        f"-> {verdict}"
    )
    return 0 if rep.passes else 2


def _cmd_analyze(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    rep = run_analyze(cfg)
    print(
        f"[analyze] n={rep.n_judgments} -> {rep.summary_csv}, {rep.summary_md}, "
        f"{len(rep.figures)} figures in {rep.figures[0].parent}"
    )
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    """Run generate -> judge -> analyze. Stops before sample/kappa (need human)."""
    rc = _cmd_generate(args)
    if rc != 0:
        return rc
    rc = _cmd_judge(args)
    if rc != 0:
        return rc
    return _cmd_analyze(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="syco", description="Epistemic sycophancy benchmark.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate", help="Stage 1 — generate model responses")
    _add_common(p)
    p.add_argument("--models", nargs="+", help="Subset of model labels (default: all)")
    p.add_argument("--prompt-ids", nargs="+", help="Subset of prompt ids")
    p.add_argument("--retry-errors", action="store_true", help="Re-attempt prompts that previously errored")
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("judge", help="Stage 2 — judge each response")
    _add_common(p)
    p.add_argument("--prompt-ids", nargs="+")
    p.add_argument("--models", nargs="+")
    p.add_argument("--judge-model", help="Override judge model id")
    p.set_defaults(func=_cmd_judge)

    p = sub.add_parser("sample", help="Stage 3a — stratified sample for human labeling")
    _add_common(p)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=_cmd_sample)

    p = sub.add_parser("kappa", help="Stage 3c — Cohen's κ vs. human labels")
    _add_common(p)
    p.add_argument("--labels", type=Path, default=None, help="Path to sample_60_labeled.csv")
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=_cmd_kappa)

    p = sub.add_parser("analyze", help="Stage 4 — tables and figures")
    _add_common(p)
    p.set_defaults(func=_cmd_analyze)

    p = sub.add_parser("all", help="generate -> judge -> analyze")
    _add_common(p)
    p.add_argument("--models", nargs="+")
    p.add_argument("--prompt-ids", nargs="+")
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--judge-model", default=None)
    p.set_defaults(func=_cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
