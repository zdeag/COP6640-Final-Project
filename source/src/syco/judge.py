"""Stage 2 — judge each response with a strong model."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import ValidationError
from tqdm.asyncio import tqdm as atqdm

from .config import Config
from .io_utils import done_set, jsonl_append, jsonl_read
from .logging_utils import get_logger, log_usage, utc_now_iso
from .openrouter import OpenRouterClient
from .schemas import (
    JudgeVerdict,
    JudgmentRecord,
    Usage,
    make_judgment_id,
)


class JudgeParseError(ValueError):
    """Raised when a judge response cannot be parsed into JudgeVerdict."""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_fences(s: str) -> str:
    return _FENCE_RE.sub("", s).strip()


def parse_judge_output(raw: str) -> JudgeVerdict:
    """Best-effort JSON extraction. Tolerates markdown fences, leading/trailing prose."""
    if not raw or not raw.strip():
        raise JudgeParseError("empty judge output")
    text = _strip_fences(raw)
    # Try the whole text first (clean case).
    candidates: list[str] = [text]
    m = _JSON_OBJ_RE.search(text)
    if m:
        candidates.append(m.group(0))
    last_err: Optional[Exception] = None
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError as e:
            last_err = e
            continue
        try:
            return JudgeVerdict.model_validate(obj)
        except ValidationError as e:
            last_err = e
            continue
    raise JudgeParseError(f"could not parse judge output: {last_err}") from last_err


def _render_judge_prompt(template: str, *, severity: int, domain: str, prompt: str, response: str) -> str:
    # Use literal replace rather than str.format because the template
    # contains an example JSON object with bare `{` and `}` characters that
    # would crash a format() call.
    return (
        template
        .replace("{severity}", str(severity))
        .replace("{domain}", domain)
        .replace("{prompt}", prompt)
        .replace("{response}", response)
    )


@dataclass
class JudgeReport:
    attempted: int
    succeeded: int
    parse_failures: int
    skipped: int
    total_cost_usd: float


async def _one(
    *,
    client: OpenRouterClient,
    cfg: Config,
    response_rec: dict,
    judge_template: str,
    run_id: str,
    out_path: Path,
    log,
) -> tuple[bool, float]:
    judge_model = cfg.judge.model
    judge_version = cfg.judge.prompt_version
    rid = make_judgment_id(response_rec["record_id"], judge_model, judge_version)

    rendered = _render_judge_prompt(
        judge_template,
        severity=int(response_rec["severity"]),
        domain=response_rec["domain"],
        prompt=response_rec["prompt"],
        response=response_rec["response"],
    )
    messages = [{"role": "system", "content": rendered}]

    raw_outputs: list[str] = []
    total_pt = 0
    total_ct = 0
    total_cost = 0.0
    verdict: Optional[JudgeVerdict] = None
    parse_err: Optional[str] = None

    attempts = cfg.judge.max_parse_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            result = await client.chat(
                model=judge_model,
                messages=messages,
                temperature=cfg.judge.params.temperature,
                top_p=cfg.judge.params.top_p,
                max_tokens=cfg.judge.params.max_tokens,
                seed=cfg.judge.params.seed,
            )
        except Exception as e:
            log.warning(
                "judge api error rid=%s prompt=%s err=%s",
                rid, response_rec.get("prompt_id"), e,
            )
            rec = JudgmentRecord(
                record_id=rid,
                response_record_id=response_rec["record_id"],
                prompt_id=response_rec["prompt_id"],
                model_label=response_rec["model_label"],
                severity=int(response_rec["severity"]),
                domain=response_rec["domain"],
                judge_model=judge_model,
                judge_prompt_version=judge_version,
                parse_attempts=attempt,
                raw_judge_output="\n---\n".join(raw_outputs),
                usage=Usage(
                    prompt_tokens=total_pt,
                    completion_tokens=total_ct,
                    cost_usd=total_cost or None,
                ),
                run_id=run_id,
                error=f"{type(e).__name__}: {e}",
            )
            jsonl_append(out_path, rec.model_dump())
            return False, total_cost

        raw_outputs.append(result.text)
        total_pt += result.prompt_tokens
        total_ct += result.completion_tokens
        total_cost += (result.cost_usd or 0.0)
        log_usage(
            cfg.paths.usage_jsonl,
            stage="judge",
            model_label="judge",
            model_id=judge_model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
            extra={"prompt_id": response_rec.get("prompt_id"), "attempt": attempt},
        )

        try:
            verdict = parse_judge_output(result.text)
            break
        except JudgeParseError as e:
            parse_err = str(e)
            if attempt < attempts:
                messages = [
                    {"role": "system", "content": rendered},
                    {"role": "assistant", "content": result.text},
                    {
                        "role": "user",
                        "content": (
                            "Your previous output was not valid JSON. "
                            "Return ONLY the JSON object with keys "
                            '"category", "appropriateness", "reasoning". '
                            "No prose, no markdown fences."
                        ),
                    },
                ]
                continue

    if verdict is None:
        log.warning(
            "judge parse failure rid=%s prompt=%s err=%s",
            rid, response_rec.get("prompt_id"), parse_err,
        )
        rec = JudgmentRecord(
            record_id=rid,
            response_record_id=response_rec["record_id"],
            prompt_id=response_rec["prompt_id"],
            model_label=response_rec["model_label"],
            severity=int(response_rec["severity"]),
            domain=response_rec["domain"],
            judge_model=judge_model,
            judge_prompt_version=judge_version,
            parse_attempts=attempts,
            raw_judge_output="\n---\n".join(raw_outputs),
            usage=Usage(
                prompt_tokens=total_pt,
                completion_tokens=total_ct,
                cost_usd=total_cost or None,
            ),
            run_id=run_id,
            error=f"parse: {parse_err}",
        )
        jsonl_append(out_path, rec.model_dump())
        return False, total_cost

    rec = JudgmentRecord(
        record_id=rid,
        response_record_id=response_rec["record_id"],
        prompt_id=response_rec["prompt_id"],
        model_label=response_rec["model_label"],
        severity=int(response_rec["severity"]),
        domain=response_rec["domain"],
        judge_model=judge_model,
        judge_prompt_version=judge_version,
        category=verdict.category,
        appropriateness=verdict.appropriateness,
        reasoning=verdict.reasoning,
        parse_attempts=len(raw_outputs),
        raw_judge_output="\n---\n".join(raw_outputs),
        usage=Usage(
            prompt_tokens=total_pt,
            completion_tokens=total_ct,
            cost_usd=total_cost or None,
        ),
        run_id=run_id,
    )
    jsonl_append(out_path, rec.model_dump())
    return True, total_cost


async def run_judge(
    cfg: Config,
    *,
    prompt_ids: Optional[list[str]] = None,
    model_labels: Optional[list[str]] = None,
    judge_model_override: Optional[str] = None,
    run_id: Optional[str] = None,
) -> JudgeReport:
    cfg.ensure_dirs()
    log = get_logger("syco.judge", cfg.paths.logs_dir / "judge.log")

    if judge_model_override:
        cfg.judge.model = judge_model_override
        log.info("judge model overridden to %s", judge_model_override)

    judge_template = cfg.paths.judge_prompt.read_text()

    in_path = cfg.paths.responses_jsonl
    out_path = cfg.paths.judgments_jsonl

    if not in_path.exists():
        raise FileNotFoundError(
            f"{in_path} not found. Run `syco generate` first."
        )

    response_recs: list[dict] = []
    for rec in jsonl_read(in_path):
        if rec.get("error"):
            continue
        if not rec.get("response"):
            continue
        if prompt_ids and rec.get("prompt_id") not in set(prompt_ids):
            continue
        if model_labels and rec.get("model_label") not in set(model_labels):
            continue
        response_recs.append(rec)

    done = done_set(out_path)
    pending = []
    skipped = 0
    for r in response_recs:
        rid = make_judgment_id(r["record_id"], cfg.judge.model, cfg.judge.prompt_version)
        if rid in done:
            skipped += 1
            continue
        pending.append(r)

    if not pending:
        log.info("nothing to judge (skipped=%d)", skipped)
        return JudgeReport(attempted=0, succeeded=0, parse_failures=0, skipped=skipped, total_cost_usd=0.0)

    run_id = run_id or utc_now_iso()
    log.info(
        "starting judge run_id=%s pending=%d skipped=%d judge=%s",
        run_id, len(pending), skipped, cfg.judge.model,
    )

    async with OpenRouterClient(
        api_key=cfg.api_key,
        app_name=cfg.openrouter.app_name,
        site_url=cfg.openrouter.site_url,
        max_concurrent=cfg.judge.max_concurrent,
        timeout_s=cfg.openrouter.request_timeout_s,
    ) as client:
        tasks = [
            _one(
                client=client,
                cfg=cfg,
                response_rec=r,
                judge_template=judge_template,
                run_id=run_id,
                out_path=out_path,
                log=log,
            )
            for r in pending
        ]
        results = await atqdm.gather(*tasks, desc="judge")

    succ = sum(1 for ok, _ in results if ok)
    fail = sum(1 for ok, _ in results if not ok)
    cost = sum(c for _, c in results)
    log.info("done attempted=%d succ=%d parse_fail=%d cost=$%.4f", len(pending), succ, fail, cost)
    return JudgeReport(
        attempted=len(pending),
        succeeded=succ,
        parse_failures=fail,
        skipped=skipped,
        total_cost_usd=cost,
    )


def run_judge_sync(cfg: Config, **kwargs) -> JudgeReport:
    return asyncio.run(run_judge(cfg, **kwargs))
