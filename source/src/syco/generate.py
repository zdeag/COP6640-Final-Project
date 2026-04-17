"""Stage 1 — generate model responses to all prompts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm.asyncio import tqdm as atqdm

from .config import Config, ModelSpec
from .io_utils import done_set, jsonl_append, jsonl_read
from .logging_utils import get_logger, log_usage, utc_now_iso
from .openrouter import OpenRouterClient
from .prompts import PromptRow, load_prompts
from .schemas import GenParams, ResponseRecord, Usage, make_response_id


@dataclass
class GenerateReport:
    attempted: int
    succeeded: int
    failed: int
    skipped: int
    total_cost_usd: float


def _build_messages(prompt: str, system_prompt: Optional[str]) -> list[dict]:
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})
    return msgs


async def _one(
    *,
    client: OpenRouterClient,
    cfg: Config,
    prompt_row: PromptRow,
    model: ModelSpec,
    run_id: str,
    out_path: Path,
    log,
) -> tuple[bool, float]:
    record_id = make_response_id(prompt_row.id, model.label)
    params = cfg.generate.default_params
    messages = _build_messages(prompt_row.prompt, cfg.generate.system_prompt)
    try:
        result = await client.chat(
            model=model.openrouter_id,
            messages=messages,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            seed=params.seed,
        )
    except Exception as e:  # final failure after retries
        log.warning("generate failed prompt=%s model=%s err=%s", prompt_row.id, model.label, e)
        rec = ResponseRecord(
            record_id=record_id,
            prompt_id=prompt_row.id,
            severity=prompt_row.severity,
            domain=prompt_row.domain,
            source=prompt_row.source,
            prompt=prompt_row.prompt,
            model_label=model.label,
            model_id=model.openrouter_id,
            model_reported=None,
            system_prompt=cfg.generate.system_prompt,
            params=GenParams(
                temperature=params.temperature,
                top_p=params.top_p,
                max_tokens=params.max_tokens,
                seed=params.seed,
            ),
            response="",
            finish_reason=None,
            usage=Usage(),
            run_id=run_id,
            error=f"{type(e).__name__}: {e}",
        )
        jsonl_append(out_path, rec.model_dump())
        return False, 0.0

    rec = ResponseRecord(
        record_id=record_id,
        prompt_id=prompt_row.id,
        severity=prompt_row.severity,
        domain=prompt_row.domain,
        source=prompt_row.source,
        prompt=prompt_row.prompt,
        model_label=model.label,
        model_id=model.openrouter_id,
        model_reported=result.model_reported,
        system_prompt=cfg.generate.system_prompt,
        params=GenParams(
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            seed=params.seed,
        ),
        response=result.text,
        finish_reason=result.finish_reason,
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost_usd=result.cost_usd,
        ),
        run_id=run_id,
    )
    jsonl_append(out_path, rec.model_dump())
    log_usage(
        cfg.paths.usage_jsonl,
        stage="generate",
        model_label=model.label,
        model_id=model.openrouter_id,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=result.cost_usd,
        extra={"prompt_id": prompt_row.id},
    )
    return True, (result.cost_usd or 0.0)


async def run_generate(
    cfg: Config,
    *,
    model_labels: Optional[list[str]] = None,
    prompt_ids: Optional[list[str]] = None,
    retry_errors: bool = False,
    run_id: Optional[str] = None,
) -> GenerateReport:
    cfg.ensure_dirs()
    log = get_logger("syco.generate", cfg.paths.logs_dir / "generate.log")
    out_path = cfg.paths.responses_jsonl

    prompts = load_prompts(cfg.paths.prompts_csv)
    if prompt_ids:
        wanted = set(prompt_ids)
        prompts = [p for p in prompts if p.id in wanted]
        if not prompts:
            raise ValueError(f"No prompts matched ids {prompt_ids}")

    models = cfg.models
    if model_labels:
        wanted_m = set(model_labels)
        models = [m for m in cfg.models if m.label in wanted_m]
        if not models:
            raise ValueError(f"No models matched labels {model_labels}")

    done = done_set(out_path)
    error_ids: set[str] = set()
    if retry_errors:
        for rec in jsonl_read(out_path):
            if rec.get("error"):
                error_ids.add(rec.get("record_id", ""))

    pending: list[tuple[PromptRow, ModelSpec]] = []
    skipped = 0
    for p in prompts:
        for m in models:
            rid = make_response_id(p.id, m.label)
            if rid in done and rid not in error_ids:
                skipped += 1
                continue
            pending.append((p, m))

    if not pending:
        log.info("nothing to do (skipped=%d)", skipped)
        return GenerateReport(attempted=0, succeeded=0, failed=0, skipped=skipped, total_cost_usd=0.0)

    run_id = run_id or utc_now_iso()
    log.info(
        "starting generate run_id=%s pending=%d skipped=%d models=%s",
        run_id, len(pending), skipped, [m.label for m in models],
    )

    async with OpenRouterClient(
        api_key=cfg.api_key,
        app_name=cfg.openrouter.app_name,
        site_url=cfg.openrouter.site_url,
        max_concurrent=cfg.generate.max_concurrent,
        timeout_s=cfg.openrouter.request_timeout_s,
    ) as client:
        tasks = [
            _one(client=client, cfg=cfg, prompt_row=p, model=m, run_id=run_id, out_path=out_path, log=log)
            for (p, m) in pending
        ]
        results = await atqdm.gather(*tasks, desc="generate")

    succ = sum(1 for ok, _ in results if ok)
    fail = sum(1 for ok, _ in results if not ok)
    cost = sum(c for _, c in results)
    log.info("done attempted=%d succ=%d fail=%d cost=$%.4f", len(pending), succ, fail, cost)
    return GenerateReport(
        attempted=len(pending),
        succeeded=succ,
        failed=fail,
        skipped=skipped,
        total_cost_usd=cost,
    )


def run_generate_sync(cfg: Config, **kwargs) -> GenerateReport:
    return asyncio.run(run_generate(cfg, **kwargs))
