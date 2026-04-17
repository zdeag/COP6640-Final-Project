# syco — Epistemic Sycophancy Benchmark

Measures how language models respond to user prompts expressing irrational, grandiose, or potentially delusional beliefs. Pipeline: **generate -> judge -> validate-judge -> analyze**.

## 1. Install

Requires **Python 3.11**. From the `source/` directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the `syco` package and the `pytest`/`ruff` dev tools. Run `pytest tests/` to verify (should be 20/20 passing).

## 2. Environment

The OpenRouter API key lives in `.env` at the **repo root** (one level above `source/`):

```
OPENROUTER_API_KEY=sk-or-v1-...
```

Settings (models, generation params, judge model, paths) are in `source/config.yaml`. Defaults already match the locked decisions: 5 models, judge = `anthropic/claude-3.7-sonnet`, `temperature=0.7` for generation, `temperature=0.0` for judge.

## 3. Commands

All commands run from `source/` with the venv activated. Every stage is **idempotent** — re-running picks up where it left off via the JSONL done-set.

| Stage | Command | What it does | Approx. cost |
|---|---|---|---|
| 1. Generate | `python -m syco.cli generate` | 150 prompts × 5 models = 750 responses → `data/responses/responses.jsonl` | < $0.10 |
| 2. Judge | `python -m syco.cli judge` | Claude 3.7 Sonnet judges every response → `data/judgments/judgments.jsonl` | ~$2–3 |
| 3a. Sample | `python -m syco.cli sample --n 60` | Stratified blinded CSV → `data/human/sample_60.csv` | free |
| 3b. *Hand-label* | (open the CSV, fill `human_category` + `human_appropriateness`, save as `sample_60_labeled.csv`) | – | – |
| 3c. Kappa | `python -m syco.cli kappa` | Cohen's κ vs. judge → `data/analysis/kappa.md` (target ≥ 0.6) | free |
| 4. Analyze | `python -m syco.cli analyze` | Tables + 3 figures → `data/analysis/` | free |

Convenience: `python -m syco.cli all` runs **generate → judge → analyze** (skips the human steps).

### Useful flags

- `--models llama31_8b mistral_7b` — restrict to a subset of models
- `--prompt-ids p001 p002 p003` — restrict to a subset of prompts (great for smoke tests)
- `--retry-errors` (generate only) — retry prompts that previously failed
- `--judge-model openai/gpt-4o` (judge only) — swap the judge model

### Smoke test in 30 seconds

```bash
python -m syco.cli generate --models llama31_8b --prompt-ids p001 p005 p010
python -m syco.cli judge --prompt-ids p001 p005 p010
python -m syco.cli analyze
```

## Layout

```
source/
  src/syco/      package modules
  judge/         judge system prompt template
  data/          gitignored outputs (responses, judgments, analysis, logs)
  tests/         pytest suite
  config.yaml    model registry + params
```
