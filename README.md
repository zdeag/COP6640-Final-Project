# COP6640 — Epistemic Sycophancy Benchmark (`syco`)

Measures how language models respond to user prompts expressing irrational, grandiose, or potentially delusional beliefs. The pipeline is **generate → judge → validate-judge → analyze**: five models each answer 150 prompts, an LLM judge labels every response, a stratified human sample validates the judge via Cohen's κ, and the analyzer produces tables and figures.

## Repository layout

```
.
├── prompts/
│   └── prompts.csv         # 150 benchmark prompts (input to stage 1)
├── source/                 # Python package, CLI, tests, config
│   ├── src/syco/           # package modules
│   ├── judge/              # judge system prompt template
│   ├── data/               # gitignored outputs (responses, judgments, analysis, logs)
│   ├── tests/              # pytest suite (20 tests)
│   ├── config.yaml         # model registry + generation/judge/analysis params
│   ├── pyproject.toml
│   └── README.md           # full setup + command reference
├── .env                    # OPENROUTER_API_KEY (repo root, gitignored)
└── README.md               # this file
```

## Quick start

Requires **Python 3.11**. Put your OpenRouter key in `.env` at the repo root:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

Then from `source/`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/                    # sanity check (20/20)
python -m syco.cli all           # generate → judge → analyze
```

## Pipeline stages

| Stage | Command (run from `source/`) | Output |
|---|---|---|
| 1. Generate | `python -m syco.cli generate` | `data/responses/responses.jsonl` (750 responses) |
| 2. Judge | `python -m syco.cli judge` | `data/judgments/judgments.jsonl` |
| 3a. Sample | `python -m syco.cli sample --n 60` | `data/human/sample_60.csv` (blinded) |
| 3b. Label | *(fill in `human_category` + `human_appropriateness`, save as `sample_60_labeled.csv`)* | – |
| 3c. Kappa | `python -m syco.cli kappa` | `data/analysis/kappa.md` (target κ ≥ 0.6) |
| 4. Analyze | `python -m syco.cli analyze` | tables + 3 figures in `data/analysis/` |

Every stage is idempotent — re-running resumes from the JSONL done-set.

## Configuration

Models, generation params, and judge model live in [source/config.yaml](source/config.yaml). Defaults:

- **5 models**: llama-3.1-8b, mistral-7b, qwen-2.5-7b, gpt-4o-mini, claude-3.5-haiku
- **Judge**: `anthropic/claude-3.7-sonnet` at `temperature=0.0`
- **Generation**: `temperature=0.7`, `max_tokens=800`, `seed=42`

See [source/README.md](source/README.md) for flags (`--models`, `--prompt-ids`, `--retry-errors`, `--judge-model`) and smoke-test recipes.
