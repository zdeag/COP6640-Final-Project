"""Config loading: YAML + .env -> typed dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO_ROOT / "source"
DEFAULT_CONFIG_PATH = SOURCE_ROOT / "config.yaml"


@dataclass
class PathsCfg:
    prompts_csv: Path
    judge_prompt: Path
    data_dir: Path

    @property
    def responses_jsonl(self) -> Path:
        return self.data_dir / "responses" / "responses.jsonl"

    @property
    def judgments_jsonl(self) -> Path:
        return self.data_dir / "judgments" / "judgments.jsonl"

    @property
    def human_dir(self) -> Path:
        return self.data_dir / "human"

    @property
    def analysis_dir(self) -> Path:
        return self.data_dir / "analysis"

    @property
    def figures_dir(self) -> Path:
        return self.analysis_dir / "figures"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def usage_jsonl(self) -> Path:
        return self.logs_dir / "usage.jsonl"


@dataclass
class GenParamsCfg:
    temperature: float
    max_tokens: int
    top_p: float = 1.0
    seed: Optional[int] = None


@dataclass
class GenerateCfg:
    max_concurrent: int
    default_params: GenParamsCfg
    system_prompt: Optional[str]


@dataclass
class JudgeCfg:
    model: str
    prompt_version: str
    params: GenParamsCfg
    max_parse_retries: int
    max_concurrent: int


@dataclass
class ModelSpec:
    label: str
    openrouter_id: str


@dataclass
class AnalysisCfg:
    bootstrap_n: int
    ci: float
    rng_seed: int


@dataclass
class SamplingCfg:
    n_human_label: int
    rng_seed: int


@dataclass
class OpenRouterCfg:
    app_name: str
    site_url: str
    request_timeout_s: float


@dataclass
class Config:
    paths: PathsCfg
    generate: GenerateCfg
    judge: JudgeCfg
    models: list[ModelSpec]
    analysis: AnalysisCfg
    sampling: SamplingCfg
    openrouter: OpenRouterCfg
    api_key: str = field(repr=False)

    def model_by_label(self, label: str) -> ModelSpec:
        for m in self.models:
            if m.label == label:
                return m
        raise KeyError(f"Unknown model label: {label}")

    def ensure_dirs(self) -> None:
        for p in [
            self.paths.data_dir,
            self.paths.responses_jsonl.parent,
            self.paths.judgments_jsonl.parent,
            self.paths.human_dir,
            self.paths.analysis_dir,
            self.paths.figures_dir,
            self.paths.logs_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


def _resolve(base: Path, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path).resolve()
    base = path.parent
    raw = yaml.safe_load(path.read_text())

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Add it to .env at repo root."
        )

    paths = PathsCfg(
        prompts_csv=_resolve(base, raw["paths"]["prompts_csv"]),
        judge_prompt=_resolve(base, raw["paths"]["judge_prompt"]),
        data_dir=_resolve(base, raw["paths"]["data_dir"]),
    )

    gen_params = GenParamsCfg(**raw["generate"]["default_params"])
    judge_params = GenParamsCfg(**raw["judge"]["params"])

    return Config(
        paths=paths,
        generate=GenerateCfg(
            max_concurrent=int(raw["generate"]["max_concurrent"]),
            default_params=gen_params,
            system_prompt=raw["generate"].get("system_prompt"),
        ),
        judge=JudgeCfg(
            model=raw["judge"]["model"],
            prompt_version=raw["judge"]["prompt_version"],
            params=judge_params,
            max_parse_retries=int(raw["judge"].get("max_parse_retries", 1)),
            max_concurrent=int(raw["judge"].get("max_concurrent", 6)),
        ),
        models=[ModelSpec(**m) for m in raw["models"]],
        analysis=AnalysisCfg(**raw["analysis"]),
        sampling=SamplingCfg(**raw["sampling"]),
        openrouter=OpenRouterCfg(**raw["openrouter"]),
        api_key=api_key,
    )
