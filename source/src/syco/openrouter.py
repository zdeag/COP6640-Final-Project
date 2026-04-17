"""Async OpenRouter client built on the OpenAI SDK."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class ChatResult:
    text: str
    model_reported: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: Optional[float]
    finish_reason: Optional[str]
    raw: dict = field(repr=False)

    @classmethod
    def from_openai(cls, resp: Any) -> "ChatResult":
        choice = resp.choices[0]
        msg = choice.message
        text = (msg.content or "").strip()
        usage = resp.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0
        cost = None
        try:
            raw_dict = resp.model_dump()
        except Exception:
            raw_dict = {}
        u = (raw_dict.get("usage") or {})
        if "cost" in u:
            try:
                cost = float(u["cost"])
            except (TypeError, ValueError):
                cost = None
        return cls(
            text=text,
            model_reported=getattr(resp, "model", "") or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            finish_reason=getattr(choice, "finish_reason", None),
            raw=raw_dict,
        )


_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)


class OpenRouterClient:
    """Thin async wrapper with retry/backoff and a global concurrency cap."""

    def __init__(
        self,
        *,
        api_key: str,
        app_name: str = "syco-bench",
        site_url: str = "https://example.org/syco",
        max_concurrent: int = 6,
        timeout_s: float = 60.0,
    ):
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers={"HTTP-Referer": site_url, "X-Title": app_name},
            timeout=timeout_s,
        )
        self._sem = asyncio.Semaphore(max_concurrent)

    async def aclose(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    async def _call(self, **kwargs) -> Any:
        return await self._client.chat.completions.create(**kwargs)

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        extra_body: Optional[dict] = None,
    ) -> ChatResult:
        body: dict[str, Any] = {"usage": {"include": True}}
        if extra_body:
            body.update(extra_body)
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            extra_body=body,
        )
        if seed is not None:
            kwargs["seed"] = seed
        async with self._sem:
            try:
                resp = await self._call(**kwargs)
            except BadRequestError:
                raise
        return ChatResult.from_openai(resp)
