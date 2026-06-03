from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import httpx
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from .config import get_settings

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"


class LLMError(RuntimeError):
    """Raised when the LLM request fails after retries."""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str = DEFAULT_MODEL_NAME
    timeout_seconds: float = 30.0


def _safe_preview(text: str, limit: int = 200) -> str:
    compact = " ".join(text.split())
    return compact[:limit] + "…" if len(compact) > limit else compact


def build_openrouter_client(config: LLMConfig) -> OpenAI:
    timeout = httpx.Timeout(
        timeout=config.timeout_seconds,
        connect=min(10.0, config.timeout_seconds),
        read=config.timeout_seconds,
        write=config.timeout_seconds,
        pool=config.timeout_seconds,
    )
    return OpenAI(
        api_key=config.api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=timeout,
        max_retries=0,
    )


def _default_config() -> LLMConfig:
    settings = get_settings()
    return LLMConfig(api_key=settings.openrouter_api_key, model=DEFAULT_MODEL_NAME)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=0.5, max=8.0),
    retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
)
def _call_llm(client: OpenAI, model: str, messages: List[Dict[str, str]]) -> str:
    logger.debug("Calling LLM model=%s", model)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise LLMError("LLM returned an empty response")
    return content


def _call_llm_stream(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
) -> Iterator[str]:
    """Yield text tokens from the LLM as they arrive."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def generate_answer(
    messages: List[Dict[str, str]],
    *,
    config: Optional[LLMConfig] = None,
) -> str:
    """Generate a complete answer from a messages list."""
    llm_config = config or _default_config()
    client = build_openrouter_client(llm_config)
    try:
        answer = _call_llm(client, llm_config.model, messages)
    except (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError, httpx.HTTPError) as exc:
        logger.exception("LLM request failed model=%s", llm_config.model)
        raise LLMError(str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected LLM error")
        raise LLMError("Unexpected LLM error") from exc
    logger.info("LLM response generated model=%s chars=%d", llm_config.model, len(answer))
    return answer


def generate_answer_stream(
    messages: List[Dict[str, str]],
    *,
    config: Optional[LLMConfig] = None,
) -> Iterator[str]:
    """Yield text tokens one by one for streaming responses."""
    llm_config = config or _default_config()
    client = build_openrouter_client(llm_config)
    try:
        yield from _call_llm_stream(client, llm_config.model, messages)
    except (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError, httpx.HTTPError) as exc:
        logger.exception("LLM stream failed model=%s", llm_config.model)
        raise LLMError(str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected LLM stream error")
        raise LLMError("Unexpected LLM error") from exc
