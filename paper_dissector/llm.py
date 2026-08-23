"""Shared LLM call helpers: JSON-mode fallback, retries, graceful degradation.

All agents route their chat calls through here so that provider quirks
(e.g. an OpenAI-compatible endpoint that rejects ``response_format``) and
transient rate limits are handled in exactly one place.
"""

from __future__ import annotations

import json
import logging
import re

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from paper_dissector.config import (
    get_client_for_agent, input_token_budget, provider_for_agent,
)

log = logging.getLogger(__name__)

# Endpoints that turned out to reject response_format={"type": "json_object"}.
# Populated at runtime the first time a call fails, so we only pay the cost once.
# Keyed by base_url AND model: JSON-mode support varies per model on a single
# provider (on Groq, gpt-oss-120b 400s on it while qwen3.8-27b handles it), so
# keying by provider alone would disable JSON mode for models that support it.
_NO_JSON_MODE: set[str] = set()


def _endpoint_key(client, model: str) -> str:
    return f"{client.base_url}::{model}"

_TRANSIENT = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

# A per-minute throttle clears on its own; a per-day cap does not. Retrying the
# latter just burns minutes of backoff before failing anyway, so it is raised
# straight away and the caller's fallback path takes over.
_DAILY_QUOTA_MARKERS = (
    "perday",
    "per day",
    "requests per day",
    "free_tier_requests",
    "resource_exhausted",
)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):
        message = str(exc).lower()
        if any(marker in message for marker in _DAILY_QUOTA_MARKERS):
            log.warning("daily quota exhausted, not retrying: %s", message[:160])
            return False
        return True
    return isinstance(exc, (APIConnectionError, APITimeoutError, InternalServerError))

# Appended to the system prompt when we cannot use native JSON mode.
_JSON_NUDGE = (
    "\n\nIMPORTANT: Respond with a single valid JSON object and NOTHING else. "
    "No markdown code fences, no commentary before or after the JSON."
)


class LLMJSONError(ValueError):
    """Raised when a model response cannot be coerced into JSON."""


# ── JSON extraction ──────────────────────────────────────────────

def extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response that may be wrapped in prose/fences."""
    if not text:
        raise LLMJSONError("empty response from model")

    candidate = text.strip()

    # Strip ```json ... ``` or ``` ... ``` fences.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}
    except json.JSONDecodeError:
        pass

    # Fall back to the first balanced {...} block in the text.
    block = _first_balanced_object(candidate)
    if block is not None:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            # Last resort: models sometimes emit trailing commas.
            repaired = re.sub(r",\s*([}\]])", r"\1", block)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as exc:
                raise LLMJSONError(f"could not parse JSON from response: {exc}") from exc

    raise LLMJSONError(f"no JSON object found in response: {text[:300]!r}")


def _first_balanced_object(text: str) -> str | None:
    """Return the first brace-balanced ``{...}`` substring, ignoring braces in strings."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# ── Request size guard ───────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """
    Conservative token estimate.

    Deliberately over-counts: the common four-chars-per-token rule underestimated
    a real judge payload by about 8% (estimated 6500, Groq measured 7036) and the
    request was rejected. Over-estimating only costs a little context; under-
    estimating costs the whole call.
    """
    return int(len(text) / 3.4) + 1


def _is_too_large(exc: BaseException) -> bool:
    """True for a 'request too large' rejection, which shrinking can fix."""
    message = str(exc).lower()
    return "413" in message or "request too large" in message or "reduce your message size" in message


def _fit_to_budget(messages: list[dict], provider: str, budget: int | None = None) -> list[dict]:
    """
    Trim message content so a request cannot exceed the provider's per-request
    ceiling.

    Groq's free tier allows only 8000 tokens per minute, and a single request
    larger than that is rejected outright with HTTP 413 rather than throttled —
    which previously killed claim extraction and produced an empty report.
    Truncating loses some context; failing loses the whole stage.
    """
    budget = budget if budget is not None else input_token_budget(provider)
    if not budget:
        return messages

    total = sum(
        _estimate_tokens(m["content"]) if isinstance(m.get("content"), str) else 400
        for m in messages
    )
    if total <= budget:
        return messages

    # Trim the largest string payload (the document body), never the system
    # prompt, which carries the output contract.
    trimmed = [dict(m) for m in messages]
    candidates = [
        i for i, m in enumerate(trimmed)
        if m.get("role") == "user" and isinstance(m.get("content"), str)
    ]
    if not candidates:
        log.warning("request of ~%d tokens exceeds the %d budget but cannot be trimmed",
                    total, budget)
        return messages

    biggest = max(candidates, key=lambda i: len(trimmed[i]["content"]))
    overflow_tokens = total - budget
    content = trimmed[biggest]["content"]
    # Use the same ratio the estimator does, plus a fixed margin for the marker
    # and any tokeniser variance.
    keep_chars = max(1000, len(content) - int(overflow_tokens * 3.4) - 600)

    if keep_chars < len(content):
        log.warning(
            "trimming request from ~%d to ~%d tokens to fit the %s per-request budget",
            total, budget, provider,
        )
        trimmed[biggest]["content"] = (
            content[:keep_chars] + "\n\n[...truncated to fit the provider's request limit...]"
        )
    return trimmed


# ── Chat wrappers ────────────────────────────────────────────────

@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _create(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def _create_fitting(client, provider: str, messages: list[dict], **kwargs):
    """
    Send a request, shrinking it and retrying if the provider says it is too large.

    Token estimation cannot be exact and per-model ceilings differ (on Groq,
    qwen3.8-27b allows 7000 input tokens where gpt-oss-120b allows 8000), so a
    rejection is treated as a signal to cut the payload rather than as a failure.
    """
    budget = input_token_budget(provider)
    last_exc: BaseException | None = None

    for attempt in range(3):
        fitted = _fit_to_budget(messages, provider, budget)
        try:
            return _create(client, messages=fitted, **kwargs)
        except Exception as exc:
            if not _is_too_large(exc) or attempt == 2:
                raise
            last_exc = exc
            budget = max(1200, int(budget * 0.6))
            log.warning("request rejected as too large; retrying at a %d-token budget", budget)

    raise last_exc  # pragma: no cover - loop always returns or raises above


def chat_text(
    agent_name: str,
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str:
    """Plain text completion for a named agent role, with retry on transient errors."""
    client, model = get_client_for_agent(agent_name)
    kwargs: dict = {"model": model, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = _create_fitting(client, provider_for_agent(agent_name), messages, **kwargs)
    return resp.choices[0].message.content or ""


def chat_json(
    agent_name: str,
    system: str,
    user: str | list,
    temperature: float = 0.1,
    max_tokens: int | None = None,
) -> dict:
    """
    Completion for a named agent role that must return a JSON object.

    Tries native ``response_format={"type": "json_object"}`` first. If the provider
    rejects it, remembers that and falls back to prompt-based JSON with tolerant
    parsing (fence stripping, balanced-brace scan, trailing-comma repair).
    """
    client, model = get_client_for_agent(agent_name)
    endpoint_key = _endpoint_key(client, model)
    use_json_mode = endpoint_key not in _NO_JSON_MODE

    def _call(json_mode: bool) -> str:
        sys_content = system if json_mode else system + _JSON_NUDGE
        kwargs: dict = {"model": model, "temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = _create_fitting(
            client,
            provider_for_agent(agent_name),
            [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    if use_json_mode:
        try:
            raw = _call(json_mode=True)
        except (BadRequestError, TypeError) as exc:
            log.warning(
                "%s rejected JSON mode (%s); falling back to prompt-based JSON",
                endpoint_key, exc,
            )
            _NO_JSON_MODE.add(endpoint_key)
            raw = _call(json_mode=False)
    else:
        raw = _call(json_mode=False)

    try:
        return extract_json(raw)
    except LLMJSONError:
        # One reinforced retry before giving up — models often comply on a nudge.
        log.warning("unparseable JSON from %s; retrying with explicit instruction", agent_name)
        _NO_JSON_MODE.add(endpoint_key)
        return extract_json(_call(json_mode=False))
