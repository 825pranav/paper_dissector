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
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from paper_dissector.config import get_client_for_agent

log = logging.getLogger(__name__)

# Providers that turned out to reject response_format={"type": "json_object"}.
# Populated at runtime the first time a call fails, so we only pay the cost once.
_NO_JSON_MODE: set[str] = set()

_TRANSIENT = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

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


# ── Chat wrappers ────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(_TRANSIENT),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _create(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def chat_text(
    agent_name: str,
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str:
    """Plain text completion for a named agent role, with retry on transient errors."""
    client, model = get_client_for_agent(agent_name)
    kwargs: dict = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = _create(client, **kwargs)
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
    provider_key = f"{type(client).__name__}:{str(client.base_url)}"
    use_json_mode = provider_key not in _NO_JSON_MODE

    def _call(json_mode: bool) -> str:
        sys_content = system if json_mode else system + _JSON_NUDGE
        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = _create(client, **kwargs)
        return resp.choices[0].message.content or ""

    if use_json_mode:
        try:
            raw = _call(json_mode=True)
        except (BadRequestError, TypeError) as exc:
            log.warning(
                "provider %s rejected JSON mode (%s); falling back to prompt-based JSON",
                client.base_url, exc,
            )
            _NO_JSON_MODE.add(provider_key)
            raw = _call(json_mode=False)
    else:
        raw = _call(json_mode=False)

    try:
        return extract_json(raw)
    except LLMJSONError:
        # One reinforced retry before giving up — models often comply on a nudge.
        log.warning("unparseable JSON from %s; retrying with explicit instruction", agent_name)
        _NO_JSON_MODE.add(provider_key)
        return extract_json(_call(json_mode=False))
