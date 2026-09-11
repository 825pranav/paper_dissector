"""Stance classification: does a passage SUPPORT, CONTRADICT, or remain NEUTRAL toward a claim?"""

from __future__ import annotations

import logging

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from paper_dissector import cache
from paper_dissector.config import HF_API_KEY, HF_ENABLED, HF_INFERENCE_BASE
from paper_dissector.schemas import Stance

log = logging.getLogger(__name__)

# DeBERTa-v3 fine-tuned on NLI — good zero-shot stance classification
MODEL_ID = "cross-encoder/nli-deberta-v3-base"
LABEL_MAP = {
    "ENTAILMENT": Stance.SUPPORT,
    "CONTRADICTION": Stance.CONTRADICT,
    "NEUTRAL": Stance.NEUTRAL,
}

_MAX_PASSAGE_CHARS = 1500

# Flipped off for the rest of the process once HF proves unavailable, so we stop
# paying the timeout cost on every single passage.
_hf_available = HF_ENABLED and bool(HF_API_KEY)


class _HFTransient(RuntimeError):
    """HF inference is warming up or rate limited."""


def _normalise_label(label: str) -> Stance:
    """Map the many spellings NLI checkpoints use onto our Stance enum."""
    key = (label or "").strip().upper().replace("-", "_")
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    # Checkpoints variously emit LABEL_0/1/2, "entail", "refutes", etc.
    if key.startswith("ENTAIL") or key in ("LABEL_0", "SUPPORT", "SUPPORTS"):
        return Stance.SUPPORT
    if key.startswith("CONTRA") or key in ("LABEL_2", "REFUTE", "REFUTES"):
        return Stance.CONTRADICT
    return Stance.NEUTRAL


# ── HuggingFace path ─────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(_HFTransient),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _hf_request(claim: str, passage: str) -> list | dict:
    resp = httpx.post(
        f"{HF_INFERENCE_BASE}/models/{MODEL_ID}",
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        # Cross-encoder NLI is a text-classification model over a sentence pair.
        json={
            "inputs": {"text": passage, "text_pair": claim},
            "options": {"wait_for_model": True},
        },
        timeout=30,
    )
    if resp.status_code in (429, 503):
        raise _HFTransient(f"HF transient ({resp.status_code})")
    if resp.status_code >= 500:
        raise _HFTransient(f"HF server error ({resp.status_code})")
    resp.raise_for_status()
    return resp.json()


def _hf_classify(claim: str, passage: str) -> tuple[Stance, float] | None:
    """Try the HF NLI endpoint. Returns None if HF is unusable."""
    global _hf_available
    if not _hf_available:
        return None

    try:
        result = _hf_request(claim, passage)
    except Exception as exc:
        log.warning("HF stance classification unavailable (%s); using LLM fallback", exc)
        _hf_available = False
        return None

    # HF returns [[{label, score}, ...]] for a single pair, or [{label, score}, ...]
    if isinstance(result, list) and result and isinstance(result[0], list):
        result = result[0]

    if isinstance(result, dict) and "labels" in result:
        return _normalise_label(result["labels"][0]), round(float(result["scores"][0]), 4)

    if isinstance(result, list) and result and isinstance(result[0], dict):
        best = max(result, key=lambda x: x.get("score", 0))
        return _normalise_label(best.get("label", "")), round(float(best.get("score", 0.0)), 4)

    log.warning("unexpected HF response shape: %s", str(result)[:200])
    _hf_available = False
    return None


# ── LLM fallback path ────────────────────────────────────────────

_LLM_SYSTEM = """You are a scientific stance classifier. For each numbered passage,
decide its stance toward the CLAIM.

SUPPORT — the passage is evidence that the claim is true:
- it reports a measurement of the same quantity that agrees with the claim
- it independently replicates or corroborates the claimed result
- it reports a finding that would be unlikely if the claim were false
A passage SUPPORTS the claim even if it uses different wording, reports a
slightly different number in the same direction, or never cites the claim's
authors. Independent corroboration is the strongest kind of support.

CONTRADICT — the passage is evidence that the claim is false or overstated:
- a failed replication, or a measurement of the same quantity that disagrees
- a result showing the claimed effect is smaller, absent, or explained away

NEUTRAL — the passage does not bear on whether the claim is true:
- a different task, dataset, metric or population
- merely the same research area, with no measurement relevant to the claim

Judge relevance, not politeness: do not default to NEUTRAL for a passage that
genuinely measures the same thing. Reserve NEUTRAL for passages that leave the
claim's truth untouched.

Respond ONLY with JSON:
{"results": [{"index": 1, "stance": "SUPPORT", "confidence": 0.82}]}
Include exactly one entry per passage, in order."""


def _llm_classify_batch(claim: str, passages: list[str]) -> list[tuple[Stance, float]]:
    """Classify every passage for a claim in a single LLM call."""
    from paper_dissector.llm import chat_json  # local import avoids an import cycle

    neutral: list[tuple[Stance, float]] = [(Stance.NEUTRAL, 0.0)] * len(passages)
    if not passages:
        return []

    numbered = "\n\n".join(
        f"[{i + 1}] {p[:_MAX_PASSAGE_CHARS]}" for i, p in enumerate(passages)
    )
    try:
        parsed = chat_json(
            "stance_classifier",
            _LLM_SYSTEM,
            f"CLAIM:\n{claim}\n\nPASSAGES:\n{numbered}",
            temperature=0.0,
        )
    except Exception as exc:
        log.warning("LLM stance fallback failed (%s); defaulting to NEUTRAL", exc)
        return neutral

    out = list(neutral)
    for row in parsed.get("results", []) or []:
        try:
            idx = int(row.get("index", 0)) - 1
            if not 0 <= idx < len(passages):
                continue
            confidence = min(max(float(row.get("confidence", 0.5)), 0.0), 1.0)
            out[idx] = (_normalise_label(str(row.get("stance", ""))), round(confidence, 4))
        except (AttributeError, TypeError, ValueError):
            continue
    return out


# ── Public API ───────────────────────────────────────────────────

def classify_stance(claim: str, passage: str) -> tuple[Stance, float]:
    """
    Classify whether a passage supports, contradicts, or is neutral toward a claim.

    Tries the HuggingFace NLI Inference API, then falls back to an LLM classifier,
    then to NEUTRAL. It never raises.

    Returns:
        (stance, confidence) e.g. (Stance.CONTRADICT, 0.87)
    """
    passage = (passage or "")[:_MAX_PASSAGE_CHARS]
    if not claim or not passage:
        return Stance.NEUTRAL, 0.0

    key = cache.make_key("stance", [MODEL_ID, claim, passage])
    cached = cache.get(key)
    if cached is not None:
        return Stance(cached[0]), cached[1]

    result = _hf_classify(claim, passage)
    if result is None:
        result = _llm_classify_batch(claim, [passage])[0]

    cache.set(key, (result[0].value, result[1]))
    return result


def batch_classify(claim: str, passages: list[str]) -> list[tuple[Stance, float]]:
    """
    Classify multiple passages against the same claim.

    Uses HF per passage while it is available; once it isn't, the whole remaining
    batch is classified in a single LLM call rather than one call per passage.
    """
    if not passages:
        return []

    results: list[tuple[Stance, float] | None] = [None] * len(passages)
    pending: list[int] = []

    for i, passage in enumerate(passages):
        clipped = (passage or "")[:_MAX_PASSAGE_CHARS]
        if not clipped:
            results[i] = (Stance.NEUTRAL, 0.0)
            continue

        key = cache.make_key("stance", [MODEL_ID, claim, clipped])
        cached = cache.get(key)
        if cached is not None:
            results[i] = (Stance(cached[0]), cached[1])
            continue

        hf_result = _hf_classify(claim, clipped)
        if hf_result is not None:
            cache.set(key, (hf_result[0].value, hf_result[1]))
            results[i] = hf_result
        else:
            pending.append(i)

    if pending:
        batch = _llm_classify_batch(
            claim, [(passages[i] or "")[:_MAX_PASSAGE_CHARS] for i in pending]
        )
        for slot, result in zip(pending, batch):
            clipped = (passages[slot] or "")[:_MAX_PASSAGE_CHARS]
            cache.set(
                cache.make_key("stance", [MODEL_ID, claim, clipped]),
                (result[0].value, result[1]),
            )
            results[slot] = result

    return [r if r is not None else (Stance.NEUTRAL, 0.0) for r in results]
