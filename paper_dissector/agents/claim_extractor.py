"""Stage 2: Extract structured, falsifiable claims from parsed paper."""

from __future__ import annotations

import logging

from paper_dissector.config import AUDIT_CONTEXT_CHARS, MAX_CLAIMS
from paper_dissector.llm import chat_json
from paper_dissector.sanitize import as_text
from paper_dissector.schemas import Claim, ClaimExtractionResult
from paper_dissector.state import PaperState

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a scientific claim extraction agent. Your job is to extract
specific, falsifiable empirical claims from a research paper.

RULES:
- Extract only EMPIRICAL claims — quantitative results, performance comparisons, causal assertions.
- Skip hedged/qualified statements ("may", "could", "we believe").
- Skip background/related work claims — only extract claims the AUTHORS make about THEIR work.
- Each claim must be atomic — one measurable assertion per claim.
- Fill in ALL schema fields. If a baseline isn't mentioned, set it to null.
- The falsifiability_threshold should specify what evidence would DISPROVE the claim.
- Target 5–20 claims per paper depending on paper length.

Respond ONLY with valid JSON matching this schema:
{
  "claims": [
    {
      "claim_id": "C1",
      "raw_text": "exact quote from paper",
      "subject": "model/system name",
      "intervention": "what was done",
      "baseline": "what it's compared to or null",
      "metric": "F1/accuracy/BLEU/etc or null",
      "reported_value": 94.2 or "significant" or null,
      "scope": "dataset, setting, constraints",
      "falsifiability_threshold": "what would disprove this",
      "source_section": "Section X.Y / Table N"
    }
  ]
}"""


def _build_claim(raw: dict, index: int) -> Claim | None:
    """Construct a Claim from loose model JSON, filling gaps rather than crashing."""
    if not isinstance(raw, dict):
        return None

    text = as_text(raw.get("raw_text") or raw.get("claim") or raw.get("text"))
    if not text:
        log.warning("skipping claim %d with no raw_text", index)
        return None

    reported = raw.get("reported_value")
    if isinstance(reported, (dict, list)):
        reported = as_text(reported) or None

    def optional(key: str) -> str | None:
        value = as_text(raw.get(key))
        return value if value and value.lower() not in ("null", "none", "n/a") else None

    try:
        return Claim(
            claim_id=as_text(raw.get("claim_id")) or f"C{index}",
            raw_text=text,
            subject=as_text(raw.get("subject"), default="unspecified"),
            intervention=as_text(raw.get("intervention"), default="unspecified"),
            baseline=optional("baseline"),
            metric=optional("metric"),
            reported_value=reported,
            scope=as_text(raw.get("scope"), default="unspecified"),
            falsifiability_threshold=as_text(
                raw.get("falsifiability_threshold"), default="unspecified"
            ),
            source_section=as_text(raw.get("source_section"), default="unspecified"),
        )
    except Exception as exc:
        log.warning("skipping malformed claim %d: %s", index, exc)
        return None


def extract_claims(state: PaperState) -> dict:
    """LangGraph node: extract claims from parsed markdown."""
    markdown = state.get("parsed_markdown") or ""
    if not markdown.strip():
        log.error("no parsed markdown available; cannot extract claims")
        return {"claims": []}

    # Gemini Flash handles 1M tokens, but a huge paper still costs latency and
    # can trip free-tier per-request limits — keep the body bounded.
    if len(markdown) > AUDIT_CONTEXT_CHARS:
        log.info("truncating paper from %d to %d chars for extraction",
                 len(markdown), AUDIT_CONTEXT_CHARS)
        markdown = markdown[:AUDIT_CONTEXT_CHARS]

    try:
        parsed = chat_json(
            "claim_extractor",
            SYSTEM_PROMPT,
            f"Extract all falsifiable claims from this paper:\n\n{markdown}",
            temperature=0.1,
        )
    except Exception as exc:
        log.error("claim extraction failed: %s", exc)
        return {"claims": []}

    raw_claims = parsed.get("claims")
    if not isinstance(raw_claims, list):
        # Some models return a bare list, which chat_json wraps as {"items": [...]}.
        raw_claims = parsed.get("items", [])

    claims: list[Claim] = []
    for i, raw in enumerate(raw_claims, start=1):
        claim = _build_claim(raw, i)
        if claim is not None:
            claims.append(claim)

    claims = deduplicate_claims(claims)

    if MAX_CLAIMS and len(claims) > MAX_CLAIMS:
        log.info("capping %d extracted claims at MAX_CLAIMS=%d", len(claims), MAX_CLAIMS)
        claims = claims[:MAX_CLAIMS]

    # Re-key sequentially so downstream joins are stable even if the model
    # reused or skipped identifiers.
    for i, claim in enumerate(claims, start=1):
        claim.claim_id = f"C{i}"

    log.info("extracted %d claims", len(claims))
    return {"claims": claims}


def deduplicate_claims(claims: list[Claim], threshold: float = 0.88) -> list[Claim]:
    """
    Merge near-duplicate claims, keeping the more complete of each pair.

    Uses token overlap plus a sequence-ratio check on ``raw_text``; models often
    emit the abstract's version and the results section's version of one result.
    """
    from difflib import SequenceMatcher

    def completeness(claim: Claim) -> int:
        return sum(
            1 for v in (claim.baseline, claim.metric, claim.reported_value) if v is not None
        )

    kept: list[Claim] = []
    for claim in claims:
        normalised = " ".join(claim.raw_text.lower().split())
        duplicate_of = None

        for i, existing in enumerate(kept):
            other = " ".join(existing.raw_text.lower().split())
            ratio = SequenceMatcher(None, normalised, other).ratio()
            if ratio < threshold:
                # Cheap token-overlap check catches reworded duplicates.
                a, b = set(normalised.split()), set(other.split())
                if not a or not b:
                    continue
                overlap = len(a & b) / min(len(a), len(b))
                same_number = (
                    claim.reported_value is not None
                    and claim.reported_value == existing.reported_value
                )
                if not (overlap > 0.85 and same_number):
                    continue
            duplicate_of = i
            break

        if duplicate_of is None:
            kept.append(claim)
        elif completeness(claim) > completeness(kept[duplicate_of]):
            kept[duplicate_of] = claim

    if len(kept) < len(claims):
        log.info("deduplicated %d claims down to %d", len(claims), len(kept))
    return kept
