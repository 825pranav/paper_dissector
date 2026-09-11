"""Stage 2: Extract structured, falsifiable claims from parsed paper."""

from __future__ import annotations

import logging

from paper_dissector.config import (
    EXTRACTION_CHUNK_CHARS, EXTRACTION_CONTEXT_CHARS, MAX_CLAIMS,
)
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


def _claims_from_response(parsed: dict, start_index: int = 1) -> list[Claim]:
    """Turn one extraction response into Claim objects, skipping malformed rows."""
    raw_claims = parsed.get("claims")
    if not isinstance(raw_claims, list):
        # Some models return a bare list, which chat_json wraps as {"items": [...]}.
        raw_claims = parsed.get("items", [])

    out: list[Claim] = []
    for offset, raw in enumerate(raw_claims):
        claim = _build_claim(raw, start_index + offset)
        if claim is not None:
            out.append(claim)
    return out


def _chunk_markdown(markdown: str, chunk_chars: int) -> list[str]:
    """Split the paper on section boundaries into chunks of roughly chunk_chars."""
    blocks, current = [], []
    for line in markdown.splitlines():
        if line.lstrip().startswith("#") and current:
            blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))

    chunks: list[str] = []
    buf = ""
    for block in blocks:
        # A single oversized section is split on its own rather than dropped.
        while len(block) > chunk_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(block[:chunk_chars])
            block = block[chunk_chars:]
        if len(buf) + len(block) > chunk_chars and buf:
            chunks.append(buf)
            buf = block
        else:
            buf = f"{buf}\n{block}" if buf else block
    if buf.strip():
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


def _extract_chunked(markdown: str) -> list[Claim]:
    """
    Fallback extraction for a small-context provider.

    The primary extractor sees the whole paper in one call. When that provider
    is unavailable — Gemini's 20-requests-per-day exhausted, or the endpoint
    down — the paper is split into section-aligned chunks that fit Groq's
    per-request ceiling and each chunk is extracted separately.
    """
    chunks = _chunk_markdown(markdown, EXTRACTION_CHUNK_CHARS)
    log.info("falling back to chunked extraction over %d chunk(s)", len(chunks))

    claims: list[Claim] = []
    for i, chunk in enumerate(chunks, start=1):
        try:
            parsed = chat_json(
                "claim_extractor_fallback",
                SYSTEM_PROMPT,
                (
                    f"This is part {i} of {len(chunks)} of a research paper. "
                    f"Extract the falsifiable claims it contains:\n\n{chunk}"
                ),
                temperature=0.1,
            )
        except Exception as exc:
            log.warning("chunked extraction failed on chunk %d/%d: %s", i, len(chunks), exc)
            continue
        claims.extend(_claims_from_response(parsed, start_index=len(claims) + 1))

    return claims


def extract_claims(state: PaperState) -> dict:
    """LangGraph node: extract claims from parsed markdown."""
    markdown = state.get("parsed_markdown") or ""
    if not markdown.strip():
        log.error("no parsed markdown available; cannot extract claims")
        return {"claims": []}

    # The extractor runs on a large-context provider so it sees the whole paper;
    # this ceiling only guards against a pathologically long document.
    if len(markdown) > EXTRACTION_CONTEXT_CHARS:
        log.info("truncating paper from %d to %d chars for extraction",
                 len(markdown), EXTRACTION_CONTEXT_CHARS)
        markdown = markdown[:EXTRACTION_CONTEXT_CHARS]

    claims: list[Claim] = []
    try:
        parsed = chat_json(
            "claim_extractor",
            SYSTEM_PROMPT,
            f"Extract all falsifiable claims from this paper:\n\n{markdown}",
            temperature=0.1,
        )
        claims = _claims_from_response(parsed)
    except Exception as exc:
        log.warning("primary claim extraction failed (%s); trying chunked fallback", exc)

    if not claims:
        claims = _extract_chunked(markdown)

    if not claims:
        log.error("claim extraction produced nothing; downstream stages will be empty")
        return {"claims": []}

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
