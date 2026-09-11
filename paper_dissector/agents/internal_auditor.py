"""Stage 3: Internal methodology audit — does the paper's own data support its claims?"""

from __future__ import annotations

import logging
import re

from paper_dissector.config import AUDIT_CONTEXT_CHARS, MAX_FIGURES_PER_CLAIM
from paper_dissector.llm import chat_json, chat_text
from paper_dissector.sanitize import as_bool, as_text, clamp01, coerce_enum
from paper_dissector.schemas import AuditSeverity, Claim, InternalAuditResult
from paper_dissector.state import PaperState

log = logging.getLogger(__name__)

_REF_RE = re.compile(r"\b(figure|fig\.?|table|tab\.?)\s*([0-9]+|[ivxlc]+)\b", re.IGNORECASE)

AUDIT_SYSTEM_PROMPT = """You are a methodology auditor for scientific papers. Given a paper's
full text and a specific claim, you must check whether the paper's OWN data supports the claim.

CHECK EACH OF THESE:
1. TABLE CONSISTENCY: Does the claim's number match what's in the paper's tables?
2. FIGURE CONSISTENCY: Does the described trend match what figures show?
3. BASELINE PRESENCE: Is the claimed baseline actually present in experiments?
4. STATISTICAL RIGOR: Are p-values, CIs, or effect sizes reported where needed?
5. METHODOLOGY GAP: Does the experimental design actually test what the claim asserts?

For each check, assign: PASS, WARN, FAIL, or MISMATCH.
Compute a mismatch_score from 0.0 (perfectly supported) to 1.0 (total mismatch).

Respond ONLY with valid JSON:
{
  "claim_id": "C1",
  "table_consistency": "PASS",
  "table_detail": null or "explanation",
  "figure_consistency": "WARN",
  "figure_detail": null or "explanation",
  "visual_mismatch_detail": null or "what the figure actually shows vs what claim says",
  "baseline_present": true,
  "statistical_rigor": "WARN",
  "statistical_detail": null or "explanation",
  "methodology_gap": null or "explanation",
  "mismatch_score": 0.25
}"""

VISION_PROMPT = """You are analyzing a figure/table from a scientific paper.
Extract ALL quantitative information visible in this image:
- Exact numbers, data points, bar heights, line values
- Axis labels, units, scales
- Trends: increasing, decreasing, flat, overlapping
- Error bars, confidence intervals if visible
- Which method/model appears to perform best

Be precise. Report exact numbers where readable. If error bars overlap between
two methods, say so explicitly."""


# ── Figure ↔ claim matching ──────────────────────────────────────

def _normalise_ref(kind: str, number: str) -> str:
    """Canonicalise a 'Fig. 3' / 'Table IV' reference to 'figure:3' / 'table:iv'."""
    kind = kind.lower().rstrip(".")
    kind = "figure" if kind.startswith("fig") else "table"
    return f"{kind}:{number.lower()}"


def _claim_figure_refs(claim: Claim) -> list[str]:
    """Every figure/table reference a claim makes, in priority order."""
    refs: list[str] = []
    # source_section is the authoritative pointer; raw_text is a weaker signal.
    for text in (claim.source_section or "", claim.raw_text or ""):
        for kind, number in _REF_RE.findall(text):
            ref = _normalise_ref(kind, number)
            if ref not in refs:
                refs.append(ref)
    return refs


def match_figures_for_claim(claim: Claim, figures: list[dict]) -> list[dict]:
    """
    Find the figures a claim actually cites.

    Matches on the figure/table number parsed out of the claim's ``source_section``
    (falling back to its text), so a claim about Table 2 is checked against Table 2
    rather than against whichever figure happened to be extracted first.
    """
    if not figures:
        return []

    refs = _claim_figure_refs(claim)
    if not refs:
        return []

    by_number: dict[str, list[dict]] = {}
    for fig in figures:
        by_number.setdefault(str(fig.get("figure_number", "")).lower(), []).append(fig)

    matched: list[dict] = []
    for ref in refs:
        for fig in by_number.get(ref, []):
            if fig not in matched:
                matched.append(fig)
        if len(matched) >= MAX_FIGURES_PER_CLAIM:
            break

    return matched[:MAX_FIGURES_PER_CLAIM]


# ── Audit steps ──────────────────────────────────────────────────

def _audit_claim_text(claim: Claim, markdown: str) -> dict:
    """Audit a claim against the paper text."""
    return chat_json(
        "internal_auditor",
        AUDIT_SYSTEM_PROMPT,
        (
            f"CLAIM TO AUDIT:\n{claim.model_dump_json(indent=2)}\n\n"
            f"FULL PAPER TEXT:\n{markdown[:AUDIT_CONTEXT_CHARS]}"
        ),
        temperature=0.1,
    )


def _verify_figure(claim: Claim, figure: dict) -> str | None:
    """Use vision model to verify claim against a figure image."""
    if not figure.get("image_b64"):
        return None

    try:
        return chat_text(
            "visual_verifier",
            [
                {"role": "system", "content": VISION_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": (
                        f"This figure is from a paper. A claim states: '{claim.raw_text}'\n"
                        f"The claim references: {claim.source_section}\n"
                        f"Figure caption: {figure.get('caption') or 'N/A'}\n\n"
                        f"Extract all data from this figure and assess whether it supports the claim."
                    )},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{figure['image_b64']}"
                    }},
                ]},
            ],
            temperature=0.1,
        )
    except Exception as exc:
        log.warning("visual verification failed for %s on %s: %s",
                    claim.claim_id, figure.get("figure_id"), exc)
        return None


def _build_audit(raw: dict, claim_id: str) -> InternalAuditResult:
    """Coerce loose audit JSON into a valid InternalAuditResult."""
    def severity(key: str) -> AuditSeverity:
        return coerce_enum(raw.get(key), AuditSeverity, AuditSeverity.WARN)

    def detail(key: str) -> str | None:
        value = as_text(raw.get(key))
        return value if value and value.lower() not in ("null", "none", "n/a") else None

    return InternalAuditResult(
        claim_id=claim_id,
        table_consistency=severity("table_consistency"),
        table_detail=detail("table_detail"),
        figure_consistency=severity("figure_consistency"),
        figure_detail=detail("figure_detail"),
        visual_mismatch_detail=detail("visual_mismatch_detail"),
        baseline_present=as_bool(raw.get("baseline_present"), default=False),
        statistical_rigor=severity("statistical_rigor"),
        statistical_detail=detail("statistical_detail"),
        methodology_gap=detail("methodology_gap"),
        mismatch_score=clamp01(raw.get("mismatch_score"), default=0.5),
    )


def _degraded_audit(claim_id: str, reason: str) -> InternalAuditResult:
    """Placeholder audit used when a claim's audit fails, so the pipeline continues."""
    return InternalAuditResult(
        claim_id=claim_id,
        table_consistency=AuditSeverity.WARN,
        table_detail=f"Audit unavailable: {reason}",
        figure_consistency=AuditSeverity.WARN,
        baseline_present=False,
        statistical_rigor=AuditSeverity.WARN,
        mismatch_score=0.5,
    )


def audit_claims(state: PaperState) -> dict:
    """LangGraph node: audit each claim against the paper's own data."""
    results: list[InternalAuditResult] = []
    figures = state.get("extracted_figures") or []
    markdown = state.get("parsed_markdown") or ""
    claims = state.get("claims") or []

    for claim in claims:
        # One failing claim must not take down the remaining claims.
        try:
            audit_raw = _audit_claim_text(claim, markdown)
        except Exception as exc:
            log.error("internal audit failed for %s: %s", claim.claim_id, exc)
            results.append(_degraded_audit(claim.claim_id, str(exc)[:200]))
            continue

        visual_notes = []
        for fig in match_figures_for_claim(claim, figures):
            output = _verify_figure(claim, fig)
            if output:
                label = fig.get("figure_number") or fig.get("figure_id")
                visual_notes.append(f"[{label}] {output}")

        if visual_notes:
            audit_raw["visual_mismatch_detail"] = "\n\n".join(visual_notes)

        try:
            results.append(_build_audit(audit_raw, claim.claim_id))
        except Exception as exc:
            log.error("could not build audit result for %s: %s", claim.claim_id, exc)
            results.append(_degraded_audit(claim.claim_id, str(exc)[:200]))

    log.info("audited %d/%d claims", len(results), len(claims))
    return {"internal_audits": results}
