"""Stage 3: Internal methodology audit — does the paper's own data support its claims?"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

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
   Papers often report results only in the text. If no table reports this
   quantity, that is not a failure: use PASS if the text states the number
   consistently, WARN if it is ambiguous. Use FAIL or MISMATCH only when a
   table reports a DIFFERENT value for the same quantity.
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


# Words that appear in most captions or describe any experiment, so sharing
# one says nothing about whether a figure shows this claim's result.
_GENERIC_TERMS = frozenset("""
the and for with from this that are was were its our their each all per via
figure fig table tab section results result dataset datasets data set sets
test tests train training trained validation val model models method methods
using used use show shows shown left right top bottom curve curves plot plots
median mean average error errors rate rates accuracy loss losses score scores
performance run runs setting settings experiment experiments light lighter dark
darker thin thick line lines number numbers image images input inputs
""".split())

_TERM_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_NUMBER_RE = re.compile(r"\d+\.\d+|\d{3,}")
_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−"})


def _terms(text: str) -> set[str]:
    """Distinctive lowercase terms, with unicode hyphens folded ("CIFAR‑10" == "CIFAR-10")."""
    out = set()
    for term in _TERM_RE.findall((text or "").translate(_DASHES).lower()):
        term = term.strip(".-")
        if term.isalpha() and len(term) > 4 and term.endswith("s"):
            term = term[:-1]
        if term.replace(".", "").isdigit() and not _NUMBER_RE.fullmatch(term):
            continue    # "3" in "Section 3" / "Table 3" identifies nothing

        if (len(term) >= 3 or any(ch.isdigit() for ch in term)) and term not in _GENERIC_TERMS:
            out.add(term)
    return out


# Acronyms and names with digits: "TIMIT", "MFCC", "CIFAR-10", "WMT14".
_NAMED_RE = re.compile(r"\b(?=[A-Za-z0-9-]*[A-Z0-9]{2})[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b")


def _named_terms(text: str) -> set[str]:
    """Terms written as names (acronyms, or containing digits) — datasets and benchmarks."""
    folded = (text or "").translate(_DASHES)
    return {
        t for match in _NAMED_RE.findall(folded)
        for t in _terms(match)
        if any(ch.isdigit() for ch in t) or match.isupper()
    }


def _claim_numbers(claim: Claim) -> set[str]:
    text = f"{claim.raw_text} {claim.reported_value if claim.reported_value is not None else ''}"
    return set(_NUMBER_RE.findall(text))


def _match_by_caption(claim: Claim, figures: list[dict]) -> list[dict]:
    """
    Match a claim that names no figure to the figure/table whose caption fits it.

    A caption must share an anchor with the claim: a term from the claim's scope
    or source section that few captions contain (a dataset or task name such as
    "TIMIT" or "CIFAR-10"), or one of the claim's reported numbers. Method names
    are deliberately not anchors: every claim in a paper mentions its method, so
    they would pull all claims onto whichever figure introduces it. Candidates
    are ranked by how many claim terms they share, weighted by rarity.
    """
    captioned = [(fig, _terms(fig.get("caption") or "")) for fig in figures]
    captioned = [(fig, terms) for fig, terms in captioned if terms]
    if not captioned:
        return []

    df = Counter(t for _, terms in captioned for t in terms)
    max_df = max(1, len(captioned) // 3)
    anchor_text = f"{claim.scope} {claim.source_section}"
    anchors = _terms(anchor_text)
    named = _named_terms(anchor_text)
    numbers = _claim_numbers(claim)
    context = anchors | _terms(
        f"{claim.raw_text} {claim.metric or ''} {claim.subject} {claim.intervention}"
    )

    scored = []
    for order, (fig, terms) in enumerate(captioned):
        shared_anchors = {t for t in anchors & terms if df[t] <= max_df} | (numbers & terms)
        if not shared_anchors:
            continue
        # A shared dataset name outweighs a shared common word ("classification").
        score = sum(6 if t in named or t in numbers else 2 for t in shared_anchors) + sum(
            math.log(1 + len(captioned) / df[t]) for t in context & terms
        )
        scored.append((-score, order, fig))

    return [fig for _, _, fig in sorted(scored, key=lambda t: (t[0], t[1]))]


def match_figures_for_claim(claim: Claim, figures: list[dict]) -> list[dict]:
    """
    Find the figures a claim actually cites.

    Matches on the figure/table number parsed out of the claim's ``source_section``
    (falling back to its text), so a claim about Table 2 is checked against Table 2
    rather than against whichever figure happened to be extracted first. Claims
    that name no figure — most of them, in practice — fall back to caption
    matching, without which the VLM never ran on a real paper.
    """
    matched = _match_by_reference(claim, figures) if figures else []
    if not matched and figures and not _claim_figure_refs(claim):
        matched = _match_by_caption(claim, figures)[:MAX_FIGURES_PER_CLAIM]
    if not matched:
        log.warning(
            "%s: no figure or table matched (%d extracted); visual verification skipped",
            claim.claim_id, len(figures),
        )
    return matched


def _match_by_reference(claim: Claim, figures: list[dict]) -> list[dict]:
    """Figures matched by an explicit 'Figure N' / 'Table N' reference."""
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

def _split_blocks(markdown: str) -> list[str]:
    """Split the paper into heading-delimited blocks."""
    blocks: list[str] = []
    current: list[str] = []
    for line in markdown.splitlines():
        if line.lstrip().startswith("#") and current:
            blocks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [b for b in blocks if b]


def _score_block(block: str, claim: Claim, refs: list[str]) -> int:
    """How relevant a block is to auditing this claim. Higher is better."""
    lowered = block.lower()
    score = 0

    # The block quoting the claim is the single most useful piece of evidence.
    snippet = " ".join(claim.raw_text.lower().split())[:90]
    if snippet and snippet in " ".join(lowered.split()):
        score += 100

    # Blocks holding a figure/table the claim cites.
    for ref in refs:
        kind, _, number = ref.partition(":")
        if re.search(rf"\b{kind}s?\.?\s*{re.escape(number)}\b", lowered):
            score += 40

    # The section the claim says it came from.
    section = (claim.source_section or "").lower()
    if section and section not in ("unspecified", "n/a"):
        for token in re.findall(r"[0-9]+(?:\.[0-9]+)*|[a-z]{4,}", section):
            if token in lowered:
                score += 8

    # Tables are where reported numbers live.
    if block.count("|") > 6:
        score += 15

    if claim.reported_value is not None and str(claim.reported_value).lower() in lowered:
        score += 30
    if claim.metric and claim.metric.lower() in lowered:
        score += 10
    if claim.baseline and claim.baseline.lower() in lowered:
        score += 10

    # Statistical reporting is one of the five checks the auditor must make.
    if re.search(r"\bp\s*[<=>]\s*0?\.\d|confidence interval|std|significan", lowered):
        score += 6

    return score


def build_audit_excerpt(claim: Claim, markdown: str, max_chars: int = AUDIT_CONTEXT_CHARS) -> str:
    """
    Assemble the slice of the paper needed to audit one claim.

    Sending the whole paper per claim is both slow and impossible on Groq's free
    tier, where any single request above ~8000 tokens is rejected outright. The
    auditor only needs the claim's own section, the tables and figures it cites,
    and the paper's opening for context — so select those by relevance.
    """
    if not markdown:
        return ""
    if len(markdown) <= max_chars:
        return markdown

    refs = _claim_figure_refs(claim)
    blocks = _split_blocks(markdown)
    if not blocks:
        return markdown[:max_chars]

    ranked = sorted(
        ((_score_block(b, claim, refs), i, b) for i, b in enumerate(blocks)),
        key=lambda t: (-t[0], t[1]),
    )

    # Always lead with the opening block (title/abstract) for context.
    chosen: dict[int, str] = {0: blocks[0]}
    used = len(blocks[0])

    for score, index, block in ranked:
        if score <= 0 or index in chosen:
            continue
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 600:      # a fragment this size is still informative
                chosen[index] = block[:remaining]
                used = max_chars
            continue
        chosen[index] = block
        used += len(block)

    # Restore document order so the auditor reads a coherent excerpt.
    ordered = [chosen[i] for i in sorted(chosen)]
    return "\n\n[...]\n\n".join(ordered)


def _audit_claim_text(claim: Claim, markdown: str) -> dict:
    """Audit a claim against the paper text."""
    excerpt = build_audit_excerpt(claim, markdown, AUDIT_CONTEXT_CHARS)
    return chat_json(
        "internal_auditor",
        AUDIT_SYSTEM_PROMPT,
        (
            f"CLAIM TO AUDIT:\n{claim.model_dump_json(indent=2)}\n\n"
            f"RELEVANT PAPER EXCERPTS (non-contiguous sections are separated by [...]):\n"
            f"{excerpt}"
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


def _build_audit(raw: dict, claim_id: str, vlm_reading: str | None = None) -> InternalAuditResult:
    """
    Coerce loose audit JSON into a valid InternalAuditResult.

    ``visual_mismatch_detail`` comes only from ``vlm_reading``, never from the
    text auditor's JSON: that model sees captions, not images, and in a real run
    it filled the field with a guess the UI then presented as the VLM's reading.
    """
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
        visual_mismatch_detail=vlm_reading or None,
        baseline_present=as_bool(raw.get("baseline_present"), default=False),
        statistical_rigor=severity("statistical_rigor"),
        statistical_detail=detail("statistical_detail"),
        methodology_gap=detail("methodology_gap"),
        mismatch_score=clamp01(raw.get("mismatch_score"), default=0.5),
    )


def _is_table_line(line: str) -> bool:
    return line.count("|") >= 2


def stated_only_in_text(claim: Claim, markdown: str) -> bool:
    """
    True when the claim's numbers appear in the paper's prose and no table
    covers the claim.

    Many papers report results only in the text. In a real run the auditor
    marked such claims FAIL because "no table lists these numbers", and the
    judge turned that into TABLE_MISMATCH. A table is taken to cover the claim
    if it holds one of the claim's numbers, or names the claim's method or
    baseline alongside a decimal value, in which case a real mismatch is
    possible and the audit is left alone.
    """
    numbers = _claim_numbers(claim)
    if not numbers or not markdown:
        return False

    lines = markdown.splitlines()
    table = [line.lower() for line in lines if _is_table_line(line)]
    prose = " ".join(line for line in lines if not _is_table_line(line))

    if not any(n in prose for n in numbers):
        return False
    if any(n in line for n in numbers for line in table):
        return False

    entities = _terms(f"{claim.subject} {claim.baseline or ''}")
    for line in table:
        if re.search(r"\d\.\d", line) and entities & _terms(line):
            return False
    return True


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

        try:
            audit = _build_audit(audit_raw, claim.claim_id, "\n\n".join(visual_notes) or None)
        except Exception as exc:
            log.error("could not build audit result for %s: %s", claim.claim_id, exc)
            results.append(_degraded_audit(claim.claim_id, str(exc)[:200]))
            continue

        if (audit.table_consistency in (AuditSeverity.FAIL, AuditSeverity.MISMATCH)
                and stated_only_in_text(claim, markdown)):
            log.info("%s: value is reported only in the text; table check downgraded to WARN",
                     claim.claim_id)
            audit.table_consistency = AuditSeverity.WARN
            audit.table_detail = (
                "Reported in the text; no table in the paper reports this value, so "
                "there is nothing to cross-check. " + (audit.table_detail or "")
            ).strip()
        results.append(audit)

    log.info("audited %d/%d claims", len(results), len(claims))
    return {"internal_audits": results}
