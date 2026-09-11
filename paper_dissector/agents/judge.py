"""Stage 6: Adjudication — Judge reads debates and issues verdicts."""

from __future__ import annotations

import logging

from paper_dissector.llm import chat_json
from paper_dissector.sanitize import as_str_list, as_text, clamp01, coerce_enum
from paper_dissector.schemas import (
    Claim, ClaimVerdict, DebateTranscript, ExternalEvidenceResult,
    FinalReport, InternalAuditResult, VerdictLabel,
)
from paper_dissector.state import PaperState

log = logging.getLogger(__name__)

# Credibility bands, highest first. A verdict label maps to a score, and a
# score maps back to a label, so the paper-level number and the per-claim
# labels can never disagree.
_BANDS: list[tuple[float, VerdictLabel]] = [
    (0.85, VerdictLabel.STRONGLY_SUPPORTED),
    (0.65, VerdictLabel.SUPPORTED),
    (0.40, VerdictLabel.PARTIALLY_SUPPORTED),
    (0.20, VerdictLabel.WEAKLY_SUPPORTED),
]

# Representative credibility for each label — the middle of its band.
_CREDIBILITY: dict[VerdictLabel, float] = {
    VerdictLabel.STRONGLY_SUPPORTED: 0.93,
    VerdictLabel.SUPPORTED: 0.75,
    VerdictLabel.PARTIALLY_SUPPORTED: 0.52,
    VerdictLabel.WEAKLY_SUPPORTED: 0.30,
    VerdictLabel.NOT_SUPPORTED: 0.10,
}


def credibility_for(label: VerdictLabel) -> float:
    """Credibility score implied by a verdict label."""
    return _CREDIBILITY.get(label, 0.5)

JUDGE_SYSTEM = """You are the JUDGE in a scientific claim credibility analysis.
You have read the full adversarial debate between a Prosecutor (arguing the claim
is not credible) and a Defender (arguing it is credible).

You also have access to the internal methodology audit and external evidence.

YOUR JOB:
1. Weigh the arguments from both sides based on EVIDENCE QUALITY, not rhetoric.
2. Apply this scoring rubric:
   - Internal data consistency: 25%
   - Visual-textual alignment: 15%
   - External literature support: 20%
   - Baseline recency/fairness: 15%
   - Debate argument quality: 15%
   - Statistical rigor: 10%

3. Issue a verdict — this carries how well supported the claim is:
   - STRONGLY_SUPPORTED  the evidence firmly establishes the claim
   - SUPPORTED           the claim holds, with minor gaps
   - PARTIALLY_SUPPORTED parts hold, parts do not
   - WEAKLY_SUPPORTED    little more than suggestive
   - NOT_SUPPORTED       the evidence does not establish the claim

4. Report "confidence" separately: how certain YOU are of that verdict, from
   0.0 to 1.0. This is NOT how good the claim is. If the evidence clearly
   refutes the claim, the correct answer is verdict NOT_SUPPORTED with a HIGH
   confidence, because you are certain of the refutation.

Respond ONLY with valid JSON:
{
  "claim_id": "C1",
  "verdict": "PARTIALLY_SUPPORTED",
  "confidence": 0.55,
  "justification": "2-3 sentence explanation",
  "flags": ["STALE_BASELINE", "VISUAL_MISMATCH"],
  "prosecutor_strongest": "their best point in one line",
  "defender_strongest": "their best point in one line",
  "unresolved": ["list of points neither side settled"]
}"""


def _label_for_score(score: float) -> VerdictLabel:
    """Map a confidence score onto its verdict band."""
    for threshold, label in _BANDS:
        if score >= threshold:
            return label
    return VerdictLabel.NOT_SUPPORTED


def _build_judge_context(
    claim: Claim,
    audit: InternalAuditResult | None,
    evidence: ExternalEvidenceResult | None,
    transcript: DebateTranscript | None,
) -> str:
    """Compile everything the Judge needs to see."""
    parts = [f"CLAIM: {claim.raw_text}\n"]

    if audit:
        parts.append(f"INTERNAL AUDIT:\n{audit.model_dump_json(indent=2)}\n")

    if evidence:
        parts.append(f"EXTERNAL EVIDENCE:\n{evidence.model_dump_json(indent=2)}\n")

    if transcript and transcript.turns:
        parts.append(f"DEBATE TRANSCRIPT ({transcript.total_rounds} rounds, ended: {transcript.terminated_reason}):")
        for t in transcript.turns:
            label = t.agent.value.upper()
            parts.append(f"\n[{label} — Round {t.round_num}]")
            parts.append(t.argument)
            if t.new_retrieval:
                parts.append(f"  (Retrieved mid-debate: {t.new_retrieval.get('query', '')})")
            if t.concedes:
                parts.append(f"  ** {label} CONCEDED **")
    else:
        parts.append("DEBATE TRANSCRIPT: unavailable — judge on the audit and evidence alone.")

    return "\n".join(parts)


def _build_verdict(raw: dict, claim_id: str) -> ClaimVerdict:
    """Coerce loose judge JSON into a valid ClaimVerdict."""
    confidence = clamp01(raw.get("confidence"), default=0.5)
    verdict = coerce_enum(raw.get("verdict"), VerdictLabel, _label_for_score(confidence))

    return ClaimVerdict(
        claim_id=claim_id,
        verdict=verdict,
        confidence=confidence,
        credibility=credibility_for(verdict),
        justification=as_text(raw.get("justification"), default="No justification provided."),
        flags=[f.upper().replace(" ", "_") for f in as_str_list(raw.get("flags"))],
        prosecutor_strongest=as_text(raw.get("prosecutor_strongest")),
        defender_strongest=as_text(raw.get("defender_strongest")),
        unresolved=as_str_list(raw.get("unresolved")),
    )


def _degraded_verdict(claim_id: str, reason: str) -> ClaimVerdict:
    """Placeholder verdict so one failed adjudication doesn't drop a claim."""
    return ClaimVerdict(
        claim_id=claim_id,
        verdict=VerdictLabel.PARTIALLY_SUPPORTED,
        confidence=0.0,   # we are not confident of anything here
        credibility=credibility_for(VerdictLabel.PARTIALLY_SUPPORTED),
        justification=f"Adjudication unavailable for this claim: {reason}",
        flags=["ADJUDICATION_FAILED"],
    )


def adjudicate(state: PaperState) -> dict:
    """LangGraph node: Judge issues verdicts for all claims."""
    audits = {a.claim_id: a for a in state.get("internal_audits", [])}
    evidence = {e.claim_id: e for e in state.get("external_evidence", [])}
    transcripts = {t.claim_id: t for t in state.get("debate_transcripts", [])}
    claims = state.get("claims") or []

    verdicts: list[ClaimVerdict] = []
    for claim in claims:
        context = _build_judge_context(
            claim,
            audits.get(claim.claim_id),
            evidence.get(claim.claim_id),
            transcripts.get(claim.claim_id),
        )

        try:
            parsed = chat_json("judge", JUDGE_SYSTEM, context, temperature=0.1)
            verdicts.append(_build_verdict(parsed, claim.claim_id))
        except Exception as exc:
            log.error("adjudication failed for %s: %s", claim.claim_id, exc)
            verdicts.append(_degraded_verdict(claim.claim_id, str(exc)[:200]))

    # ── Compile final report ──
    # Aggregate credibility. Averaging confidence would mean a paper whose
    # claims were all confidently rejected scored highly.
    scores = [v.credibility for v in verdicts]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    overall = _label_for_score(avg_score)

    # Collect systemic issues
    systemic = []
    stale_count = sum(1 for v in verdicts if "STALE_BASELINE" in v.flags)
    visual_count = sum(1 for v in verdicts if "VISUAL_MISMATCH" in v.flags)
    failed_count = sum(1 for v in verdicts if "ADJUDICATION_FAILED" in v.flags)
    if stale_count > 1:
        systemic.append(f"Paper relies on outdated baselines ({stale_count} claims affected)")
    if visual_count > 0:
        systemic.append(f"{visual_count} figure(s) do not support narrative claims")
    if failed_count:
        systemic.append(
            f"{failed_count} claim(s) could not be adjudicated — treat the overall score as partial"
        )

    report = FinalReport(
        paper_title=state.get("paper_title") or "Unknown",
        authors=state.get("paper_authors") or [],
        overall_score=round(avg_score, 3),
        overall_verdict=overall,
        total_claims=len(verdicts),
        claim_verdicts=verdicts,
        systemic_issues=systemic,
    )

    log.info("adjudicated %d claims; overall credibility %.3f (%s)",
             len(verdicts), avg_score, overall.value)
    return {"verdicts": verdicts, "final_report": report}
