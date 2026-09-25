"""Stage 6: Adjudication — Judge reads debates and issues verdicts."""

from __future__ import annotations

import logging

from paper_dissector.llm import chat_json
from paper_dissector.sanitize import as_str_list, as_text, clamp01, coerce_enum
from paper_dissector.schemas import (
    AuditSeverity, Claim, ClaimVerdict, DebateTranscript, ExternalEvidenceResult,
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

HOW TO READ THE EVIDENCE — these mistakes invalidate a verdict:

- The paper under analysis is NOT an external source. The claim comes from it.
  A claim does not become false by matching the paper it was extracted from.
- ABSENCE OF SUPPORTING PAPERS IS NOT EVIDENCE AGAINST THE CLAIM. Retrieved
  abstracts almost never restate another paper's exact numbers, so genuine
  corroboration is usually invisible at this stage. "0 supporting papers" means
  the search found nothing either way, NOT that the claim was refuted.
- Papers marked NEUTRAL bear on nothing. Do not read them as doubt.
- The internal audit is the strongest evidence available, because it checks the
  claim against the paper's OWN data. If table_consistency is PASS, the claim's
  numbers match the paper's own reported data (its tables, or its text where no
  table reports them), and that is substantial support. Do not describe a
  number as appearing in a table unless the audit says so.
- visual_mismatch_detail is the vision model's reading of a figure. If it is
  null, no figure was examined, so do not raise VISUAL_MISMATCH or any other
  figure-based flag.
- Raise STALE_BASELINE only if a staleness entry's verdict is STALE, never when
  it says CURRENT. A number reported in the text but in no table is not a
  table mismatch.
- Cite only numbers that appear in the material you were given. Do not state a
  figure for a baseline unless it is in the audit, the evidence or the debate.
  An argument resting on a number you supplied yourself is worthless.

YOUR JOB:
1. Weigh the arguments from both sides based on EVIDENCE QUALITY, not rhetoric.
   A debater's confidence is not evidence, and a concession extracted by a
   forceful opponent is not proof.
2. Apply this scoring rubric:
   - Internal data consistency: 25%   (the audit's table/figure checks)
   - Visual-textual alignment: 15%
   - External literature support: 20%  (score this NEUTRAL, i.e. do not move
     the verdict either way, when no relevant external evidence was found)
   - Baseline recency/fairness: 15%
   - Debate argument quality: 15%
   - Statistical rigor: 10%

   Missing statistical tests and thin external evidence are weaknesses worth
   flagging, not refutations. Reserve NOT_SUPPORTED for a claim actually
   contradicted by the paper's own data or by a specific external result.

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
  "flags": ["NO_STATISTICAL_TESTS"],
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


def _has_vlm_reading(audit: InternalAuditResult | None) -> bool:
    return bool(audit and audit.visual_mismatch_detail)


def _is_visual_flag(flag: str) -> bool:
    return "VISUAL" in flag or "FIGURE" in flag


def ground_flags(
    flags: list[str],
    audit: InternalAuditResult | None,
    evidence: ExternalEvidenceResult | None,
) -> list[str]:
    """
    Keep only the judge's flags that the stage outputs actually back.

    The judge is an LLM and raises flags from its reading of the debate, which
    in real runs included VISUAL_MISMATCH for claims whose figure was never
    looked at. Each rule here ties a flag to the stage output it asserts.
    """
    stale = any(
        (s.verdict or "").strip().upper().startswith("STALE")
        for s in (evidence.staleness_entries if evidence else [])
    )
    table_failed = bool(audit) and audit.table_consistency in (
        AuditSeverity.FAIL, AuditSeverity.MISMATCH,
    )
    contradicted = bool(evidence and evidence.contradicting_papers)

    kept: list[str] = []
    for flag in flags:
        reason = None
        if _is_visual_flag(flag) and not _has_vlm_reading(audit):
            reason = "no VLM reading exists for this claim"
        elif "STALE" in flag and not stale:
            reason = "the baseline check did not find the baseline stale"
        elif "TABLE" in flag and ("MISMATCH" in flag or "INCONSIST" in flag) and not table_failed:
            # Includes a number stated in the text but absent from any table:
            # the audit downgrades that to WARN, since no table contradicts it.
            reason = "the audit's table check did not fail"
        elif "EXTERNAL" in flag and "CONTRADICT" in flag and not contradicted:
            reason = "no retrieved paper contradicts the claim"
        if reason:
            log.info("dropping judge flag %s on %s: %s",
                     flag, audit.claim_id if audit else "?", reason)
            continue
        kept.append(flag)
    return kept


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


def compile_report(state: PaperState, verdicts: list[ClaimVerdict]) -> FinalReport:
    """
    Build the paper-level report from per-claim verdicts.

    Separate from adjudicate() so a saved analysis can be re-scored, or
    partially re-adjudicated, without replaying the whole pipeline.
    """
    # Aggregate credibility. Averaging confidence would mean a paper whose
    # claims were all confidently rejected scored highly.
    scores = [v.credibility for v in verdicts]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    systemic = []
    stale_count = sum(1 for v in verdicts if "STALE_BASELINE" in v.flags)
    # Only a claim whose figure the VLM actually read can count as a figure
    # that fails to support the text.
    audits = {a.claim_id: a for a in state.get("internal_audits") or []}
    visual_count = sum(
        1 for v in verdicts
        if "VISUAL_MISMATCH" in v.flags and _has_vlm_reading(audits.get(v.claim_id))
    )
    failed_count = sum(1 for v in verdicts if "ADJUDICATION_FAILED" in v.flags)
    if stale_count > 1:
        systemic.append(f"Paper relies on outdated baselines ({stale_count} claims affected)")
    if visual_count > 0:
        systemic.append(f"{visual_count} figure(s) do not support narrative claims")
    if failed_count:
        systemic.append(
            f"{failed_count} claim(s) could not be adjudicated — treat the overall score as partial"
        )

    return FinalReport(
        paper_title=state.get("paper_title") or "Unknown",
        authors=state.get("paper_authors") or [],
        overall_score=round(avg_score, 3),
        overall_verdict=_label_for_score(avg_score),
        total_claims=len(verdicts),
        claim_verdicts=verdicts,
        systemic_issues=systemic,
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
            verdict = _build_verdict(parsed, claim.claim_id)
            verdict.flags = ground_flags(
                verdict.flags, audits.get(claim.claim_id), evidence.get(claim.claim_id),
            )
            verdicts.append(verdict)
        except Exception as exc:
            log.error("adjudication failed for %s: %s", claim.claim_id, exc)
            verdicts.append(_degraded_verdict(claim.claim_id, str(exc)[:200]))

    report = compile_report(state, verdicts)
    log.info("adjudicated %d claims; overall credibility %.3f (%s)",
             len(verdicts), report.overall_score, report.overall_verdict.value)
    return {"verdicts": verdicts, "final_report": report}
