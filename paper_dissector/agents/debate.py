"""Stage 5: Adversarial debate — Prosecutor vs Defender with progressive RAG."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from difflib import SequenceMatcher

from paper_dissector.config import (
    CONVERGENCE_THRESHOLD, MAX_DEBATE_ROUNDS, MAX_PRAG_RETRIEVALS,
)
from paper_dissector.llm import chat_text
from paper_dissector.schemas import (
    Claim, DebateRole, DebateTranscript, DebateTurn,
    ExternalEvidenceResult, InternalAuditResult,
)
from paper_dissector.state import PaperState
from paper_dissector.tools.literature import search_papers

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SEARCH_RE = re.compile(r"SEARCH_REQUEST:\s*(.+)")

PROSECUTOR_SYSTEM = """You are the PROSECUTOR in a scientific claim credibility debate.
Your goal is to argue that the claim is NOT credible or is overstated.

You have access to:
- Internal audit results (methodology problems, data mismatches, figure issues)
- External contradicting papers
- Baseline staleness analysis

RULES:
- Every argument MUST cite specific evidence (audit field, paper DOI, staleness entry).
- Be precise and technical. No vague attacks.
- If the evidence you were given is thin, or you need a specific number or a
  stronger baseline to make your case, GO AND FIND IT. Put this on its own line:
  SEARCH_REQUEST: "your query here"
  You may do this up to {max_prag} times. Using it is expected, not exceptional:
  an argument backed by a result you retrieved is far stronger than one that
  only reasons about the evidence already in front of you.
- You can CONCEDE a point if the defender's rebuttal is genuinely strong.
  Format: CONCEDE: "what you're conceding and why"
- Stay focused on the specific claim, not the paper in general.
- Do not repeat an argument you have already made. Advance the debate or concede.

Respond with your argument. Be concise but devastating.""".format(max_prag=MAX_PRAG_RETRIEVALS)

DEFENDER_SYSTEM = """You are the DEFENDER in a scientific claim credibility debate.
Your goal is to argue that the claim IS credible and well-supported.

You have access to:
- The internal audit of the paper's own data. This is your strongest material:
  a table_consistency of PASS means the claim's numbers match the paper's own
  tables, which is direct evidence the claim holds.
- Any external supporting papers, plus the paper's methodology and context.

Note: retrieved abstracts rarely restate another paper's exact numbers, so an
empty supporting-papers list does not mean the claim lacks corroboration. Say
so if the prosecutor argues from that absence.

RULES:
- Directly address each of the prosecutor's points. Don't ignore attacks.
- Cite specific supporting papers or methodological justifications.
- If you lack corroborating evidence for a point under attack, GO AND FIND IT.
  Put this on its own line:
  SEARCH_REQUEST: "your query here"
  You may do this up to {max_prag} times. Using it is expected, not exceptional:
  independent corroboration you retrieved is the strongest defence available.
- You can CONCEDE a point if the prosecutor's evidence is genuinely strong.
  Format: CONCEDE: "what you're conceding and why"
- Acknowledge limitations honestly — partial concessions build credibility.
- Do not repeat an argument you have already made. Advance the debate or concede.

Respond with your rebuttal. Be precise and evidence-based.""".format(max_prag=MAX_PRAG_RETRIEVALS)


def _build_context(
    claim: Claim,
    audit: InternalAuditResult | None,
    evidence: ExternalEvidenceResult | None,
    prev_turns: list[DebateTurn],
) -> str:
    """Build the debate context string from all available evidence."""
    parts = [f"CLAIM UNDER DEBATE:\n{claim.raw_text}\n"]

    if audit:
        parts.append(f"INTERNAL AUDIT:\n{audit.model_dump_json(indent=2)}\n")

    if evidence:
        if evidence.supporting_papers:
            parts.append("SUPPORTING PAPERS:")
            for p in evidence.supporting_papers[:3]:
                parts.append(f"  - {p.title} ({p.year}) [confidence: {p.stance_confidence}]")
        if evidence.contradicting_papers:
            parts.append("CONTRADICTING PAPERS:")
            for p in evidence.contradicting_papers[:3]:
                parts.append(f"  - {p.title} ({p.year}) [confidence: {p.stance_confidence}]")
        if evidence.staleness_entries:
            parts.append("STALENESS ANALYSIS:")
            for s in evidence.staleness_entries:
                parts.append(f"  - {s.baseline_name}: {s.verdict}")

    if prev_turns:
        parts.append("\nDEBATE SO FAR:")
        for t in prev_turns:
            label = "PROSECUTOR" if t.agent == DebateRole.PROSECUTOR else "DEFENDER"
            parts.append(f"  [{label} Round {t.round_num}]: {t.argument[:500]}")

    return "\n".join(parts)


# ── Convergence detection ────────────────────────────────────────

# Two arguments about the same claim inevitably share the claim's vocabulary and
# a lot of function words, which inflates similarity. Comparing content words
# only keeps genuine repetition high while letting distinct arguments separate.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those there here is are was were
be been being am do does did doing have has had having will would shall should
can could may might must of in on at to for with without from by as into over
under about against between during before after above below up down out off
again further once it its it s he she they them his her their we us our you your
i me my not no nor only own same so too very s t don now also however moreover
which who whom what when where why how all any both each few more most other some
such more paper claim authors argument point evidence
""".split())


def _tokens(text: str) -> Counter:
    return Counter(
        w for w in _TOKEN_RE.findall((text or "").lower())
        if w not in _STOPWORDS and len(w) > 2
    )


def _cosine(a: Counter, b: Counter) -> float:
    """Cosine similarity between two bag-of-words vectors."""
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in shared)
    if not dot:
        return 0.0
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (norm_a * norm_b)


def argument_similarity(first: str, second: str) -> float:
    """
    Similarity between two arguments, used to detect a debate going in circles.

    Combines bag-of-words cosine (catches reordered restatements) with a sequence
    ratio (catches near-verbatim repetition) and takes the stronger signal. This is
    a deliberately dependency-free stand-in for sentence embeddings.
    """
    if not first or not second:
        return 0.0
    cosine = _cosine(_tokens(first), _tokens(second))
    sequence = SequenceMatcher(None, first.lower(), second.lower()).ratio()
    return max(cosine, sequence)


def _has_converged(turns: list[DebateTurn], role: DebateRole) -> bool:
    """True when this role's two most recent arguments say the same thing."""
    same_role = [t.argument for t in turns if t.agent == role]
    if len(same_role) < 2:
        return False
    score = argument_similarity(same_role[-1], same_role[-2])
    if score > CONVERGENCE_THRESHOLD:
        log.info("%s converged (similarity %.3f > %.2f)", role.value, score, CONVERGENCE_THRESHOLD)
        return True
    return False


# ── Progressive RAG ──────────────────────────────────────────────

# Where a model stops stating its query and starts writing something else.
# Observed in real runs: an HTML line break, and the model inventing its own
# "RESULT:" section complete with fabricated citations.
_QUERY_TERMINATORS = re.compile(
    r"(?:<br|&lt;|&gt;|\bRESULT\s*:|\bRESULTS\s*:|\bANSWER\s*:|[}\]]|\n)",
    re.IGNORECASE,
)

MAX_QUERY_CHARS = 120


def _extract_search_query(response_text: str) -> str | None:
    """
    Pull a SEARCH_REQUEST query out of an agent's response.

    Models embed the marker in prose and markdown, so the captured text arrives
    decorated. Three shapes seen in real runs:
        '** "WMT14 English German bootstrap significance'
        '“BLEU variance across seeds” &lt;br RESULT: Ott et al., ...'
        'transformer ablation study.'
    All of it would otherwise be sent to the search backend verbatim.
    """
    match = _SEARCH_RE.search(response_text or "")
    if not match:
        return None

    query = match.group(1)

    # Stop at the point the model stopped stating a query.
    cut = _QUERY_TERMINATORS.search(query)
    if cut:
        query = query[: cut.start()]

    query = re.sub(r"<[^>]+>", " ", query)               # stray html tags
    query = re.sub(r"&[a-z]+;", " ", query)              # html entities
    query = query.translate(str.maketrans({             # smart punctuation
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "‑": "-", "–": "-", "—": "-", " ": " ",
    }))
    query = re.sub(r"[*_`#>]+", " ", query)              # markdown emphasis
    query = re.sub(r"[\"']", " ", query)                 # any remaining quotes
    query = re.sub(r"\s+", " ", query).strip(" .:;,-")

    if not query:
        return None

    if len(query) > MAX_QUERY_CHARS:
        # Trim to a word boundary rather than mid-term.
        query = query[:MAX_QUERY_CHARS].rsplit(" ", 1)[0]
        log.debug("truncated an over-long search request")

    return query or None


def _format_results(results: list[dict]) -> str:
    if not results:
        return "No results found for that query."
    return "\n".join(
        f"- {r.get('title', '?')} ({r.get('year', '?')}): {(r.get('abstract') or '')[:300]}"
        for r in results
    )


def _handle_progressive_rag(
    agent_name: str,
    system_prompt: str,
    context: str,
    response_text: str,
    prag_budget: int,
) -> tuple[str, dict | None, int]:
    """
    Execute a mid-debate search the agent asked for and let it revise its argument.

    The scaffold recorded the retrieval but never showed it to the agent; here the
    results are fed back so the follow-up argument can actually use them.
    """
    if prag_budget <= 0:
        return response_text, None, prag_budget

    query = _extract_search_query(response_text)
    if not query:
        return response_text, None, prag_budget

    try:
        results = search_papers(query, limit=3)
    except Exception as exc:
        log.warning("progressive RAG search failed for %r: %s", query, exc)
        return response_text, None, prag_budget - 1

    log.info("progressive RAG: %s searched %r mid-debate, %d result(s)",
             agent_name, query, len(results))

    retrieval_info = {
        "query": query,
        "results": [
            {"title": p.get("title"), "year": p.get("year"), "abstract": (p.get("abstract") or "")[:200]}
            for p in results
        ],
    }

    try:
        revised = chat_text(
            agent_name,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": (
                    f"Results for your search {query!r}:\n{_format_results(results)}\n\n"
                    "Now give your final argument for this round, incorporating anything "
                    "useful from these results. Do not issue another SEARCH_REQUEST."
                )},
            ],
            temperature=0.4,
        )
        if revised.strip():
            response_text = revised
    except Exception as exc:
        log.warning("progressive RAG follow-up failed: %s", exc)

    return response_text, retrieval_info, prag_budget - 1


def _check_concession(response_text: str) -> bool:
    """Check if agent explicitly conceded."""
    return "CONCEDE:" in (response_text or "")


def _cited_evidence(response_text: str) -> list[str]:
    """Extract DOIs and audit field names the argument explicitly cites."""
    text = response_text or ""
    cited = re.findall(r"10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+", text)
    for field in (
        "table_consistency", "figure_consistency", "statistical_rigor",
        "methodology_gap", "baseline_present", "mismatch_score",
        "visual_mismatch_detail",
    ):
        if field in text:
            cited.append(field)
    seen = set()
    return [c for c in cited if not (c in seen or seen.add(c))]


# ── Debate driver ────────────────────────────────────────────────

def _run_one_debate(
    claim: Claim,
    audit: InternalAuditResult | None,
    ext: ExternalEvidenceResult | None,
) -> DebateTranscript:
    """Run the full prosecutor/defender exchange for a single claim."""
    turns: list[DebateTurn] = []
    budgets = {"prosecutor": MAX_PRAG_RETRIEVALS, "defender": MAX_PRAG_RETRIEVALS}
    terminated_reason = "max_rounds"

    for round_num in range(1, MAX_DEBATE_ROUNDS + 1):
        for role, agent_name, system_prompt in (
            (DebateRole.PROSECUTOR, "prosecutor", PROSECUTOR_SYSTEM),
            (DebateRole.DEFENDER, "defender", DEFENDER_SYSTEM),
        ):
            context = _build_context(claim, audit, ext, turns)

            try:
                text = chat_text(
                    agent_name,
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context},
                    ],
                    temperature=0.4,
                )
            except Exception as exc:
                log.error("%s turn failed for %s round %d: %s",
                          agent_name, claim.claim_id, round_num, exc)
                return DebateTranscript(
                    claim_id=claim.claim_id,
                    turns=turns,
                    total_rounds=len([t for t in turns if t.agent == DebateRole.PROSECUTOR]),
                    terminated_reason=f"error: {str(exc)[:120]}",
                )

            text, retrieval, budgets[agent_name] = _handle_progressive_rag(
                agent_name, system_prompt, context, text, budgets[agent_name]
            )

            turn = DebateTurn(
                agent=role,
                round_num=round_num,
                argument=text,
                evidence_cited=_cited_evidence(text),
                new_retrieval=retrieval,
                concedes=_check_concession(text),
            )
            turns.append(turn)

            if turn.concedes:
                return DebateTranscript(
                    claim_id=claim.claim_id,
                    turns=turns,
                    total_rounds=len([t for t in turns if t.agent == DebateRole.PROSECUTOR]),
                    terminated_reason=f"{agent_name}_concession",
                )

            if _has_converged(turns, role):
                return DebateTranscript(
                    claim_id=claim.claim_id,
                    turns=turns,
                    total_rounds=len([t for t in turns if t.agent == DebateRole.PROSECUTOR]),
                    terminated_reason="convergence",
                )

    return DebateTranscript(
        claim_id=claim.claim_id,
        turns=turns,
        total_rounds=len([t for t in turns if t.agent == DebateRole.PROSECUTOR]),
        terminated_reason=terminated_reason,
    )


def run_debate(state: PaperState) -> dict:
    """LangGraph node: run structured debate for each claim."""
    transcripts: list[DebateTranscript] = []

    # Pair up claims with their audit + evidence results
    audits = {a.claim_id: a for a in state.get("internal_audits", [])}
    evidence = {e.claim_id: e for e in state.get("external_evidence", [])}
    claims = state.get("claims") or []

    for claim in claims:
        try:
            transcript = _run_one_debate(
                claim, audits.get(claim.claim_id), evidence.get(claim.claim_id)
            )
        except Exception as exc:
            log.error("debate failed for %s: %s", claim.claim_id, exc)
            transcript = DebateTranscript(
                claim_id=claim.claim_id,
                turns=[],
                total_rounds=0,
                terminated_reason=f"error: {str(exc)[:120]}",
            )
        transcripts.append(transcript)
        log.info("%s debate: %d rounds, ended by %s",
                 claim.claim_id, transcript.total_rounds, transcript.terminated_reason)

    return {"debate_transcripts": transcripts}
