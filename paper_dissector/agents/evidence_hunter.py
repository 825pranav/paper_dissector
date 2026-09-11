"""Stage 4: External evidence retrieval + temporal staleness detection."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from paper_dissector.llm import chat_json
from paper_dissector.sanitize import as_text
from paper_dissector.schemas import (
    Claim, ExternalEvidenceResult, RetrievedPaper, Stance, StalenessEntry,
)
from paper_dissector.state import PaperState
from paper_dissector.tools.literature import search_papers, search_sota_for_task
from paper_dissector.tools.stance_classifier import batch_classify

log = logging.getLogger(__name__)

MAX_QUERIES_PER_CLAIM = 3
MAX_PAPERS_PER_CLAIM = 12

QUERY_GEN_PROMPT = """Given a scientific claim, generate 2-3 search queries to find
papers that might SUPPORT or CONTRADICT this claim. Queries should be short (3-8 words),
use technical terms, and target the specific metric/task/method mentioned.

Respond ONLY with JSON: {"queries": ["query1", "query2", "query3"]}"""

STALENESS_PROMPT = """Given a baseline model/method cited in a paper's experiments and
a list of potentially stronger alternatives that existed before the paper was written,
assess whether the baseline choice was stale or cherry-picked.

Respond ONLY with JSON:
{
  "baseline_name": "...",
  "baseline_year": 2019,
  "staleness_score": 5.0,
  "missed_stronger": [{"name": "...", "year": 2023, "reason": "higher cited score on same task"}],
  "verdict": "STALE — N stronger baselines existed at submission time" or "CURRENT"
}"""


def _generate_queries(claim: Claim) -> list[str]:
    """Use LLM to generate search queries for a claim."""
    try:
        parsed = chat_json(
            "evidence_hunter",
            QUERY_GEN_PROMPT,
            f"Claim: {claim.raw_text}\nMetric: {claim.metric}\nScope: {claim.scope}",
            temperature=0.3,
        )
    except Exception as exc:
        log.warning("query generation failed for %s: %s", claim.claim_id, exc)
        parsed = {}

    queries = [as_text(q) for q in (parsed.get("queries") or []) if as_text(q)]
    if not queries:
        # Degrade to a keyword query built from the claim itself.
        fallback = " ".join(
            part for part in (claim.subject, claim.intervention, claim.metric) if part
        ).strip()
        queries = [fallback or claim.raw_text[:120]]
        log.info("using fallback query for %s: %r", claim.claim_id, queries[0])

    return queries[:MAX_QUERIES_PER_CLAIM]


def _dedupe_key(paper: dict) -> str:
    """
    Identity for a retrieved paper, stable across backends.

    A DOI is preferred: OpenAlex in particular stores reindexed duplicates of the
    same work under different IDs but the same DOI, so deduplicating on the
    provider's own id alone lets the same paper through twice.
    """
    doi = (paper.get("externalIds") or {}).get("DOI")
    if doi:
        return f"doi:{str(doi).strip().lower()}"
    title = " ".join((paper.get("title") or "").lower().split())
    if title:
        return f"title:{title}"
    return f"id:{paper.get('paperId', '')}"


def _normalise_title(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def is_same_paper(candidate_title: str, paper_title: str) -> bool:
    """
    True when a retrieved record is the paper being analysed.

    Searching a paper's own claims reliably retrieves that paper, and counting
    it as supporting evidence is circular — it inflates every claim's support
    by exactly one. Matched on normalised title rather than id, because the
    same work appears under different ids across backends.
    """
    a, b = _normalise_title(candidate_title), _normalise_title(paper_title)
    if not a or not b or len(b) < 10:
        return False
    if a == b:
        return True

    # Containment alone is not enough: "Attention Is All You Need In Speech
    # Separation" contains the title of a different paper. Require the two to be
    # nearly the same length as well, so only punctuation or subtitle noise
    # separates them.
    if (a in b or b in a) and min(len(a), len(b)) / max(len(a), len(b)) > 0.85:
        return True

    return SequenceMatcher(None, a, b).ratio() > 0.9


def _retrieve_and_classify(
    claim: Claim, queries: list[str], paper_title: str = "",
) -> tuple[list[RetrievedPaper], list[RetrievedPaper], list[RetrievedPaper]]:
    """Search the literature backend and classify the stance of retrieved papers."""
    supporting, contradicting, neutral = [], [], []
    seen: set[str] = set()
    candidates: list[dict] = []

    for query in queries:
        for p in search_papers(query, limit=5):
            if not p.get("abstract"):
                continue
            if paper_title and is_same_paper(p.get("title") or "", paper_title):
                log.debug("excluding the paper under analysis from its own evidence")
                continue
            key = _dedupe_key(p)
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append(p)
        if len(candidates) >= MAX_PAPERS_PER_CLAIM:
            break

    candidates = candidates[:MAX_PAPERS_PER_CLAIM]
    if not candidates:
        return [], [], []

    # One batched classification call rather than one per paper.
    stances = batch_classify(claim.raw_text, [p["abstract"] for p in candidates])

    for p, (stance, confidence) in zip(candidates, stances):
        try:
            rp = RetrievedPaper(
                paper_id=p.get("paperId", ""),
                title=p.get("title") or "",
                year=p.get("year") or 0,
                authors=[a.get("name", "") for a in (p.get("authors") or [])[:3]],
                doi=(p.get("externalIds") or {}).get("DOI"),
                url=p.get("url"),
                relevant_passage=(p.get("abstract") or "")[:500],
                stance=stance,
                stance_confidence=confidence,
            )
        except Exception as exc:
            log.debug("skipping malformed paper record: %s", exc)
            continue

        if stance == Stance.SUPPORT:
            supporting.append(rp)
        elif stance == Stance.CONTRADICT:
            contradicting.append(rp)
        else:
            neutral.append(rp)

    return supporting, contradicting, neutral


def _staleness_score(value) -> float:
    """Coerce the model's staleness rating onto a 0-10 scale."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score != score:  # NaN
        return 0.0
    if 0.0 <= score <= 1.0:
        # Some responses use a 0-1 scale instead of 0-10.
        score *= 10.0
    return min(max(score, 0.0), 10.0)


def _check_staleness(claim: Claim, paper_year: int | None) -> list[StalenessEntry]:
    """Check if baselines cited in a claim were already outdated at submission time."""
    if not claim.baseline or not paper_year:
        return []

    sota_papers = search_sota_for_task(
        task=claim.scope,
        metric=claim.metric or "",
        before_year=paper_year,
        limit=5,
    )
    if not sota_papers:
        return []

    try:
        parsed = chat_json(
            "staleness_checker",
            STALENESS_PROMPT,
            (
                f"Paper submission year: {paper_year}\n"
                f"Baseline used: {claim.baseline}\n"
                f"Task/scope: {claim.scope}\n"
                f"Metric: {claim.metric}\n\n"
                f"Potentially stronger alternatives found:\n"
                + "\n".join(
                    f"- {p.get('title', '?')} ({p.get('year', '?')}), citations: {p.get('citationCount', '?')}"
                    for p in sota_papers
                )
            ),
            temperature=0.1,
        )
    except Exception as exc:
        log.warning("staleness check failed for %s: %s", claim.claim_id, exc)
        return []

    try:
        year = parsed.get("baseline_year")
        missed = parsed.get("missed_stronger") or []
        return [StalenessEntry(
            baseline_name=as_text(parsed.get("baseline_name"), default=claim.baseline),
            baseline_year=int(year) if isinstance(year, (int, float, str)) and str(year).strip().isdigit() else paper_year,
            staleness_score=_staleness_score(parsed.get("staleness_score")),
            missed_stronger=[m for m in missed if isinstance(m, dict)],
            verdict=as_text(parsed.get("verdict"), default="UNKNOWN"),
        )]
    except Exception as exc:
        log.warning("could not build staleness entry for %s: %s", claim.claim_id, exc)
        return []


def gather_evidence(state: PaperState) -> dict:
    """LangGraph node: retrieve external evidence + staleness analysis for each claim."""
    results: list[ExternalEvidenceResult] = []
    paper_year = state.get("paper_year")
    claims = state.get("claims") or []

    if not paper_year:
        log.info("no publication year available — skipping staleness detection")

    paper_title = state.get("paper_title") or ""

    for claim in claims:
        # A failure on one claim degrades that claim only.
        try:
            queries = _generate_queries(claim)
            supporting, contradicting, neutral = _retrieve_and_classify(
                claim, queries, paper_title,
            )
            staleness = _check_staleness(claim, paper_year)
        except Exception as exc:
            log.error("evidence gathering failed for %s: %s", claim.claim_id, exc)
            supporting, contradicting, neutral, staleness = [], [], [], []

        results.append(ExternalEvidenceResult(
            claim_id=claim.claim_id,
            supporting_papers=supporting,
            contradicting_papers=contradicting,
            neutral_papers=neutral,
            staleness_entries=staleness,
        ))
        log.info(
            "%s: %d supporting, %d contradicting, %d neutral, %d staleness",
            claim.claim_id, len(supporting), len(contradicting), len(neutral), len(staleness),
        )

    return {"external_evidence": results}
