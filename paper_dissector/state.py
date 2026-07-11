from __future__ import annotations
from typing import TypedDict
from paper_dissector.schemas import (
    Claim, InternalAuditResult, ExternalEvidenceResult,
    DebateTranscript, ClaimVerdict, FinalReport,
)


class PaperState(TypedDict):
    """LangGraph state object — flows through every node."""

    # ── Input ────────────────────────────────────────
    pdf_path: str

    # ── Stage 1: Ingestion ───────────────────────────
    parsed_markdown: str
    paper_title: str
    paper_authors: list[str]
    paper_year: int | None
    extracted_figures: list[dict]  # [{figure_id, image_b64, caption, page}]

    # ── Stage 2: Claim Extraction ────────────────────
    claims: list[Claim]

    # ── Stage 3: Internal Audit ──────────────────────
    internal_audits: list[InternalAuditResult]

    # ── Stage 4: External Evidence + Staleness ───────
    external_evidence: list[ExternalEvidenceResult]

    # ── Stage 5: Debate ──────────────────────────────
    debate_transcripts: list[DebateTranscript]
    current_claim_idx: int
    current_round: int

    # ── Stage 6: Adjudication ────────────────────────
    verdicts: list[ClaimVerdict]
    final_report: FinalReport | None
