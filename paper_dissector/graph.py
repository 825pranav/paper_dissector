"""LangGraph state machine — wires all agents into the pipeline."""

from __future__ import annotations

import logging

from langgraph.graph import StateGraph, START, END
from paper_dissector.state import PaperState
from paper_dissector.tools.pdf_parser import parse_pdf
from paper_dissector.agents.claim_extractor import extract_claims
from paper_dissector.agents.internal_auditor import audit_claims
from paper_dissector.agents.evidence_hunter import gather_evidence
from paper_dissector.agents.debate import run_debate
from paper_dissector.agents.judge import adjudicate

log = logging.getLogger(__name__)


# ── Node: Ingest PDF ─────────────────────────────────────────────

def ingest_paper(state: PaperState) -> dict:
    """Parse PDF into structured markdown + extract figures."""
    result = parse_pdf(state["pdf_path"])
    return {
        "parsed_markdown": result["markdown"],
        "paper_title": result["title"],
        "paper_authors": result["authors"],
        "extracted_figures": result["figures"],
        "paper_year": result.get("year"),
    }


# ── Build the graph ──────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Build and compile the Paper Dissector pipeline.

    Flow:
        START → ingest → extract_claims → internal_audit
              → external_evidence → debate → adjudicate → END
    """
    graph = StateGraph(PaperState)

    # Add nodes
    graph.add_node("ingest", ingest_paper)
    graph.add_node("extract_claims", extract_claims)
    graph.add_node("internal_audit", audit_claims)
    graph.add_node("external_evidence", gather_evidence)
    graph.add_node("debate", run_debate)
    graph.add_node("adjudicate", adjudicate)

    # Wire edges (linear for v1 — no conditional branching yet)
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "extract_claims")
    graph.add_edge("extract_claims", "internal_audit")
    graph.add_edge("internal_audit", "external_evidence")
    graph.add_edge("external_evidence", "debate")
    graph.add_edge("debate", "adjudicate")
    graph.add_edge("adjudicate", END)

    return graph.compile()


def run_pipeline(pdf_path: str) -> PaperState:
    """Run the full pipeline on a PDF and return final state."""
    pipeline = build_graph()
    initial_state: PaperState = {
        "pdf_path": pdf_path,
        "parsed_markdown": "",
        "paper_title": "",
        "paper_authors": [],
        "paper_year": None,
        "extracted_figures": [],
        "claims": [],
        "internal_audits": [],
        "external_evidence": [],
        "debate_transcripts": [],
        "current_claim_idx": 0,
        "current_round": 0,
        "verdicts": [],
        "final_report": None,
    }
    return pipeline.invoke(initial_state)
