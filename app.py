"""Streamlit UI for Paper Dissector."""

import logging
import tempfile

import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

st.set_page_config(page_title="Paper Dissector", page_icon="🔬", layout="wide")

# ── Header ───────────────────────────────────────────────────────

st.title("🔬 Paper Dissector")
st.caption("Multi-agent adversarial credibility analysis for research papers")

# Node name → human label, in pipeline order.
STAGES = {
    "ingest": "1️⃣ Parse PDF",
    "extract_claims": "2️⃣ Extract Claims",
    "internal_audit": "3️⃣ Internal Audit",
    "external_evidence": "4️⃣ External Evidence",
    "debate": "5️⃣ Adversarial Debate",
    "adjudicate": "6️⃣ Judge Verdicts",
}

SEVERITY_ICON = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "MISMATCH": "🔀"}

# ── Sidebar: Upload ──────────────────────────────────────────────

with st.sidebar:
    st.header("Upload Paper")
    uploaded = st.file_uploader("Drop a PDF", type=["pdf"])
    run_btn = st.button("🚀 Analyze", type="primary", disabled=not uploaded)

    st.divider()
    st.markdown("**Pipeline stages:**")
    for label in STAGES.values():
        st.markdown(f"- {label}")


def _stage_summary(node: str, update: dict) -> str:
    """One-line result description for a completed pipeline stage."""
    if node == "ingest":
        figures = update.get("extracted_figures") or []
        year = update.get("paper_year")
        return (
            f"{len(update.get('parsed_markdown') or '')} chars parsed, "
            f"{len(figures)} figure(s), year: {year if year else 'unknown'}"
        )
    if node == "extract_claims":
        return f"{len(update.get('claims') or [])} claim(s) extracted"
    if node == "internal_audit":
        return f"{len(update.get('internal_audits') or [])} claim(s) audited"
    if node == "external_evidence":
        evidence = update.get("external_evidence") or []
        papers = sum(
            len(e.supporting_papers) + len(e.contradicting_papers) + len(e.neutral_papers)
            for e in evidence
        )
        stale = sum(len(e.staleness_entries) for e in evidence)
        return f"{papers} paper(s) retrieved, {stale} staleness finding(s)"
    if node == "debate":
        transcripts = update.get("debate_transcripts") or []
        rounds = sum(t.total_rounds for t in transcripts)
        return f"{len(transcripts)} debate(s), {rounds} total round(s)"
    if node == "adjudicate":
        return f"{len(update.get('verdicts') or [])} verdict(s) issued"
    return "done"


# ── Main content ─────────────────────────────────────────────────

if run_btn and uploaded:
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(uploaded.read())
        pdf_path = f.name

    # Import here to avoid slow load on page refresh
    from paper_dissector.graph import build_graph
    from paper_dissector.state import PaperState

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

    # Stream the graph so each stage reports as it finishes, instead of one
    # opaque spinner for the whole 2-5 minute run.
    result = dict(initial_state)
    failed = None
    with st.status("Running analysis pipeline…", expanded=True) as status:
        try:
            for chunk in pipeline.stream(initial_state, stream_mode="updates"):
                for node, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    result.update(update)
                    st.write(f"{STAGES.get(node, node)} — {_stage_summary(node, update)}")
            status.update(label="Analysis complete", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001 - surface any pipeline failure in the UI
            failed = exc
            status.update(label="Pipeline failed", state="error")

    if failed is not None:
        st.error(f"Pipeline failed: {failed}")
        st.exception(failed)
        st.stop()

    report = result.get("final_report")
    if not report:
        st.error("Pipeline failed to produce a report.")
        st.stop()

    # ── Executive Summary ────────────────────────────────────────

    st.header(f"📄 {report.paper_title}")
    if report.authors:
        st.caption(", ".join(report.authors))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Score", f"{report.overall_score:.2f}")
    col2.metric("Verdict", report.overall_verdict.value.replace("_", " ").title())
    col3.metric("Claims Analyzed", report.total_claims)

    strong = sum(1 for v in report.claim_verdicts if v.confidence >= 0.65)
    col4.metric("Strong Claims", f"{strong}/{report.total_claims}")

    if report.systemic_issues:
        st.warning("**Systemic Issues Found:**\n" + "\n".join(f"- {i}" for i in report.systemic_issues))

    st.divider()

    # ── Per-Claim Breakdown ──────────────────────────────────────

    st.header("Per-Claim Analysis")

    audits = {a.claim_id: a for a in result.get("internal_audits", [])}
    evidence = {e.claim_id: e for e in result.get("external_evidence", [])}
    transcripts = {t.claim_id: t for t in result.get("debate_transcripts", [])}

    for v in report.claim_verdicts:
        claim = next((c for c in result["claims"] if c.claim_id == v.claim_id), None)
        if not claim:
            continue

        # Color code by verdict
        color_map = {
            "STRONGLY_SUPPORTED": "🟢",
            "SUPPORTED": "🟡",
            "PARTIALLY_SUPPORTED": "🟠",
            "WEAKLY_SUPPORTED": "🔴",
            "NOT_SUPPORTED": "⛔",
        }
        icon = color_map.get(v.verdict.value, "⚪")

        with st.expander(f"{icon} [{v.claim_id}] {claim.raw_text[:100]}... — **{v.confidence:.2f}**"):
            st.markdown(f"**Verdict:** {v.verdict.value.replace('_', ' ').title()}")
            st.markdown(f"**Confidence:** {v.confidence:.2f}")
            st.markdown(f"**Justification:** {v.justification}")

            if v.flags:
                st.markdown(f"**Flags:** {', '.join(v.flags)}")

            tab_claim, tab_audit, tab_evidence, tab_debate = st.tabs(
                ["Claim", "Internal Audit", "External Evidence", "Debate"]
            )

            with tab_claim:
                st.json(claim.model_dump(), expanded=False)

            with tab_audit:
                audit = audits.get(v.claim_id)
                if not audit:
                    st.info("No internal audit recorded for this claim.")
                else:
                    acol1, acol2, acol3 = st.columns(3)
                    acol1.metric(
                        "Tables",
                        f"{SEVERITY_ICON.get(audit.table_consistency.value, '')} {audit.table_consistency.value}",
                    )
                    acol2.metric(
                        "Figures",
                        f"{SEVERITY_ICON.get(audit.figure_consistency.value, '')} {audit.figure_consistency.value}",
                    )
                    acol3.metric(
                        "Statistics",
                        f"{SEVERITY_ICON.get(audit.statistical_rigor.value, '')} {audit.statistical_rigor.value}",
                    )
                    st.markdown(
                        f"**Mismatch score:** {audit.mismatch_score:.2f} &nbsp;|&nbsp; "
                        f"**Baseline present:** {'yes' if audit.baseline_present else 'no'}",
                        unsafe_allow_html=True,
                    )
                    for label, detail in (
                        ("Table detail", audit.table_detail),
                        ("Figure detail", audit.figure_detail),
                        ("Statistical detail", audit.statistical_detail),
                        ("Methodology gap", audit.methodology_gap),
                    ):
                        if detail:
                            st.markdown(f"**{label}:** {detail}")
                    if audit.visual_mismatch_detail:
                        st.markdown("**🖼️ Visual verification (VLM read of the figure):**")
                        st.info(audit.visual_mismatch_detail)

            with tab_evidence:
                ext = evidence.get(v.claim_id)
                if not ext:
                    st.info("No external evidence recorded for this claim.")
                else:
                    if ext.staleness_entries:
                        for s in ext.staleness_entries:
                            st.warning(
                                f"**⏳ Baseline staleness — {s.baseline_name} ({s.baseline_year}), "
                                f"score {s.staleness_score:.1f}/10**\n\n{s.verdict}"
                            )
                            for m in s.missed_stronger:
                                st.caption(
                                    f"• Missed stronger: {m.get('name', '?')} "
                                    f"({m.get('year', '?')}) — {m.get('reason', '')}"
                                )
                    else:
                        st.caption("No staleness findings for this claim.")

                    for heading, papers in (
                        ("✅ Supporting", ext.supporting_papers),
                        ("❌ Contradicting", ext.contradicting_papers),
                        ("➖ Neutral", ext.neutral_papers),
                    ):
                        st.markdown(f"**{heading} ({len(papers)})**")
                        if not papers:
                            st.caption("None found.")
                        for p in papers:
                            link = f"[{p.title}]({p.url})" if p.url else p.title
                            st.markdown(
                                f"- {link} ({p.year}) — confidence {p.stance_confidence:.2f}"
                            )

            with tab_debate:
                transcript = transcripts.get(v.claim_id)
                if not transcript or not transcript.turns:
                    reason = transcript.terminated_reason if transcript else "not run"
                    st.info(f"No debate transcript available ({reason}).")
                else:
                    st.caption(
                        f"{transcript.total_rounds} round(s) — ended by "
                        f"{transcript.terminated_reason.replace('_', ' ')}"
                    )
                    for turn in transcript.turns:
                        role = "🔴 Prosecutor" if turn.agent.value == "prosecutor" else "🔵 Defender"
                        st.markdown(f"**{role} (Round {turn.round_num}):**")
                        st.markdown(turn.argument)
                        if turn.evidence_cited:
                            st.caption(f"📎 Cited: {', '.join(turn.evidence_cited)}")
                        if turn.new_retrieval:
                            st.caption(f"📡 Mid-debate search: \"{turn.new_retrieval.get('query', '')}\"")
                        if turn.concedes:
                            st.success(f"✋ {turn.agent.value.title()} conceded this point")
                        st.divider()

            # Strongest arguments
            if v.prosecutor_strongest:
                st.markdown(f"**🔴 Prosecutor's strongest point:** {v.prosecutor_strongest}")
            if v.defender_strongest:
                st.markdown(f"**🔵 Defender's strongest point:** {v.defender_strongest}")
            if v.unresolved:
                st.markdown(f"**❓ Unresolved:** {', '.join(v.unresolved)}")

    st.divider()
    st.download_button(
        "⬇️ Download full report (JSON)",
        data=report.model_dump_json(indent=2),
        file_name="paper_dissector_report.json",
        mime="application/json",
    )

elif not uploaded:
    # Landing page
    st.markdown("""
    ### How it works

    **Upload a research paper** and the system will:

    1. **Extract claims** — identifies specific, falsifiable assertions
    2. **Audit internally** — checks if the paper's own tables, figures, and stats support each claim
    3. **Search literature** — finds supporting and contradicting papers + checks for stale baselines
    4. **Debate** — a Prosecutor argues the claim is weak, a Defender argues it's strong, with live evidence retrieval
    5. **Judge** — weighs both sides and issues a verdict with confidence scores

    The output is a per-claim credibility report with the full debate as an audit trail.

    ---

    *Built with LangGraph, Gemini, Groq, Semantic Scholar, and adversarial multi-agent debate.*
    """)
