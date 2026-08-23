"""Offline regression checks — no API keys and no network required.

Run with:  python -m pytest tests/ -q      (or)      python tests/test_offline.py

Everything here exercises pure logic: JSON coercion, claim dedup, figure
matching, convergence detection, year resolution, verdict banding. The stages
that need a provider are covered by running the pipeline itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_dissector.agents.claim_extractor import deduplicate_claims
from paper_dissector.agents.debate import (
    _cited_evidence, _extract_search_query, argument_similarity,
)
from paper_dissector.agents.evidence_hunter import _dedupe_key, _staleness_score
from paper_dissector.agents.internal_auditor import build_audit_excerpt, match_figures_for_claim
from paper_dissector.agents.judge import _build_verdict, _label_for_score
from paper_dissector.config import CONVERGENCE_THRESHOLD, input_token_budget
from paper_dissector.llm import (
    LLMJSONError, _estimate_tokens, _fit_to_budget, extract_json,
)
from paper_dissector.sanitize import as_bool, as_str_list, as_text, clamp01, coerce_enum
from paper_dissector.schemas import AuditSeverity, Claim, VerdictLabel
from paper_dissector.tools.openalex import _reconstruct_abstract, _year_filter
from paper_dissector.tools.pdf_parser import _figure_number, _plausible_years, extract_year


def mkclaim(section="S1", text="x", **kw):
    base = dict(
        claim_id="C1", raw_text=text, subject="s", intervention="i",
        scope="sc", falsifiability_threshold="f", source_section=section,
    )
    base.update(kw)
    return Claim(**base)


# ── LLM JSON extraction ──────────────────────────────────────────

def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_code_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('```\n{"a": 2}\n```') == {"a": 2}


def test_extract_json_wrapped_in_prose():
    assert extract_json('Sure!\n{"a": 3}\nHope that helps.') == {"a": 3}


def test_extract_json_repairs_trailing_comma():
    assert extract_json('{"a": 4,}') == {"a": 4}


def test_extract_json_ignores_braces_inside_strings():
    assert extract_json('{"a": "has { brace"}') == {"a": "has { brace"}


def test_extract_json_wraps_bare_list():
    assert extract_json("[1,2,3]") == {"items": [1, 2, 3]}


def test_extract_json_raises_on_garbage():
    try:
        extract_json("no json here at all")
    except LLMJSONError:
        return
    raise AssertionError("expected LLMJSONError")


# ── Sanitisation of loose model output ───────────────────────────

def test_clamp01_rescales_percentages():
    assert clamp01(0.55) == 0.55
    assert clamp01(85) == 0.85
    assert clamp01(150) == 1.0
    assert clamp01(-3) == 0.0
    assert clamp01("abc", default=0.4) == 0.4


def test_coerce_enum_tolerates_model_phrasing():
    assert coerce_enum("supported", VerdictLabel, VerdictLabel.NOT_SUPPORTED) is VerdictLabel.SUPPORTED
    assert coerce_enum("partially supported", VerdictLabel, VerdictLabel.NOT_SUPPORTED) is VerdictLabel.PARTIALLY_SUPPORTED
    assert coerce_enum("BANANA", VerdictLabel, VerdictLabel.NOT_SUPPORTED) is VerdictLabel.NOT_SUPPORTED
    assert coerce_enum("fail", AuditSeverity, AuditSeverity.WARN) is AuditSeverity.FAIL


def test_list_and_text_coercion():
    assert as_str_list("one") == ["one"]
    assert as_str_list([{"point": "x"}]) == ["x"]
    assert as_str_list(None) == []
    assert as_text({"a": "hello"}) == "hello"
    assert as_bool("yes") is True and as_bool("no") is False


# ── Debate convergence and parsing ───────────────────────────────

def test_repeated_argument_is_detected_as_convergence():
    same = "The baseline is stale because ResNet-50 was superseded by ConvNeXt in 2022."
    reworded = "The baseline is stale, because ResNet-50 was superseded by ConvNeXt back in 2022."
    assert argument_similarity(same, same) > CONVERGENCE_THRESHOLD
    assert argument_similarity(same, reworded) > CONVERGENCE_THRESHOLD


def test_distinct_arguments_do_not_converge():
    a = "The baseline is stale because ResNet-50 was superseded by ConvNeXt in 2022."
    b = "Statistical rigor is absent: no confidence intervals are reported anywhere."
    assert argument_similarity(a, b) < CONVERGENCE_THRESHOLD
    assert argument_similarity("", "x") == 0.0


def test_convergence_separates_realistic_debate_turns():
    """
    Two arguments about one claim share its vocabulary, so similarity has to be
    measured over content words — with stopwords included, distinct arguments
    scored high enough to end debates on round two.
    """
    stale = ("The baseline is stale: ResNet-50 (2015) was superseded by ConvNeXt in 2022, "
             "and the audit shows baseline_present=false for the claimed comparison.")
    stale_reworded = ("The baseline is stale, because ResNet-50 from 2015 was superseded by "
                      "ConvNeXt back in 2022, and the audit reports baseline_present=false "
                      "for that comparison.")
    stats = ("Statistical rigor is absent. No confidence intervals, p-values or effect sizes "
             "are reported anywhere in Table 3, so the 0.4 point gap is indistinguishable "
             "from noise.")

    assert argument_similarity(stale, stale_reworded) > CONVERGENCE_THRESHOLD
    # A genuinely new line of attack must leave clear headroom under the threshold.
    assert argument_similarity(stale, stats) < 0.5


def test_progressive_rag_query_parsing():
    assert _extract_search_query('x\nSEARCH_REQUEST: "convnext imagenet"\ny') == "convnext imagenet"
    assert _extract_search_query("just an argument") is None


def test_evidence_citation_extraction():
    assert "10.1234/abc.def" in _cited_evidence("see 10.1234/abc.def")
    assert "table_consistency" in _cited_evidence("the table_consistency check failed")


# ── Figure to claim matching ─────────────────────────────────────

FIGURES = [
    {"figure_id": "figure_1", "kind": "figure", "figure_number": "figure:1", "caption": "Figure 1"},
    {"figure_id": "figure_2", "kind": "figure", "figure_number": "figure:2", "caption": "Figure 2"},
    {"figure_id": "table_1", "kind": "table", "figure_number": "table:3", "caption": "Table 3"},
]


def test_claim_matches_the_figure_it_cites():
    got = match_figures_for_claim(mkclaim("Section 4 / Figure 2"), FIGURES)
    assert [f["figure_id"] for f in got] == ["figure_2"]


def test_claim_matches_table_and_abbreviations():
    assert [f["figure_id"] for f in match_figures_for_claim(mkclaim("Table 3"), FIGURES)] == ["table_1"]
    assert [f["figure_id"] for f in match_figures_for_claim(mkclaim("Fig. 1"), FIGURES)] == ["figure_1"]


def test_claim_without_a_reference_matches_nothing():
    # The scaffold paired every claim with the first figure; it must not.
    assert match_figures_for_claim(mkclaim("Section 3"), FIGURES) == []
    assert match_figures_for_claim(mkclaim("Figure 9"), FIGURES) == []


def test_reference_in_claim_text_is_used_as_fallback():
    got = match_figures_for_claim(mkclaim("Results", "as shown in Figure 2"), FIGURES)
    assert [f["figure_id"] for f in got] == ["figure_2"]


# ── Audit excerpting and request-size guard ──────────────────────

PAPER = "\n\n".join([
    "# A Great Paper\n\nWe present a method.",
    "## 2 Related Work\n\n" + ("Prior work discussed at length. " * 120),
    "## 3 Method\n\n" + ("Architecture details. " * 120),
    "## 4 Results\n\nOur model reaches 41.8 BLEU on WMT14 EN-FR.",
    "Table 2: BLEU scores.\n\n| Model | BLEU |\n|---|---|\n| Ours | 41.8 |\n| ConvS2S | 40.46 |",
    "## 5 Conclusion\n\n" + ("Closing remarks. " * 120),
])


def _bleu_claim():
    return mkclaim(
        section="Section 4 / Table 2",
        text="Our model reaches 41.8 BLEU on WMT14 EN-FR.",
        metric="BLEU", reported_value=41.8, baseline="ConvS2S",
    )


def test_short_paper_is_passed_through_whole():
    short = "# Tiny\n\nOne claim here."
    assert build_audit_excerpt(_bleu_claim(), short, max_chars=10_000) == short


def test_excerpt_respects_the_character_budget():
    out = build_audit_excerpt(_bleu_claim(), PAPER, max_chars=1200)
    assert len(out) <= 1200 + 200   # allow the "[...]" joiners


def test_excerpt_keeps_the_cited_table_and_reported_value():
    out = build_audit_excerpt(_bleu_claim(), PAPER, max_chars=1200)
    assert "41.8" in out
    assert "Table 2" in out
    # Bulk prose the claim does not reference should be dropped first.
    assert out.count("Prior work discussed") < PAPER.count("Prior work discussed")


def test_request_guard_trims_only_oversized_user_content():
    system = {"role": "system", "content": "S" * 400}
    user = {"role": "user", "content": "U" * 80_000}
    out = _fit_to_budget([system, user], "groq")
    assert out[0]["content"] == system["content"]        # contract preserved
    assert len(out[1]["content"]) < len(user["content"])
    # Must land under the budget by the estimator's own reckoning, not a looser one.
    assert sum(_estimate_tokens(m["content"]) for m in out) <= input_token_budget("groq")


def test_token_estimate_is_conservative():
    # chars/4 under-counted a real payload by ~8% and the request was rejected.
    text = "word " * 2000
    assert _estimate_tokens(text) > len(text) // 4


def test_explicit_budget_overrides_the_provider_default():
    msgs = [{"role": "user", "content": "U" * 40_000}]
    tight = _fit_to_budget(msgs, "groq", budget=1500)
    assert sum(_estimate_tokens(m["content"]) for m in tight) <= 1500


def test_request_guard_leaves_small_requests_alone():
    msgs = [{"role": "user", "content": "hi"}]
    assert _fit_to_budget(msgs, "groq") == msgs


# ── Claim deduplication ──────────────────────────────────────────

def test_duplicate_claims_merge_keeping_the_richer_one():
    claims = [
        mkclaim(text="Our model achieves 94.2 F1 on SciFact.", claim_id="C1"),
        mkclaim(text="Our model achieves 94.2 F1 on SciFact.", claim_id="C2", metric="F1", reported_value=94.2),
        mkclaim(text="Training converges in 3 hours on one GPU.", claim_id="C3"),
    ]
    deduped = deduplicate_claims(claims)
    assert len(deduped) == 2
    assert deduped[0].metric == "F1"


def test_distinct_claims_are_not_merged():
    claims = [
        mkclaim(text="Accuracy improved by 5 points.", claim_id="C1"),
        mkclaim(text="Latency dropped by 40 percent.", claim_id="C2"),
    ]
    assert len(deduplicate_claims(claims)) == 2


# ── Year resolution ──────────────────────────────────────────────

def test_explicit_date_lines_win():
    assert extract_year("Published: 2024\nSome abstract") == 2024
    assert extract_year("arXiv:2301.00001v2 [cs.CL] 3 Feb 2023") == 2023
    assert extract_year("(c) 2019 The Authors") == 2019


def test_model_hyperparameters_are_not_mistaken_for_years():
    # d_ff = 2048 is near-universal in transformer papers.
    assert _plausible_years("the inner-layer has dimensionality d_ff = 2048") == []
    assert _plausible_years("Bahdanau 2015, Wu 2016, ICLR 2017") == [2015, 2016, 2017]


def test_implausible_future_years_are_rejected():
    assert _plausible_years("somehow 2029 appears here") == []


def test_no_year_available():
    assert extract_year("no dates here", title="") is None


def test_figure_number_parsing():
    assert _figure_number("Figure 4: accuracy over time", "figure") == "figure:4"
    assert _figure_number("Table 2. Ablations", "table") == "table:2"
    assert _figure_number("Some caption", "figure") is None


# ── OpenAlex helpers ─────────────────────────────────────────────

def test_abstract_reconstruction_from_inverted_index():
    assert _reconstruct_abstract({"We": [0], "propose": [1], "a": [2], "model": [3]}) == "We propose a model"
    assert _reconstruct_abstract({"the": [0, 2], "big": [1], "dog": [3]}) == "the big the dog"
    assert _reconstruct_abstract(None) == ""
    assert _reconstruct_abstract({}) == ""


def test_year_range_translation():
    assert _year_filter("-2020") == "publication_year:<2020"
    assert _year_filter("2018-") == "publication_year:>2017"
    assert _year_filter("2015-2020") == "publication_year:2015-2020"
    assert _year_filter(None) is None
    assert _year_filter("junk") is None


def test_papers_dedupe_on_doi_across_provider_ids():
    # OpenAlex stores reindexed duplicates under different ids, same DOI.
    a = {"paperId": "W1", "title": "T", "externalIds": {"DOI": "10.48550/arxiv.1910.10683"}}
    b = {"paperId": "W2", "title": "T", "externalIds": {"DOI": "10.48550/ARXIV.1910.10683"}}
    c = {"paperId": "W3", "title": "Different", "externalIds": {}}
    assert _dedupe_key(a) == _dedupe_key(b)
    assert _dedupe_key(a) != _dedupe_key(c)


# ── Scoring ──────────────────────────────────────────────────────

def test_staleness_score_scaling():
    assert _staleness_score(5.0) == 5.0
    assert _staleness_score(0.8) == 8.0    # 0-1 scale rescaled to 0-10
    assert _staleness_score(42) == 10.0
    assert _staleness_score("n/a") == 0.0


def test_verdict_bands():
    assert _label_for_score(0.9) is VerdictLabel.STRONGLY_SUPPORTED
    assert _label_for_score(0.7) is VerdictLabel.SUPPORTED
    assert _label_for_score(0.5) is VerdictLabel.PARTIALLY_SUPPORTED
    assert _label_for_score(0.3) is VerdictLabel.WEAKLY_SUPPORTED
    assert _label_for_score(0.05) is VerdictLabel.NOT_SUPPORTED


def test_verdict_recovers_from_malformed_judge_output():
    v = _build_verdict(
        {"verdict": "garbage", "confidence": 75, "justification": "j",
         "flags": "STALE BASELINE", "unresolved": None},
        "C1",
    )
    assert v.verdict is VerdictLabel.SUPPORTED   # derived from the 0.75 confidence
    assert v.confidence == 0.75
    assert v.flags == ["STALE_BASELINE"]
    assert v.unresolved == []


# ── Runner for plain `python tests/test_offline.py` ──────────────

if __name__ == "__main__":
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as exc:
            print(f"  FAIL {name}: {exc}")
            failures.append(name)
    print("=" * 60)
    if failures:
        print(f"{len(failures)}/{len(tests)} FAILED: {failures}")
        sys.exit(1)
    print(f"all {len(tests)} offline tests passed")
