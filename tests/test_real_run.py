"""Regression tests driven by real pipeline output — no keys or network.

The fixtures under tests/fixtures were captured from a live run on the GELU
paper (arXiv 1606.08415) before these fixes, and every defect tested here was
observed in that run:

    gelu_run_before_fixes.json   the saved analysis (claims, audits, evidence,
                                 debate transcripts, verdicts, figure captions)
    gelu_markdown.md             the parser's markdown for the same paper
    gemini_429_per_minute.txt    the 429 Gemini returned during claim extraction

Run with:  python -m pytest tests/ -q
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_dissector.agents import judge
from paper_dissector.agents.internal_auditor import _build_audit
from paper_dissector.report_io import from_dict

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def run() -> dict:
    """The real pre-fix analysis, as pydantic objects."""
    payload = json.loads((FIXTURES / "gelu_run_before_fixes.json").read_text(encoding="utf-8"))
    state = from_dict(payload)
    state["raw"] = payload
    return state


def _by_id(items) -> dict:
    return {i.claim_id: i for i in items}


# ── 1. Fabricated visual findings ────────────────────────────────

def test_text_auditor_cannot_write_the_vlm_field(run):
    # In the real run no VLM call fired, yet C2's audit carried a "reading" of
    # Figure 5 written by the text-only auditor.
    raw_c2 = next(a for a in run["raw"]["internal_audits"] if a["claim_id"] == "C2")
    assert raw_c2["visual_mismatch_detail"], "fixture should show the fabricated field"

    rebuilt = _build_audit(raw_c2, "C2")
    assert rebuilt.visual_mismatch_detail is None
    # Everything else the text auditor said is kept.
    assert rebuilt.figure_consistency.value == raw_c2["figure_consistency"]


def test_real_vlm_reading_is_kept(run):
    raw_c2 = next(a for a in run["raw"]["internal_audits"] if a["claim_id"] == "C2")
    rebuilt = _build_audit(raw_c2, "C2", vlm_reading="[figure:5] Learning curves for GELU, ReLU, ELU")
    assert rebuilt.visual_mismatch_detail.startswith("[figure:5]")


def test_visual_mismatch_flag_needs_a_vlm_reading(run):
    verdict = _by_id(run["verdicts"])["C2"]
    assert "VISUAL_MISMATCH" in verdict.flags          # what the judge emitted

    raw_c2 = next(a for a in run["raw"]["internal_audits"] if a["claim_id"] == "C2")
    audit = _build_audit(raw_c2, "C2")                  # as the fixed auditor stores it
    evidence = _by_id(run["external_evidence"])["C2"]
    assert "VISUAL_MISMATCH" not in judge.ground_flags(verdict.flags, audit, evidence)

    with_vlm = _build_audit(raw_c2, "C2", vlm_reading="[figure:5] curves overlap")
    assert "VISUAL_MISMATCH" in judge.ground_flags(verdict.flags, with_vlm, evidence)


def test_report_does_not_claim_unexamined_figures_fail(run):
    before = run["final_report"].systemic_issues
    assert any("figure(s) do not support" in s for s in before)

    audits = [_build_audit(a, a["claim_id"]) for a in run["raw"]["internal_audits"]]
    state = dict(run, internal_audits=audits)
    report = judge.compile_report(state, run["verdicts"])
    assert not any("figure(s) do not support" in s for s in report.systemic_issues)


# ── 2. The VLM never fired ───────────────────────────────────────

def test_real_claims_without_figure_refs_match_by_caption(run):
    from paper_dissector.agents.internal_auditor import match_figures_for_claim

    claims = _by_id(run["claims"])
    figures = run["extracted_figures"]
    # None of the three real claims names a figure; all cite "Section 3.x".
    assert all("figure" not in c.source_section.lower() for c in claims.values())

    def matched(cid):
        return [f["figure_number"] for f in match_figures_for_claim(claims[cid], figures)]

    assert matched("C2") == ["figure:5"]    # TIMIT claim -> "TIMIT Frame Classification"
    assert matched("C3") == ["figure:6"]    # CIFAR-10 claim -> "CIFAR-10 Results", not CIFAR-100


def test_method_names_alone_do_not_match_a_figure(run, caplog):
    from paper_dissector.agents.internal_auditor import match_figures_for_claim

    # C1 is Twitter POS tagging, which has no figure. Its text names GELU, ReLU
    # and ELU, as does Figure 1's caption; that must not count as a match.
    c1 = _by_id(run["claims"])["C1"]
    with caplog.at_level(logging.WARNING):
        assert match_figures_for_claim(c1, run["extracted_figures"]) == []
    assert "C1: no figure or table matched" in caplog.text


def test_extractor_prompt_asks_for_figure_references():
    from paper_dissector.agents.claim_extractor import SYSTEM_PROMPT
    assert "Figure 6" in SYSTEM_PROMPT and "depends on one" in SYSTEM_PROMPT


# ── 3. Debates ended after one round ─────────────────────────────

def _replay(monkeypatch, turns):
    """Stand the debate LLM in with recorded real turns, in order."""
    from paper_dissector.agents import debate

    queue = [t.argument for t in turns]
    monkeypatch.setattr(debate, "MAX_PRAG_RETRIEVALS", 0)   # searches were already folded in
    monkeypatch.setattr(debate, "chat_text", lambda *a, **k: queue.pop(0))
    return debate


def test_partial_concessions_from_the_real_run_are_not_full(run):
    from paper_dissector.agents.debate import _check_concession

    conceding = [
        (t.claim_id, turn.argument) for t in run["debate_transcripts"]
        for turn in t.turns if turn.concedes
    ]
    # All three real debates were ended by a defender "CONCEDE:" on a side point
    # (missing confidence intervals, a table-only policy) while defending the claim.
    assert len(conceding) == 3
    assert all("CONCEDE:" in text for _, text in conceding)
    assert not any(_check_concession(text) for _, text in conceding)


def test_real_c1_debate_continues_past_round_one(run, monkeypatch):
    transcript = _by_id(run["debate_transcripts"])["C1"]
    assert transcript.total_rounds == 1                       # before the fix
    assert transcript.terminated_reason == "defender_concession"

    # Replay C1's real round, then C3's real rounds as further material.
    c3 = _by_id(run["debate_transcripts"])["C3"]
    debate = _replay(monkeypatch, transcript.turns + c3.turns)
    monkeypatch.setattr(debate, "MAX_DEBATE_ROUNDS", 3)
    claims, audits, ev = (_by_id(run[k]) for k in ("claims", "internal_audits", "external_evidence"))

    result = debate._run_one_debate(claims["C1"], audits["C1"], ev["C1"])
    assert result.total_rounds == 3
    assert result.terminated_reason == "max_rounds"
    assert not any(t.concedes for t in result.turns)


def test_full_concession_waits_for_the_minimum_rounds(run, monkeypatch):
    # The real C1 defender turn, with its point concession promoted to a full one.
    real = _by_id(run["debate_transcripts"])["C1"].turns
    promoted = [t.model_copy(update={"argument": t.argument.replace("CONCEDE:", "CONCEDE_CLAIM:")})
                for t in real]
    assert debate_check(promoted[1].argument)

    # Round 2 uses a different real prosecutor argument, so repetition does not stop it first.
    c3_prosecutor = _by_id(run["debate_transcripts"])["C3"].turns[0]
    debate = _replay(monkeypatch, promoted + [c3_prosecutor, promoted[1]])
    monkeypatch.setattr(debate, "MAX_DEBATE_ROUNDS", 3)
    claims, audits, ev = (_by_id(run[k]) for k in ("claims", "internal_audits", "external_evidence"))

    result = debate._run_one_debate(claims["C1"], audits["C1"], ev["C1"])
    # Round 1's concession does not stop it; round 2's does.
    assert result.total_rounds == 2
    assert result.terminated_reason == "defender_concession"


def test_repetition_still_stops_the_debate_after_the_minimum(run, monkeypatch):
    real = _by_id(run["debate_transcripts"])["C3"].turns
    same_pair = [real[0], real[1]] * 3       # both sides repeat their real round-1 argument

    debate = _replay(monkeypatch, same_pair)
    monkeypatch.setattr(debate, "MAX_DEBATE_ROUNDS", 3)
    claims, audits, ev = (_by_id(run[k]) for k in ("claims", "internal_audits", "external_evidence"))

    result = debate._run_one_debate(claims["C3"], audits["C3"], ev["C3"])
    assert result.terminated_reason == "convergence"
    assert result.total_rounds == 2


def debate_check(text: str) -> bool:
    from paper_dissector.agents.debate import _check_concession
    return _check_concession(text)


# ── 4. Gemini 429 handling ───────────────────────────────────────

def _rate_limit_error(text: str):
    import httpx
    from openai import RateLimitError

    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
    return RateLimitError(text, response=httpx.Response(429, request=request), body=None)


def _real_429() -> str:
    return (FIXTURES / "gemini_429_per_minute.txt").read_text(encoding="utf-8")


def test_real_per_minute_429_is_retried():
    from paper_dissector import llm

    exc = _rate_limit_error(_real_429())
    # The generic wording that used to be read as a daily cap is present...
    assert "free_tier_requests" in str(exc) and "RESOURCE_EXHAUSTED" in str(exc)
    # ...but the quotaId says per minute.
    assert llm._quota_ids(exc) == ["GenerateRequestsPerMinutePerProjectPerModel-FreeTier"]
    assert llm._is_retryable(exc)


def test_per_minute_429_waits_about_a_minute_then_succeeds(monkeypatch):
    from types import SimpleNamespace
    from paper_dissector import llm

    exc = _rate_limit_error(_real_429())
    calls, waits = [], []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise exc
        return "ok"

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm._create.retry, "sleep", waits.append)

    assert llm._create(client, model="gemini-3.6-flash", messages=[]) == "ok"
    assert len(calls) == 2
    assert waits == [llm.PER_MINUTE_QUOTA_WAIT] and 55 <= waits[0] <= 65


def test_per_day_quota_is_not_retried():
    from paper_dissector import llm

    # The real error with its quotaId swapped for Gemini's per-day quota.
    daily = _real_429().replace("RequestsPerMinutePerProject", "RequestsPerDayPerProject")
    assert not llm._is_retryable(_rate_limit_error(daily))


# ── 5. Judge flags not grounded ──────────────────────────────────

@pytest.fixture(scope="module")
def markdown() -> str:
    return (FIXTURES / "gelu_markdown.md").read_text(encoding="utf-8")


def test_numbers_reported_only_in_text_are_not_a_table_failure(run, markdown):
    from paper_dissector.agents.internal_auditor import stated_only_in_text

    audits, claims = _by_id(run["internal_audits"]), _by_id(run["claims"])
    # The real audit failed C2 and C3 because "no table in the paper lists these
    # numbers" — GELU's only table is a CIFAR-10 architecture listing.
    assert audits["C2"].table_consistency.value == "FAIL"
    assert audits["C3"].table_consistency.value == "FAIL"
    assert "no table" in audits["C2"].table_detail
    assert all(stated_only_in_text(claims[c], markdown) for c in ("C1", "C2", "C3"))


def test_a_table_reporting_the_quantity_keeps_the_failure(run, markdown):
    from paper_dissector.agents.internal_auditor import stated_only_in_text

    # The real paper plus a results table that disagrees with the text: now a
    # table does cover the claim, so a mismatch is possible and must stand.
    tabled = markdown + "\n\n| Model | CIFAR-10 error |\n|---|---|\n| GELU | 8.02 |\n"
    assert not stated_only_in_text(_by_id(run["claims"])["C3"], tabled)


def test_stale_baseline_flag_requires_a_stale_verdict(run):
    verdicts, ev = _by_id(run["verdicts"]), _by_id(run["external_evidence"])
    audits = _by_id(run["internal_audits"])

    # The judge flagged C2 STALE_BASELINE though its baseline check said CURRENT.
    assert "STALE_BASELINE" in verdicts["C2"].flags
    assert [s.verdict for s in ev["C2"].staleness_entries] == ["CURRENT"]
    assert "STALE_BASELINE" not in judge.ground_flags(verdicts["C2"].flags, audits["C2"], ev["C2"])

    # C3's check did find a stale baseline, so its flag stands.
    assert ev["C3"].staleness_entries[0].verdict.startswith("STALE")
    assert "STALE_BASELINE" in judge.ground_flags(verdicts["C3"].flags, audits["C3"], ev["C3"])


def test_real_judge_flags_after_grounding(run, markdown):
    """End to end over the real run: audit correction, then flag grounding."""
    from paper_dissector.agents.internal_auditor import stated_only_in_text
    from paper_dissector.schemas import AuditSeverity

    claims, ev = _by_id(run["claims"]), _by_id(run["external_evidence"])
    grounded = {}
    for raw in run["raw"]["internal_audits"]:
        cid = raw["claim_id"]
        audit = _build_audit(raw, cid)                 # the fixed auditor: no VLM ran
        if audit.table_consistency == AuditSeverity.FAIL and stated_only_in_text(claims[cid], markdown):
            audit.table_consistency = AuditSeverity.WARN
        verdict = _by_id(run["verdicts"])[cid]
        grounded[cid] = judge.ground_flags(verdict.flags, audit, ev[cid])

    assert grounded == {
        "C1": ["STAT_RIGOR_WARN"],
        "C2": [],                                       # STALE, VISUAL and TABLE all unsupported
        "C3": ["STALE_BASELINE", "STAT_RIGOR_WARN"],
    }


# ── 6. Failed stance classifications were cached ─────────────────

def _real_claim_and_abstracts(run):
    ev = _by_id(run["external_evidence"])["C1"]
    return _by_id(run["claims"])["C1"].raw_text, [p.relevant_passage for p in ev.neutral_papers[:4]]


def test_failed_stance_classification_is_not_cached(run, monkeypatch):
    from paper_dissector import llm
    from paper_dissector.tools import stance_classifier as sc

    claim, abstracts = _real_claim_and_abstracts(run)
    stored = {}
    monkeypatch.setattr(sc, "_hf_available", False)
    monkeypatch.setattr(sc.cache, "get", lambda key, default=None: stored.get(key, default))
    monkeypatch.setattr(sc.cache, "set", lambda key, value, ttl=None: stored.__setitem__(key, value))

    def fail(*a, **k):
        raise _rate_limit_error(_real_429())
    monkeypatch.setattr(llm, "chat_json", fail)

    results = sc.batch_classify(claim, abstracts)
    assert results == [(sc.Stance.NEUTRAL, 0.0)] * len(abstracts)   # degrades as before
    assert stored == {}                                              # but nothing is cached
    assert sc.classify_stance(claim, abstracts[0]) == (sc.Stance.NEUTRAL, 0.0)
    assert stored == {}


def test_only_answered_stance_classifications_are_cached(run, monkeypatch):
    from paper_dissector import llm
    from paper_dissector.tools import stance_classifier as sc

    claim, abstracts = _real_claim_and_abstracts(run)
    stored = {}
    monkeypatch.setattr(sc, "_hf_available", False)
    monkeypatch.setattr(sc.cache, "get", lambda key, default=None: stored.get(key, default))
    monkeypatch.setattr(sc.cache, "set", lambda key, value, ttl=None: stored.__setitem__(key, value))
    # The real run's answers for these abstracts (all neutral, high confidence),
    # but only for the first two: the two the model skipped stay uncached.
    real = [p.stance_confidence for p in _by_id(run["external_evidence"])["C1"].neutral_papers[:4]]
    partial = real[:2]
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: {"results": [
        {"index": i + 1, "stance": "NEUTRAL", "confidence": c} for i, c in enumerate(partial)
    ]})
    sc.batch_classify(claim, abstracts)
    assert len(stored) == 2


def test_sdk_retries_do_not_multiply_requests(monkeypatch):
    """
    In the post-fix GELU run Gemini answered six 503s and then a 429 for what
    were two of our attempts: the OpenAI SDK retried twice under each one,
    spending the 5/minute quota. Replay that pattern: 503, 503, then success
    must cost three HTTP requests, not seven.
    """
    import httpx
    from paper_dissector import config, llm

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    config.get_gemini_client.cache_clear()
    try:
        requests = []
        replies = [503, 503, 200]

        def handler(request):
            requests.append(request)
            status = replies[min(len(requests), len(replies)) - 1]
            if status == 503:
                return httpx.Response(503, json={"error": {"code": 503, "status": "UNAVAILABLE"}})
            return httpx.Response(200, json={
                "id": "x", "object": "chat.completion", "created": 0, "model": "gemini-3.6-flash",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok"}}],
            })

        client = config.get_gemini_client()
        assert client.max_retries == 0
        client = client.with_options(http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        monkeypatch.setattr(llm._create.retry, "sleep", lambda s: None)

        resp = llm._create(client, model="gemini-3.6-flash", messages=[{"role": "user", "content": "hi"}])
        assert resp.choices[0].message.content == "ok"
        assert len(requests) == 3
    finally:
        config.get_gemini_client.cache_clear()


def test_groq_retry_after_is_honoured(monkeypatch):
    """
    The post-fix run hit 34 Groq per-minute 429s, all recovered by the SDK's
    own retries, which honour retry-after. With those retries off, the wait
    must come from the server's answer instead of a short exponential guess.
    (Groq's bodies were not logged; this uses its documented 429 wording.)
    """
    import httpx
    from openai import RateLimitError
    from paper_dissector import llm

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    message = ("Rate limit reached for model `openai/gpt-oss-120b` on tokens per minute (TPM): "
               "Limit 8000, Used 7412, Requested 1830. Please try again in 9.24s.")
    with_header = RateLimitError(message, body=None, response=httpx.Response(
        429, request=request, headers={"retry-after": "9"}))
    without_header = RateLimitError(message, body=None, response=httpx.Response(429, request=request))

    assert llm._is_retryable(with_header)                 # tokens per MINUTE: not a daily cap
    assert llm._server_retry_after(with_header) == 9.0
    assert llm._server_retry_after(without_header) == 9.24

    calls, waits = [], []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) <= 6:          # more than the old 5-attempt budget
            raise with_header
        return "ok"

    from types import SimpleNamespace
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm._create.retry, "sleep", waits.append)
    assert llm._create(client, model="openai/gpt-oss-120b", messages=[]) == "ok"
    assert waits == [10.0] * 6


def test_groq_daily_token_cap_is_not_retried():
    import httpx
    from openai import RateLimitError
    from paper_dissector import llm

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    daily = RateLimitError(
        "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): "
        "Limit 200000, Used 199500, Requested 1830. Please try again in 7m12s.",
        body=None, response=httpx.Response(429, request=request))
    assert not llm._is_retryable(daily)


# ── Ollama provider ──────────────────────────────────────────────

def test_a_role_can_be_pointed_at_ollama(monkeypatch):
    from paper_dissector import config

    monkeypatch.setitem(config.AGENT_CONFIG, "judge", {"provider": "ollama", "model": "qwen2.5:7b"})
    client, model = config.get_client_for_agent("judge")
    assert model == "qwen2.5:7b"
    assert str(client.base_url).startswith("http://localhost:11434/v1")
    assert client.max_retries == 0
    # A budget exists, so oversized requests are trimmed rather than silently
    # truncated by Ollama (it cut a 7.8k-token GELU prompt to 2050 tokens in testing).
    assert 0 < config.input_token_budget("ollama") < 12288


def test_unknown_provider_names_the_options(monkeypatch):
    from paper_dissector import config

    monkeypatch.setitem(config.AGENT_CONFIG, "judge", {"provider": "huggingface", "model": "x"})
    with pytest.raises(ValueError, match="ollama"):
        config.get_client_for_agent("judge")
