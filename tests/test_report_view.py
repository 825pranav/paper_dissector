"""Render the Streamlit report view and assert every section appears.

The report view is the project's actual deliverable, and until this existed the
only way to check it was to run the pipeline and look at a browser. Here the UI
is driven headlessly against a saved analysis, so a change that silently stops
rendering the debate transcript or the staleness warning fails a test instead of
being noticed by eye.

Run with:  python tests/test_report_view.py   (or via pytest)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from streamlit.testing.v1 import AppTest  # noqa: E402

from paper_dissector.report_io import to_json  # noqa: E402
from test_offline import _sample_state  # noqa: E402


def _render_saved_analysis() -> AppTest:
    """Run app.py with a saved analysis preloaded, and return the rendered app."""
    path = Path(tempfile.gettempdir()) / "paper_dissector_test_analysis.json"
    path.write_text(to_json(_sample_state()), encoding="utf-8")

    os.environ["PD_ANALYSIS_JSON"] = str(path)
    try:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        app.run()
    finally:
        os.environ.pop("PD_ANALYSIS_JSON", None)
    return app


def test_report_view_renders_every_section():
    app = _render_saved_analysis()
    assert not list(app.exception), f"app raised: {[str(e.value) for e in app.exception]}"

    markdown = " ".join(m.value for m in app.markdown)
    headers = [h.value for h in app.header]
    metrics = {m.label: m.value for m in app.metric}
    tabs = [t.label for t in app.tabs]
    infos = " ".join(i.value for i in app.info)
    warnings = " ".join(w.value for w in app.warning)
    captions = " ".join(c.value for c in app.caption)
    successes = " ".join(s.value for s in app.success)

    # Executive summary
    assert any("A Great Paper" in h for h in headers)
    assert metrics["Overall Credibility"] == "0.75"
    assert metrics["Verdict"] == "Supported"
    assert str(metrics["Claims Analyzed"]) == "1"
    # Counted by verdict label, not by the judge's confidence.
    assert metrics["Well-supported Claims"] == "1/1"

    # Per-claim breakdown
    assert any("C1" in e.label for e in app.expander)
    assert "Matches Table 2." in markdown          # justification
    assert "NO_STATISTICAL_TEST" in markdown       # flags
    assert tabs == ["Claim", "Internal Audit", "External Evidence", "Debate"]

    # Internal audit, including the multimodal reading
    assert metrics["Tables"].endswith("PASS")
    assert "The table shows 41.8" in infos

    # External evidence and staleness
    assert "A corroborating study" in markdown
    assert "ConvS2S" in warnings

    # Debate transcript, progressive RAG and concession
    assert "No CIs reported." in markdown
    assert "bleu significance" in captions
    assert "conceded" in successes.lower()


def test_landing_page_shows_when_nothing_is_loaded():
    os.environ.pop("PD_ANALYSIS_JSON", None)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    app.run()
    assert not list(app.exception)
    assert any("How it works" in m.value for m in app.markdown)
    # The analyze button exists but is disabled until a PDF is supplied.
    assert [b.disabled for b in app.button] == [True]


def test_saved_analysis_file_is_reloadable():
    """The download the UI offers must be loadable by the uploader it offers."""
    from paper_dissector.report_io import from_dict

    payload = json.loads(to_json(_sample_state()))
    restored = from_dict(payload)
    assert restored["final_report"].total_claims == 1
    assert restored["debate_transcripts"][0].turns[-1].concedes is True
    # Credibility must survive the round trip, since the score aggregates it.
    assert restored["verdicts"][0].credibility == 0.75


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
    print(f"all {len(tests)} report-view tests passed")
