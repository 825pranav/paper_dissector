"""Re-run adjudication over a saved analysis.

The judge is the cheapest stage to change and the most expensive to test: a
full pipeline run costs several minutes and a large share of a daily token
quota, but adjudication itself is one call per claim. This replays the judge
over an analysis that already has its claims, audits, evidence and debate
transcripts, so the prompt or model can be iterated on directly.

    python readjudicate.py analysis.json
    python readjudicate.py analysis.json -o rejudged.json
    python readjudicate.py analysis.json --only-failed

``--only-failed`` retries just the claims whose adjudication failed (a quota
error, say) and leaves existing verdicts alone.
"""

from __future__ import annotations

import argparse
import logging
import sys

from paper_dissector.agents.judge import adjudicate, compile_report
from paper_dissector.report_io import load_analysis, save_analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("analysis", help="analysis JSON written by a pipeline run")
    parser.add_argument("-o", "--out", help="where to write (default: overwrite the input)")
    parser.add_argument("--only-failed", action="store_true",
                        help="keep existing verdicts, retry only the ones that failed")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    state = load_analysis(args.analysis)
    all_claims = state.get("claims") or []
    if not all_claims:
        print(f"{args.analysis}: no claims found", file=sys.stderr)
        return 1

    previous = {v.claim_id: v for v in state.get("verdicts") or []}

    if args.only_failed:
        retry_ids = {
            cid for cid, v in previous.items() if "ADJUDICATION_FAILED" in v.flags
        }
        if not retry_ids:
            print("nothing to retry: no claim has ADJUDICATION_FAILED")
            return 0
        print(f"retrying {len(retry_ids)} claim(s): {sorted(retry_ids)}")
    else:
        retry_ids = {c.claim_id for c in all_claims}

    # Adjudicate only the claims we intend to redo.
    subset = dict(state, claims=[c for c in all_claims if c.claim_id in retry_ids])
    fresh = {v.claim_id: v for v in adjudicate(subset)["verdicts"]}

    # Keep prior verdicts for everything else, in the original claim order.
    verdicts = [
        fresh.get(c.claim_id) or previous.get(c.claim_id)
        for c in all_claims
    ]
    verdicts = [v for v in verdicts if v is not None]

    state["claims"] = all_claims
    state["verdicts"] = verdicts
    state["final_report"] = compile_report(state, verdicts)
    report = state["final_report"]

    print(f"\n{report.paper_title}")
    print(f"overall credibility {report.overall_score} ({report.overall_verdict.value})")
    if report.systemic_issues:
        for issue in report.systemic_issues:
            print(f"  ! {issue}")
    for v in report.claim_verdicts:
        old = previous.get(v.claim_id)
        changed = f"   (was {old.verdict.value})" if old and old.verdict is not v.verdict else ""
        print(f"  [{v.claim_id}] {v.verdict.value:22} cred={v.credibility}{changed}")
        print(f"      {v.justification[:200]}")

    out = args.out or args.analysis
    save_analysis(state, out)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
