"""Save and reload a completed analysis.

A full run costs several minutes of rate-limited API calls, so the result is
worth keeping: this lets the UI redisplay a previous analysis without spending
quota again, and makes the report view testable without driving a browser.

Figure images are deliberately dropped — they are large base64 blobs and the
report view never displays them, only the VLM's reading of them, which is
already captured in the audit.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from paper_dissector.schemas import (
    Claim, ClaimVerdict, DebateTranscript, ExternalEvidenceResult,
    FinalReport, InternalAuditResult,
)

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# state key -> pydantic model for the list-valued sections
_COLLECTIONS: dict[str, type] = {
    "claims": Claim,
    "internal_audits": InternalAuditResult,
    "external_evidence": ExternalEvidenceResult,
    "debate_transcripts": DebateTranscript,
    "verdicts": ClaimVerdict,
}

_SCALARS = ("paper_title", "paper_authors", "paper_year")


def to_dict(state: dict) -> dict:
    """Serialise the parts of the pipeline state the report view needs."""
    out: dict[str, Any] = {"schema_version": SCHEMA_VERSION}

    for key in _SCALARS:
        out[key] = state.get(key)

    for key in _COLLECTIONS:
        out[key] = [item.model_dump(mode="json") for item in (state.get(key) or [])]

    report = state.get("final_report")
    out["final_report"] = report.model_dump(mode="json") if report else None

    # Keep figure metadata for provenance, but not the image bytes.
    out["extracted_figures"] = [
        {k: v for k, v in (fig or {}).items() if k != "image_b64"}
        for fig in (state.get("extracted_figures") or [])
    ]
    return out


def from_dict(payload: dict) -> dict:
    """Rebuild a state-shaped dict of pydantic objects from saved JSON."""
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        log.warning("analysis file schema version %r, expected %r; "
                    "loading anyway", version, SCHEMA_VERSION)

    state: dict[str, Any] = {key: payload.get(key) for key in _SCALARS}
    state["extracted_figures"] = payload.get("extracted_figures") or []

    for key, model in _COLLECTIONS.items():
        rebuilt = []
        for raw in payload.get(key) or []:
            try:
                rebuilt.append(model.model_validate(raw))
            except Exception as exc:
                log.warning("skipping malformed %s entry: %s", key, exc)
        state[key] = rebuilt

    raw_report = payload.get("final_report")
    if raw_report:
        try:
            state["final_report"] = FinalReport.model_validate(raw_report)
        except Exception as exc:
            log.error("could not load final_report: %s", exc)
            state["final_report"] = None
    else:
        state["final_report"] = None

    return state


def to_json(state: dict, indent: int | None = 2) -> str:
    return json.dumps(to_dict(state), indent=indent, ensure_ascii=False)


def save_analysis(state: dict, path: str | Path) -> Path:
    """Write a completed analysis to disk. Returns the path written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_json(state), encoding="utf-8")
    log.info("saved analysis to %s", target)
    return target


def load_analysis(path: str | Path) -> dict:
    """Read an analysis previously written by save_analysis."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return from_dict(payload)


def load_analysis_json(text: str | bytes) -> dict:
    """Read an analysis from raw JSON, e.g. an uploaded file."""
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return from_dict(json.loads(text))
