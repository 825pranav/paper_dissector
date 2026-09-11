from __future__ import annotations
from pydantic import BaseModel, Field
from enum import Enum


# ── Claim Extraction ─────────────────────────────────────────────

class Claim(BaseModel):
    claim_id: str
    raw_text: str = Field(description="Verbatim sentence(s) from the paper")
    subject: str = Field(description="Model, algorithm, compound, or system under study")
    intervention: str = Field(description="What was done — the proposed method/change")
    baseline: str | None = Field(default=None, description="What it's compared against")
    metric: str | None = Field(default=None, description="Quantitative measure (F1, accuracy, p-value)")
    reported_value: float | str | None = Field(default=None, description="The claimed number")
    scope: str = Field(description="Conditions — dataset, setting, constraints")
    falsifiability_threshold: str = Field(description="What would disprove this claim")
    source_section: str = Field(description="Section/table where claim originates")

class ClaimExtractionResult(BaseModel):
    claims: list[Claim]


# ── Internal Audit ───────────────────────────────────────────────

class AuditSeverity(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    MISMATCH = "MISMATCH"

class InternalAuditResult(BaseModel):
    claim_id: str
    table_consistency: AuditSeverity
    table_detail: str | None = None
    figure_consistency: AuditSeverity
    figure_detail: str | None = None
    visual_mismatch_detail: str | None = None
    baseline_present: bool
    statistical_rigor: AuditSeverity
    statistical_detail: str | None = None
    methodology_gap: str | None = None
    mismatch_score: float = Field(ge=0.0, le=1.0, description="0=perfect, 1=total mismatch")


# ── External Evidence ────────────────────────────────────────────

class Stance(str, Enum):
    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    NEUTRAL = "NEUTRAL"

class RetrievedPaper(BaseModel):
    paper_id: str
    title: str
    year: int
    authors: list[str] = []
    doi: str | None = None
    url: str | None = None
    relevant_passage: str
    stance: Stance
    stance_confidence: float = Field(ge=0.0, le=1.0)

class StalenessEntry(BaseModel):
    baseline_name: str
    baseline_year: int
    staleness_score: float
    missed_stronger: list[dict] = Field(default_factory=list)
    verdict: str

class ExternalEvidenceResult(BaseModel):
    claim_id: str
    supporting_papers: list[RetrievedPaper] = []
    contradicting_papers: list[RetrievedPaper] = []
    neutral_papers: list[RetrievedPaper] = []
    staleness_entries: list[StalenessEntry] = []


# ── Debate ───────────────────────────────────────────────────────

class DebateRole(str, Enum):
    PROSECUTOR = "prosecutor"
    DEFENDER = "defender"

class DebateTurn(BaseModel):
    agent: DebateRole
    round_num: int
    argument: str
    evidence_cited: list[str] = Field(default_factory=list, description="DOIs, audit fields, etc.")
    new_retrieval: dict | None = Field(default=None, description="Mid-debate P-RAG query + results")
    concedes: bool = False

class DebateTranscript(BaseModel):
    claim_id: str
    turns: list[DebateTurn] = []
    total_rounds: int = 0
    terminated_reason: str = ""  # "max_rounds" | "concession" | "convergence"


# ── Final Verdict ────────────────────────────────────────────────

class VerdictLabel(str, Enum):
    STRONGLY_SUPPORTED = "STRONGLY_SUPPORTED"
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    WEAKLY_SUPPORTED = "WEAKLY_SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"

class ClaimVerdict(BaseModel):
    claim_id: str
    verdict: VerdictLabel
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str
    flags: list[str] = Field(default_factory=list)
    prosecutor_strongest: str = ""
    defender_strongest: str = ""
    unresolved: list[str] = Field(default_factory=list)

class FinalReport(BaseModel):
    paper_title: str
    authors: list[str] = []
    overall_score: float = Field(ge=0.0, le=1.0)
    overall_verdict: VerdictLabel
    total_claims: int
    claim_verdicts: list[ClaimVerdict] = []
    systemic_issues: list[str] = []
