import os
from functools import lru_cache

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class MissingAPIKey(RuntimeError):
    """A provider was selected but its API key is not set in the environment."""


def _require(key: str, provider: str, url: str) -> str:
    value = os.getenv(key, "").strip()
    # python-dotenv keeps trailing "# comment" text as part of an unquoted value,
    # so a placeholder line reads back as a comment rather than as empty.
    if value.startswith("#"):
        value = ""
    if not value:
        raise MissingAPIKey(
            f"{key} is not set, but an agent is configured to use {provider}. "
            f"Add it to your .env file - get a free key at {url}"
        )
    return value

# ── Provider clients (all OpenAI-compatible) ─────────────────────

# Cached: the pipeline makes hundreds of calls and each OpenAI() instance
# carries its own connection pool.

@lru_cache(maxsize=1)
def get_gemini_client() -> OpenAI:
    """Gemini 2.5 Flash — best free reasoning + vision. 500 req/day."""
    return OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=_require("GEMINI_API_KEY", "Gemini", "https://aistudio.google.com/apikey"),
    )

@lru_cache(maxsize=1)
def get_groq_client() -> OpenAI:
    """Groq Llama 3.3 70B — fast inference. 14,400 req/day."""
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=_require("GROQ_API_KEY", "Groq", "https://console.groq.com/keys"),
    )

@lru_cache(maxsize=1)
def get_mistral_client() -> OpenAI:
    """Mistral Small — backup. 1B tokens/month free."""
    return OpenAI(
        base_url="https://api.mistral.ai/v1",
        api_key=_require("MISTRAL_API_KEY", "Mistral", "https://console.mistral.ai/api-keys"),
    )

# ── Model names per provider ─────────────────────────────────────

# Overridable via .env so a quota-exhausted model can be swapped without a code change.
# Verified against the live /models list and probed for chat, JSON mode and vision:
#   gemini-3.6-flash  ~2s, JSON mode + vision both work.
#   gemini-3.8-flash  15-30x slower and its JSON mode times out. Avoid.
# (gemini-2.5-flash still appears in /models but 404s for new accounts.)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
# Vision is the one capability Groq's catalogue has no model for, so the visual
# verifier stays on Gemini. Its own model name is separate because the free-tier
# request cap is per model, so giving vision a distinct model keeps it from
# competing with any other Gemini role for the same 20/day allowance.
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", GEMINI_MODEL)

# Groq splits by capability rather than one model for everything:
#   gpt-oss-120b  strongest reasoning, ~0.6s, but returns 400 on JSON mode.
#   qwen3.8-27b   native JSON mode, ~0.3s.
# So prose-generating roles get the former and JSON-returning roles the latter.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_JSON_MODEL = os.getenv("GROQ_JSON_MODEL", "qwen/qwen3.8-27b")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

# ── Per-request input budgets ────────────────────────────────────
# Measured from live rate-limit headers. Groq's free tier allows 8000 tokens
# per minute and rejects any single request above that with HTTP 413 (not a
# throttle - an outright failure), so requests must be kept under it.
# Gemini has no comparable per-request ceiling but only 20 requests per day
# per model, so it suits one big call rather than many small ones.
# 5200 not 7000: the ceiling differs per model (qwen3.8-27b allows 7000 input
# tokens where gpt-oss-120b allows 8000) and the response counts toward the
# same per-minute budget, so leave real headroom. Requests that still come back
# too large are shrunk and retried by llm._create_fitting.
INPUT_TOKEN_BUDGET = {
    "groq": int(os.getenv("GROQ_INPUT_TOKEN_BUDGET", "5200")),
    "gemini": int(os.getenv("GEMINI_INPUT_TOKEN_BUDGET", "200000")),
    "mistral": int(os.getenv("MISTRAL_INPUT_TOKEN_BUDGET", "24000")),
}


def provider_for_agent(agent_name: str) -> str:
    return AGENT_CONFIG[agent_name]["provider"]


def input_token_budget(provider: str) -> int:
    return INPUT_TOKEN_BUDGET.get(provider, 0)

# ── Role → provider mapping ─────────────────────────────────────
# Centralised so you can swap providers per agent in one place.

# Gemini's free tier allows only 20 requests per day per model, which cannot
# support a role invoked once per claim. Everything therefore runs on Groq
# (much higher free limits) except visual verification, which needs vision.
AGENT_CONFIG = {
    # Reasoning-heavy roles on the strongest Groq model. It rejects native JSON
    # mode, but llm.chat_json detects that once and switches to prompt-based
    # JSON, which it produces reliably.
    # Claim extraction is the one call that must see the entire paper (~12k
    # tokens), which exceeds Groq's per-request ceiling. It runs once per paper,
    # so Gemini's 20-requests-per-day allowance is ample.
    "claim_extractor":    {"provider": "gemini",  "model": GEMINI_MODEL},
    # Used when the primary extractor is unavailable (Gemini's 20/day exhausted
    # or the endpoint down): the paper is chunked to fit Groq's request ceiling.
    "claim_extractor_fallback": {"provider": "groq", "model": GROQ_MODEL},
    "internal_auditor":   {"provider": "groq",    "model": GROQ_MODEL},
    "prosecutor":         {"provider": "groq",    "model": GROQ_MODEL},
    "defender":           {"provider": "groq",    "model": GROQ_MODEL},
    # Lighter structured roles on the Groq model with working native JSON mode.
    "evidence_hunter":    {"provider": "groq",    "model": GROQ_JSON_MODEL},
    "staleness_checker":  {"provider": "groq",    "model": GROQ_JSON_MODEL},
    "stance_classifier":  {"provider": "groq",    "model": GROQ_JSON_MODEL},  # fallback when HF NLI is unavailable
    # Adjudication is the hardest reasoning task in the pipeline: on the
    # smaller JSON-mode model the judge confused the paper under analysis with
    # an external source and invented baseline numbers. The stronger model has
    # no native JSON mode, but llm.chat_json falls back reliably.
    "judge":              {"provider": "groq",    "model": GROQ_MODEL},
    # The only role Groq cannot serve.
    "visual_verifier":    {"provider": "gemini",  "model": GEMINI_VISION_MODEL},
}

# Per-role overrides, e.g. PD_AGENT_JUDGE_PROVIDER=gemini PD_AGENT_JUDGE_MODEL=gemini-3.6-flash
# Keeps the mapping centralised here while letting a single role be re-pointed
# without editing code.
for _role, _cfg in AGENT_CONFIG.items():
    _p = os.getenv(f"PD_AGENT_{_role.upper()}_PROVIDER", "").strip().lower()
    _m = os.getenv(f"PD_AGENT_{_role.upper()}_MODEL", "").strip()
    if _p:
        _cfg["provider"] = _p
    if _m:
        _cfg["model"] = _m

def get_client_for_agent(agent_name: str) -> tuple[OpenAI, str]:
    """Return (client, model_name) for a given agent role."""
    cfg = AGENT_CONFIG[agent_name]
    provider = cfg["provider"]
    model = cfg["model"]
    factory = {
        "gemini": get_gemini_client,
        "groq": get_groq_client,
        "mistral": get_mistral_client,
    }
    return factory[provider](), model

# ── External APIs ────────────────────────────────────────────────

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"

# ── Literature source ────────────────────────────────────────────
# "auto" uses Semantic Scholar when a key is present and OpenAlex otherwise.
# OpenAlex needs no key; Semantic Scholar's unauthenticated search pool is
# shared across all users and returns 429 on essentially every call.
LITERATURE_PROVIDER = os.getenv("LITERATURE_PROVIDER", "auto").strip().lower()

OPENALEX_BASE = os.getenv("OPENALEX_BASE", "https://api.openalex.org")
# Free and self-serve at https://openalex.org/rest-api. Worth having: OpenAlex
# periodically pauses ANONYMOUS search under load ("Anonymous search is paused
# while the search cluster recovers"), and a key is exempt from that.
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "").strip()
# Supplying a contact address moves you to OpenAlex's faster "polite pool".
# Opt-in only - left empty we use the shared common pool.
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "").strip()


def resolve_literature_provider() -> str:
    """Which literature backend to prefer: 'semantic_scholar', 'openalex' or 'arxiv'."""
    if LITERATURE_PROVIDER in ("semantic_scholar", "s2"):
        return "semantic_scholar"
    if LITERATURE_PROVIDER in ("openalex", "arxiv"):
        return LITERATURE_PROVIDER
    return "semantic_scholar" if SEMANTIC_SCHOLAR_API_KEY else "openalex"


def literature_chain() -> list[str]:
    """
    Backends to try in order, best first.

    Only backends that can actually authenticate are included, and the keyless
    ones always trail the preferred choice so an outage at one source does not
    zero the evidence stage.
    """
    preferred = resolve_literature_provider()
    chain = [preferred]
    for backend in ("openalex", "arxiv", "semantic_scholar"):
        if backend in chain:
            continue
        if backend == "semantic_scholar" and not SEMANTIC_SCHOLAR_API_KEY:
            continue  # unauthenticated S2 search 429s on essentially every call
        chain.append(backend)
    return chain


HF_API_KEY = os.getenv("HF_API_KEY", "")
# The legacy api-inference.huggingface.co host is being retired in favour of the
# router; override via .env if your account still uses the old endpoint.
HF_INFERENCE_BASE = os.getenv("HF_INFERENCE_BASE", "https://router.huggingface.co/hf-inference")
HF_ENABLED = os.getenv("HF_ENABLED", "1") not in ("0", "false", "False")

# ── Debate settings ──────────────────────────────────────────────

MAX_DEBATE_ROUNDS = int(os.getenv("MAX_DEBATE_ROUNDS", "4"))
MAX_PRAG_RETRIEVALS = int(os.getenv("MAX_PRAG_RETRIEVALS", "2"))   # P-RAG budget per agent per claim
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.85"))  # consecutive same-role args → stop

# ── Pipeline limits ──────────────────────────────────────────────
# Keep free-tier request counts sane: each claim costs ~1 audit + 1 vision
# + 1 query-gen + N stance + 2*rounds debate + 1 judge call.
MAX_CLAIMS = int(os.getenv("MAX_CLAIMS", "8"))
MAX_FIGURES_PER_CLAIM = int(os.getenv("MAX_FIGURES_PER_CLAIM", "1"))
# Audit excerpts, not the whole paper: internal_auditor.build_audit_excerpt
# selects the claim's own section plus the tables/figures it cites. ~12k chars
# is about 3k tokens, which fits Groq's per-request ceiling with room for the
# system prompt and the response.
AUDIT_CONTEXT_CHARS = int(os.getenv("AUDIT_CONTEXT_CHARS", "12000"))

# Claim extraction is a separate budget: it runs on a large-context provider so
# that it sees the entire paper. Reusing the auditor's excerpt budget here would
# silently hide three quarters of the paper from the extractor.
EXTRACTION_CONTEXT_CHARS = int(os.getenv("EXTRACTION_CONTEXT_CHARS", "400000"))
# Chunk size for the fallback path, which extracts on a small-context provider.
EXTRACTION_CHUNK_CHARS = int(os.getenv("EXTRACTION_CHUNK_CHARS", "18000"))

# ── Caching ──────────────────────────────────────────────────────
CACHE_DIR = os.getenv("PAPER_DISSECTOR_CACHE", ".cache/paper_dissector")
CACHE_ENABLED = os.getenv("PAPER_DISSECTOR_CACHE_ENABLED", "1") not in ("0", "false", "False")
CACHE_TTL_SECONDS = int(os.getenv("PAPER_DISSECTOR_CACHE_TTL", str(7 * 24 * 3600)))
