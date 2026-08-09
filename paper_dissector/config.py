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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

# ── Role → provider mapping ─────────────────────────────────────
# Centralised so you can swap providers per agent in one place.

AGENT_CONFIG = {
    "claim_extractor":    {"provider": "gemini",  "model": GEMINI_MODEL},
    "internal_auditor":   {"provider": "gemini",  "model": GEMINI_MODEL},
    "visual_verifier":    {"provider": "gemini",  "model": GEMINI_MODEL},   # needs vision
    "evidence_hunter":    {"provider": "groq",    "model": GROQ_MODEL},
    "staleness_checker":  {"provider": "groq",    "model": GROQ_MODEL},
    "stance_classifier":  {"provider": "groq",    "model": GROQ_MODEL},   # fallback when HF NLI is unavailable
    "prosecutor":         {"provider": "groq",    "model": GROQ_MODEL},
    "defender":           {"provider": "groq",    "model": GROQ_MODEL},
    "judge":              {"provider": "gemini",  "model": GEMINI_MODEL},
}

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
# Supplying a contact address moves you to OpenAlex's faster "polite pool".
# Opt-in only - left empty we use the shared common pool.
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "").strip()


def resolve_literature_provider() -> str:
    """Which literature backend to use: 'semantic_scholar' or 'openalex'."""
    if LITERATURE_PROVIDER in ("semantic_scholar", "s2"):
        return "semantic_scholar"
    if LITERATURE_PROVIDER == "openalex":
        return "openalex"
    return "semantic_scholar" if SEMANTIC_SCHOLAR_API_KEY else "openalex"


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
AUDIT_CONTEXT_CHARS = int(os.getenv("AUDIT_CONTEXT_CHARS", "60000"))

# ── Caching ──────────────────────────────────────────────────────
CACHE_DIR = os.getenv("PAPER_DISSECTOR_CACHE", ".cache/paper_dissector")
CACHE_ENABLED = os.getenv("PAPER_DISSECTOR_CACHE_ENABLED", "1") not in ("0", "false", "False")
CACHE_TTL_SECONDS = int(os.getenv("PAPER_DISSECTOR_CACHE_TTL", str(7 * 24 * 3600)))
