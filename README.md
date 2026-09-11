# 🔬 Paper Dissector

**Multi-agent adversarial credibility analysis for research papers.**

Upload a scientific paper → agents extract claims, audit methodology, search for contradictions, debate credibility, and produce a per-claim verdict with full debate transcripts.

## Architecture

```
PDF → Docling Parser → Claim Extractor → Internal Auditor (+ VLM figure check)
    → Evidence Hunter (+ staleness detection) → Prosecutor vs Defender Debate
    → Judge → Credibility Report
```

**Nine agent roles**, each mapped to the free model that suits it — chosen against measured free-tier limits rather than model prestige:

| Agent | Provider | Model | Why |
|-------|----------|-------|-----|
| Claim Extractor | Google AI Studio | Gemini 3.6 Flash | Only call that must see the whole paper |
| Internal Auditor | Groq | gpt-oss-120b | Reasoning over targeted excerpts |
| Visual Verifier | Google AI Studio | Gemini 3.6 Flash (vision) | Groq has no vision model |
| Evidence Hunter | Groq | qwen3.8-27b | Native JSON mode |
| Prosecutor | Groq | gpt-oss-120b | Strongest available reasoning |
| Defender | Groq | gpt-oss-120b | Same model as prosecutor, so the debate is symmetric |
| Judge | Groq | qwen3.8-27b | Different family from the debaters, so it is not grading itself |

The mapping lives in one place, `AGENT_CONFIG` in `config.py`. Any single role
can be re-pointed without touching code:

```bash
PD_AGENT_JUDGE_PROVIDER=gemini PD_AGENT_JUDGE_MODEL=gemini-3.6-flash streamlit run app.py
```

### Free-tier limits that shape this design

These were measured against the live APIs, not taken from documentation, and
they are the reason the roles are split the way they are:

| Provider | Limit | Consequence |
|----------|-------|-------------|
| Gemini | **20 requests/day, per model** | Cannot serve a per-claim role. Used for the single whole-paper extraction call and for vision. |
| Gemini | large context, no per-request cap | Ideal for the one call that needs all 12k tokens of the paper. |
| Groq | **~7-8k tokens per request** | Varies per model: qwen3.8-27b allows 7000 input tokens, gpt-oss-120b 8000. Above it the request is rejected with HTTP 413, not throttled. |
| Groq | **8000 tokens/minute** | Sets the pace of a run. Everything is queueing on this, not on model latency. |
| Groq | **200,000 tokens/day** | The real ceiling on how much you can analyse per day: roughly 16-20 claims total across all runs. |
| Groq | 1000 requests/day | Plenty; request count is never the binding constraint. |

Two consequences worth knowing:

- **The auditor reads excerpts, not the paper.** `build_audit_excerpt` selects
  the claim's own section, the tables and figures it cites, and the opening —
  scoring blocks by relevance and keeping them in document order. On a 15-page
  paper that is 49k chars down to about 12k. Sending the full text per claim
  would exceed Groq's per-request ceiling outright.
- **Every request is size-guarded.** `llm._fit_to_budget` estimates tokens and
  trims the largest user message (never the system prompt) before sending, so an
  unexpectedly long paper degrades to a truncated request instead of a failed
  stage.

Gemini's daily cap is per *model*, so if you exhaust one you can point a role at
another (`gemini-3.5-flash`, `gemini-3.7-flash`, …), each with its own
allowance. Note that `gemini-3.8-flash` is 15-30x slower and its JSON mode times
out; `gemini-3.6-flash` is the one to use.

**External tools:** OpenAlex / arXiv / Semantic Scholar for literature, HuggingFace Inference (DeBERTa-v3) for stance classification with a batched LLM fallback.

## Key Differentiators

- **Dual-track auditing** — checks the paper against itself AND external literature
- **Multimodal grounding** — the VLM reads the figure or table a claim actually
  cites, matched by its parsed number, to catch visual-textual mismatches
- **Temporal staleness detection** — flags cherry-picked outdated baselines
- **Progressive RAG in debate** — agents fire targeted searches mid-argument and
  argue from what they retrieve
- **Schema-constrained claims** — structured JSON, not vague text summaries
- **No self-corroboration** — searching a paper's own claims reliably retrieves
  that paper, so it is excluded from its own evidence. Without this, every claim
  gained exactly one supporting citation: itself.

## Setup

```bash
git clone <your-repo-url>
cd paper-dissector

python -m venv venv
venv/Scripts/activate        # Windows;  source venv/bin/activate on macOS/Linux
pip install -r requirements.txt

cp .env.example .env         # then fill in your keys
```

Docling pulls in PyTorch, so the first install is a multi-GB download, and the
first PDF parse downloads ~500 MB of layout models into `~/.cache/huggingface`.
Parsing a 15-page paper takes ~25 s on CPU after that.

### API keys

| Key | Needed? | Get it |
|-----|---------|--------|
| `GEMINI_API_KEY` | **Required** — claim extraction and figure reading | https://aistudio.google.com/apikey |
| `GROQ_API_KEY` | **Required** — every other role | https://console.groq.com/keys |
| `OPENALEX_API_KEY` | Recommended, free and instant | https://openalex.org/rest-api |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional | https://www.semanticscholar.org/product/api |
| `HF_API_KEY` | Optional | https://huggingface.co/settings/tokens |

Only the two LLM keys are required. Both are self-serve and free.

**Do not put a trailing `# comment` on the same line as a value in `.env`.**
`python-dotenv` keeps it as part of the value, so the key becomes the literal
comment text and you get confusing auth errors instead of a clear one.

`HF_API_KEY` is optional: if HuggingFace inference is missing, rate limited, or
returns an unexpected payload, stance classification falls back to a batched LLM
call automatically.

### Literature backend

Literature search goes through `tools/literature.py`, which picks a backend
from `LITERATURE_PROVIDER`:

| Value | Behaviour |
|-------|-----------|
| `auto` (default) | Semantic Scholar if `SEMANTIC_SCHOLAR_API_KEY` is set, else OpenAlex |
| `openalex` | Prefer OpenAlex |
| `arxiv` | Prefer arXiv |
| `semantic_scholar` | Prefer Semantic Scholar |

Whatever the preference, the remaining keyless backends are tried in turn when the preferred one returns nothing, because none of the three is reliably up on its own.

Each of the three fails in a different way, which is why the chain exists:

- **Semantic Scholar** — unauthenticated `/paper/search` returns HTTP 429 on
  essentially every call. A key fixes it but is granted by application, not
  instantly.
- **OpenAlex** — no key needed, good metadata, but it periodically pauses
  *anonymous* search under load (HTTP 503, "Anonymous search is paused while
  the search cluster recovers"). A **free, instant, self-serve API key** at
  https://openalex.org/rest-api is exempt — set `OPENALEX_API_KEY`.
- **arXiv** — always up and keyless, but preprints only and no citation
  counts, so it is tried last.

All three return records in the same shape, so nothing downstream changes.

Set `OPENALEX_MAILTO` to a contact address to use OpenAlex's faster "polite
pool". It is opt-in and left empty by default — nothing is sent unless you set it.

Two OpenAlex quirks the code compensates for, both found in testing:

- It stores several records per paper, including reindexed preprints dated years
  after the original. Title lookup therefore takes the *earliest* matching
  record, and a catalogue year later than the newest year in the paper's own
  text is rejected in favour of that bound.
- Duplicate records share a DOI but differ by id, so retrieved papers are
  deduplicated on DOI rather than the provider's id.

### Windows note

Some Windows 11 machines run an Application Control policy that refuses to load
unsigned DLLs from secondary drives, so `import torch` fails with `WinError 4551`
when the venv lives on, say, `D:`. The venv does not have to sit next to the
code — create it under your user directory on `C:` and call it directly:

```bash
python -m venv C:/Users/<you>/pdvenv
C:/Users/<you>/pdvenv/Scripts/python.exe -m pip install -r requirements.txt
C:/Users/<you>/pdvenv/Scripts/python.exe -m streamlit run app.py
```

Keep that path short — Windows long-path support is off by default and some
dependencies exceed the 260-character limit.

## Run

```bash
# Streamlit UI — per-stage progress, full debate transcripts, JSON export
streamlit run app.py

# Or programmatic
python -c "
from paper_dissector.graph import run_pipeline
result = run_pipeline('path/to/paper.pdf')
print(result['final_report'].model_dump_json(indent=2))
"
```

### Saving and reopening an analysis

A run costs several minutes of rate-limited calls, so the result is worth
keeping. The report view offers **Download full analysis (JSON)** — audits,
evidence, debate transcripts and verdicts — and the sidebar takes that file
back to redisplay everything without spending quota again.

Point the app at one directly for a demo:

```bash
PD_ANALYSIS_JSON=analysis.json streamlit run app.py
```

Programmatically:

```python
from paper_dissector.report_io import save_analysis, load_analysis
save_analysis(result, "analysis.json")
state = load_analysis("analysis.json")
```

### Tuning

Every setting below is an environment variable with a sensible default. These
are the ones that drive runtime and free-tier quota consumption:

| Variable | Default | Effect |
|----------|---------|--------|
| `MAX_CLAIMS` | 8 | Claims analysed per paper. The main cost/time dial. |
| `MAX_DEBATE_ROUNDS` | 4 | Prosecutor/defender exchanges per claim. |
| `MAX_PRAG_RETRIEVALS` | 2 | Mid-debate searches each agent may fire. |
| `AUDIT_CONTEXT_CHARS` | 12000 | Size of the excerpt built for each claim's audit. |
| `EXTRACTION_CONTEXT_CHARS` | 400000 | Ceiling on the paper text sent to the extractor. |
| `EXTRACTION_CHUNK_CHARS` | 18000 | Chunk size for the fallback (small-context) extractor. |
| `CONVERGENCE_THRESHOLD` | 0.85 | Similarity at which a circling debate stops early. |
| `GROQ_INPUT_TOKEN_BUDGET` | 6500 | Per-request ceiling enforced before sending to Groq. |

**Runtime is bounded by Groq's per-minute token ceiling, not by model speed.**
The models answer in 0.3-2s; everything else is rate-limit queueing. A measured
run on a 15-page paper, 4 claims at 3 debate rounds, took 18 minutes:

| Stage | Time | Note |
|-------|------|------|
| ingest | 44s | Docling, CPU |
| extract_claims | 147s | via the chunked fallback; ~20s when Gemini quota is available |
| internal_audit | 109s | 27s per claim |
| external_evidence | 240s | 60s per claim, including stance classification |
| debate | 488s | 122s per claim — the dominant cost, and it grows with rounds |
| adjudicate | 64s | |

A later run of 3 claims at 2 rounds completed in **382s**, with claim
extraction on Gemini rather than the chunked fallback.

Debate cost scales with `MAX_DEBATE_ROUNDS` times `MAX_CLAIMS`, and each round
carries the transcript so far, so context grows within a claim. For a quick
demonstration use `MAX_CLAIMS=2 MAX_DEBATE_ROUNDS=2`. A paid Groq tier removes
the ceiling; nothing about the code changes.

Exhausting a daily quota mid-run is handled, not fatal: a debate turn that hits
it records the error on the transcript and the claim is still adjudicated on
whatever was gathered. Daily caps are detected and not retried, since a
per-day limit will not clear within a backoff window.

External API responses are cached on disk under `.cache/`, so repeated runs
during development do not re-hit rate limits. Clear it with:

```bash
python -c "from paper_dissector import cache; cache.clear()"
```

### Credibility is not confidence

Each verdict carries two numbers, and conflating them inverts the result:

- **`credibility`** — how well supported the claim is, implied by the verdict
  label. This is what the paper-level `overall_score` averages.
- **`confidence`** — how certain the judge is of that verdict. A claim can be
  *confidently* NOT_SUPPORTED.

The scaffold averaged confidence. In a real run the judge returned two claims
as NOT_SUPPORTED with confidence 0.95 and one as SUPPORTED with 0.75, and the
paper scored **0.883, "STRONGLY_SUPPORTED"** — two thirds of its claims had been
rejected and the headline number said the opposite. Averaging credibility gives
0.317, "WEAKLY_SUPPORTED".

The same distinction applies in the UI: "Well-supported Claims" counts verdict
labels, not confidence scores.

### Resilience

The pipeline is built to finish on flaky free tiers rather than crash:

- Every external call retries with exponential backoff (tenacity).
- Each literature backend has a circuit breaker — after repeated failures it
  stops calling out for a cooldown instead of burning minutes on retries.
  Measured: six rate-limited queries take ~26s instead of ~200s.
- A failure on one claim degrades that claim only; the remaining claims continue.
- If a provider rejects `response_format={"type": "json_object"}`, the LLM layer
  detects it once and switches that provider to prompt-based JSON with tolerant
  parsing (code-fence stripping, balanced-brace scan, trailing-comma repair).
- Loose LLM JSON is coerced into schema-valid values rather than raising, so an
  out-of-range score or an off-menu enum label does not drop a claim.

## Project Structure

```
paper-dissector/
├── app.py                          # Streamlit frontend
├── requirements.txt
├── .env.example
├── paper_dissector/
│   ├── config.py                   # LLM provider configs + agent→provider mapping
│   ├── schemas.py                  # Pydantic models (claims, evidence, verdicts)
│   ├── state.py                    # LangGraph state definition
│   ├── graph.py                    # Pipeline state machine
│   ├── llm.py                      # Shared chat helpers: JSON-mode fallback + retries
│   ├── cache.py                    # Disk cache for external API responses
│   ├── sanitize.py                 # Coerces loose LLM JSON into schema-valid values
│   ├── agents/
│   │   ├── claim_extractor.py      # Stage 2: structured claim extraction
│   │   ├── internal_auditor.py     # Stage 3: methodology audit + VLM figure check
│   │   ├── evidence_hunter.py      # Stage 4: literature search + staleness detection
│   │   ├── debate.py               # Stage 5: prosecutor vs defender + progressive RAG
│   │   └── judge.py                # Stage 6: verdict adjudication
│   └── tools/
│       ├── pdf_parser.py           # Docling PDF → markdown + figures
│       ├── literature.py           # Backend-agnostic literature search
│       ├── openalex.py             # OpenAlex client (keyless, the default)
│       ├── semantic_scholar.py     # Semantic Scholar client
│       ├── _http.py                # Shared throttle/retry/cache/circuit breaker
│       └── stance_classifier.py    # HuggingFace DeBERTa-v3 NLI + LLM fallback
└── tests/
    └── test_offline.py             # Regression checks needing no keys or network
```

## Tests

```bash
python -m pytest tests/ -q
# or individually:
python tests/test_offline.py        # pure logic
python tests/test_report_view.py    # renders the Streamlit UI headlessly
```

`test_offline.py` covers JSON coercion, claim dedup, figure matching,
convergence calibration, year resolution, self-citation exclusion, request
sizing and analysis round-tripping.

`test_report_view.py` drives the Streamlit app against a saved analysis and
asserts that every section actually renders — the executive summary, the audit
severities, the VLM's figure reading, supporting papers, the staleness warning,
the debate transcript, the mid-debate search and the concession badge. Before
this existed the UI could only be checked by eye in a browser.

Neither needs API keys or network access.

## Evaluation

Benchmark against **SciFact** (1,409 expert-annotated claims). Run ablations:
- Full pipeline vs no internal audit
- Full pipeline vs no staleness detection  
- Full pipeline vs no multimodal grounding
- Full pipeline vs single-agent (no debate)
- Progressive RAG vs static retrieval

## License

MIT
