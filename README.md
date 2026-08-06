# 🔬 Paper Dissector

**Multi-agent adversarial credibility analysis for research papers.**

Upload a scientific paper → agents extract claims, audit methodology, search for contradictions, debate credibility, and produce a per-claim verdict with full debate transcripts.

## Architecture

```
PDF → Docling Parser → Claim Extractor → Internal Auditor (+ VLM figure check)
    → Evidence Hunter (+ staleness detection) → Prosecutor vs Defender Debate
    → Judge → Credibility Report
```

**6 specialized agents**, each using the best free LLM for their role:

| Agent | Provider | Model |
|-------|----------|-------|
| Claim Extractor | Google AI Studio | Gemini 2.5 Flash |
| Internal Auditor | Google AI Studio | Gemini 2.5 Flash |
| Visual Verifier | Google AI Studio | Gemini 2.5 Flash (vision) |
| Evidence Hunter | Groq | Llama 3.3 70B |
| Prosecutor | Groq | Llama 3.3 70B |
| Defender | Groq | Llama 3.3 70B |
| Judge | Google AI Studio | Gemini 2.5 Flash |

**External tools:** Semantic Scholar API, HuggingFace Inference (DeBERTa-v3 for stance classification)

## Key Differentiators

- **Dual-track auditing** — checks the paper against itself AND external literature
- **Multimodal grounding** — VLM reads figures/charts to catch visual-textual mismatches
- **Temporal staleness detection** — flags cherry-picked outdated baselines
- **Progressive RAG in debate** — agents fire targeted searches mid-argument
- **Schema-constrained claims** — structured JSON, not vague text summaries

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
| `GEMINI_API_KEY` | **Required** — extractor, auditor, visual verifier, judge | https://aistudio.google.com/apikey |
| `GROQ_API_KEY` | **Required** — evidence hunter, prosecutor, defender | https://console.groq.com/keys |
| `SEMANTIC_SCHOLAR_API_KEY` | **Strongly recommended** | https://www.semanticscholar.org/product/api |
| `HF_API_KEY` | Optional | https://huggingface.co/settings/tokens |

Two things worth knowing before you fill these in:

- **Semantic Scholar without a key is effectively unusable.** The unauthenticated
  `/paper/search` pool is shared across all users and returns HTTP 429 on nearly
  every call. The pipeline degrades gracefully — a circuit breaker stops retrying
  and the stage reports no papers found rather than crashing — but you lose the
  external-evidence half of the analysis.
- **Do not put a trailing `# comment` on the same line as a value in `.env`.**
  `python-dotenv` keeps it as part of the value, so the key becomes the literal
  comment text and you get confusing auth errors instead of a clear one.

`HF_API_KEY` is genuinely optional: if HuggingFace inference is missing, rate
limited, or returns an unexpected payload, stance classification falls back to a
batched LLM call automatically.

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

### Tuning

Every setting below is an environment variable with a sensible default. These
are the ones that drive runtime and free-tier quota consumption:

| Variable | Default | Effect |
|----------|---------|--------|
| `MAX_CLAIMS` | 8 | Claims analysed per paper. The main cost/time dial. |
| `MAX_DEBATE_ROUNDS` | 4 | Prosecutor/defender exchanges per claim. |
| `MAX_PRAG_RETRIEVALS` | 2 | Mid-debate searches each agent may fire. |
| `AUDIT_CONTEXT_CHARS` | 60000 | Paper text sent to the auditor per claim. |
| `CONVERGENCE_THRESHOLD` | 0.85 | Similarity at which a circling debate stops early. |

External API responses are cached on disk under `.cache/`, so repeated runs
during development do not re-hit rate limits. Clear it with:

```bash
python -c "from paper_dissector import cache; cache.clear()"
```

### Resilience

The pipeline is built to finish on flaky free tiers rather than crash:

- Every external call retries with exponential backoff (tenacity).
- Semantic Scholar has a circuit breaker — after repeated failures it stops
  calling out for a cooldown instead of burning minutes on retries.
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
│       ├── semantic_scholar.py     # Semantic Scholar API client
│       └── stance_classifier.py    # HuggingFace DeBERTa-v3 NLI
```

## Evaluation

Benchmark against **SciFact** (1,409 expert-annotated claims). Run ablations:
- Full pipeline vs no internal audit
- Full pipeline vs no staleness detection  
- Full pipeline vs no multimodal grounding
- Full pipeline vs single-agent (no debate)
- Progressive RAG vs static retrieval

## License

MIT
