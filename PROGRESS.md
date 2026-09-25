# Progress log

Last updated: 2026-09-25. Work is on the `ollama` branch (commit `d9b4903`, pushed). `main` is unchanged.

## Where things stand

- The pipeline runs end to end on a real paper (GELU, arXiv 1606.08415) without crashing.
- 77 tests pass (`python -m pytest tests/ -q`).
- Demo: `PD_ANALYSIS_JSON=demo/gelu_analysis.json streamlit run app.py` loads a real saved run.
- **Not yet shown live:** a VLM figure reading inside a pipeline run. The figure calls now fire, but Gemini's daily quota was used up when we tested (see "Next steps").

## 1. Audit (before fixes)

Every module was implemented, none stubbed. All 23 modules imported cleanly, and the 52 existing tests passed. A live run on GELU finished, but it found these problems:

| # | Problem | Status |
|---|---|---|
| 1 | VLM never ran: claims cite "Section 3.3", not "Figure N", so no figure matched | Fixed |
| 2 | Text-only auditor wrote the "VLM reading" field; judge raised `VISUAL_MISMATCH` with no image read | Fixed |
| 3 | Any partial `CONCEDE:` ended the debate, so most debates were 1 round | Fixed |
| 4 | Gemini per-minute 429 misread as daily cap, never retried | Fixed |
| 5 | Judge flags not backed by evidence (stale baseline when check said CURRENT; "not in a table" counted as FAIL) | Fixed |
| 6 | SPECTER2 not implemented anywhere | **Open** |
| 7 | Hugging Face is only an NLI stance endpoint, not an LLM provider | **Open** |
| 8 | Failed stance classifications cached as NEUTRAL for 7 days | Fixed |
| 9 | Stale model names in `.env.example`, config comments, README | Fixed |

Resume claims checked against the code:

| Claim | Verdict |
|---|---|
| 7-agent LangGraph workflow with structured debate | PARTIAL: 6 linear graph nodes; debate is a loop inside one node (now runs its full rounds) |
| Semantic Scholar/OpenAlex, SPECTER2, VLM cross-checks figures | PARTIAL: OpenAlex/arXiv work; S2 needs a key; **SPECTER2 missing**; VLM now fires but no live reading yet |
| Provider-agnostic across Gemini, Groq, Hugging Face | PARTIAL: swapping per agent works for Gemini, Groq, Mistral and now Ollama; **not Hugging Face** |
| Streamlit interface for live demos | TRUE |

## 2. Fixes made (all in `d9b4903`)

- **Figures:** `match_figures_for_claim` falls back to caption matching on dataset/task names and reported numbers. Method names (e.g. "GELU") don't count. A warning is logged when a claim matches nothing. The extractor prompt now asks for figure/table references.
- **VLM field:** only `_verify_figure` can write `visual_mismatch_detail`. Figure flags and the "figure(s) do not support" report line require a real VLM reading.
- **Debate:** only `CONCEDE_CLAIM:` ends it, and never before round 2. The repetition stop is kept.
- **Rate limits:** Gemini `quotaId` decides per-minute (retry after 60s) vs per-day (give up). Groq `retry-after` is honoured. Rate limits get 8 attempts, other errors 4.
- **SDK retries off (`max_retries=0`):** the OpenAI SDK was retrying inside our retries, tripling requests. That's what burned Gemini's 5/min and 20/day quota (six 503s, then a 429).
- **Judge grounding (`judge.ground_flags`):** each flag is checked against the stage output behind it. A number only in the text, with no table covering it, is no longer a table failure. Prompts now say PASS means "matches the paper's tables *or* text".
- **Stance cache:** only successful classifications are cached.
- **Ollama provider:** `PD_AGENT_<ROLE>_PROVIDER=ollama`, base URL `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`), budget `OLLAMA_INPUT_TOKEN_BUDGET` (default 9000). An unknown provider now raises a clear error.
- **Tests:** `tests/test_real_run.py` (25 tests) is driven by real output frozen in `tests/fixtures/` (the pre-fix GELU analysis, the parsed markdown, the real Gemini 429 body).

## 3. Live runs on GELU

| Run | Setup | Result |
|---|---|---|
| Before fixes | Groq + Gemini, 3 claims × 3 rounds | 484s. 0 VLM calls. Debates 1/1/3 rounds. Made-up `VISUAL_MISMATCH`. Score 0.45 |
| After fixes (saved as `demo/gelu_analysis.json`) | Groq + Gemini, 3 claims × 3 rounds | 774s. 3 VLM calls fired (Figures 7, 5, 6), all refused by Gemini's **daily** cap. Debates 3/3/3. Only `NO_STATISTICAL_TESTS` kept (backed by audit). `FIGURE_MISMATCH` dropped twice. Score 0.673 SUPPORTED |
| Ollama | All text roles on local `qwen2.5` 7B, 1 claim × 2 rounds | 317s, no crashes. **Quality poor:** confused which paper it judged, used `CONCEDE_CLAIM` backwards, baseline check copied "N stronger baselines" from the prompt |

The SDK-retry, Groq retry-after and prompt-wording changes were added after the second run: covered by tests, not yet run live with Groq.

## 4. Ollama notes

- Installed models: `qwen3:8b`, `qwen2.5:7b`, `qwen3:4b`, `llama3.2`, vision `gemma3:4b`, `gemma4:12b`. GPU: RTX 4050, 6 GB.
- **Default context is 4096 and Ollama silently truncates** (a 7.8k-token prompt became 2050). The per-request `num_ctx` setting is ignored on the OpenAI endpoint. I created `pd-qwen2.5-7b-12k` (`qwen2.5:7b` + `num_ctx 12288`); remove it with `ollama rm pd-qwen2.5-7b-12k`.
- `gemma3:4b` is **not usable as the VLM**: it invented error bars on GELU Figure 6 (checked against the image). `gemma4:12b` is untested; it won't fully fit in 6 GB.
- `qwen3` puts its thinking in a separate field (the parser is fine), but it's slower and can use up `max_tokens`.
- Don't run games (e.g. Valorant) during a local run: VRAM contention slows Ollama to a crawl.
- Verdict: use Ollama for free pipeline testing, not for credible verdicts.

To run all text roles locally:

```bash
for r in CLAIM_EXTRACTOR CLAIM_EXTRACTOR_FALLBACK INTERNAL_AUDITOR PROSECUTOR DEFENDER \
         EVIDENCE_HUNTER STALENESS_CHECKER STANCE_CLASSIFIER JUDGE; do
  export PD_AGENT_${r}_PROVIDER=ollama PD_AGENT_${r}_MODEL=pd-qwen2.5-7b-12k
done
MAX_CLAIMS=1 MAX_DEBATE_ROUNDS=2 streamlit run app.py
```

## 5. Next steps (ranked)

1. **Get a live VLM reading.** Rerun after Gemini's daily quota resets (midnight Pacific), or give vision its own quota with `PD_AGENT_VISUAL_VERIFIER_MODEL=gemini-3.7-flash`. Then re-save `demo/gelu_analysis.json`. This also verifies the retry changes live. *Small.*
2. **Stale-baseline grounding gap:** `ground_flags` accepts any verdict starting with "STALE", even the template text "STALE — N stronger baselines" with no papers. It should also require `missed_stronger` to be non-empty. *Small.*
3. Check the C1 → Figure 7 (CIFAR-100) caption match from the second run; it looks wrong. *Small.*
4. SPECTER2 retrieval, or drop it from the resume. *Medium, or trivial.*
5. Hugging Face as an LLM provider (its router is OpenAI-compatible, so it's the same pattern as Ollama), or drop it from the resume. *Small.*
6. Author extraction returns `[]` for GELU. *Small.*
7. Merge `ollama` into `main` (open a PR).

## Environment

- venv: `C:\Users\prana\pdvenv` (on C: because of the Windows DLL policy on D:).
- Keys set in `.env`: Gemini, Groq, OpenAlex. Empty: Hugging Face, Semantic Scholar.
- Free-tier limits: Gemini 20 requests/day and 5/min **per model**; Groq 8000 tokens/min and 200k tokens/day (roughly two full 3-claim runs per day).
- git has no global identity set here; commits used `Pranav Negi <143416593+825pranav@users.noreply.github.com>`.
