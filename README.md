# Research Agent for CS Literature Review

An MVP Agentic RAG assistant that plans a CS literature review, searches arXiv, indexes paper evidence in Chroma, verifies cited claims, and writes a Markdown report.

## Architecture

```text
Question
  -> LangGraph planner
  -> structured search intents
  -> arXiv query adapter and search
  -> title deduplication and paper reranking
  -> paper ingestion and chunking
  -> Chroma retrieval
  -> claim reader
  -> claim verifier
  -> report writer
  -> Markdown report + SQLite trace + LangGraph checkpoints
```

## Setup

```powershell
uv sync --extra dev
Copy-Item .env.example .env
```

Fill in `.env`:

```env
DASHSCOPE_API_KEY=your_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_CHAT_MODEL=qwen3.7-plus
QWEN_EMBEDDING_MODEL=qwen3.7-text-embedding
PAPER_RANKING_STRATEGY=qwen3_rerank
DASHSCOPE_WORKSPACE_ID=your_workspace_id
QWEN_RERANK_MODEL=qwen3-rerank
SEARCH_RESULTS_PER_QUERY=15
ARXIV_DELAY_SECONDS=3.0
MAX_PAPERS=8
PDF_CANDIDATE_LIMIT=0
MAX_CHUNKS_PER_QUERY=12
MAX_EVIDENCE_CHUNKS=30
READER_BATCH_SIZE=10
VERIFIER_BATCH_SIZE=8
USE_PDF=false
PDF_FAILURE_POLICY=fallback_abstract
MIN_PDF_PAPERS=0
EXPERIMENT_ID=
```

## Run

```powershell
uv run python -m research_agent "What are the main methods for LLM inference acceleration from 2023 to 2025?"
```

Reports are saved to `outputs/reports/`. Runtime metadata is saved in `data/research_agent.db`, and vector chunks are stored under `data/chroma/<experiment-or-run-id>/`.
LangGraph state snapshots are saved in `data/checkpoints.sqlite` using the run ID as the `thread_id`.

## What v1 Does

- Plans the research question into subquestions and structured search intents made of required synonym groups.
- Compiles those intents into complete arXiv queries using quoted all-field Boolean syntax and the selected upload-year range.
- Searches arXiv and saves both each structured intent and the exact query sent to the API.
- Combines all intent results and deduplicates papers by normalized title.
- Can send the entire deduplicated candidate pool to Qwen3-Rerank in one request,
  using the original research question and each paper's title and abstract. RRF remains
  available as a baseline through `PAPER_RANKING_STRATEGY=rrf`.
- Ingests paper abstracts by default, with optional PDF ingestion. For stricter PDF experiments, set `PDF_FAILURE_POLICY=skip` so papers without usable PDF evidence are skipped.
- Chunks documents and persists vectors in Chroma.
- Retrieves evidence chunks for the original question and subquestions.
- Extracts evidence-backed claims.
- Verifies claim support against cited chunks.
- Writes a structured Markdown literature review.
- Persists LangGraph checkpoints for each run, so the graph state can be inspected or resumed later.

## Tests

```powershell
uv run --extra dev pytest
```

The tests mock external model behavior where possible and focus on schema parsing, metadata preservation, Chroma retrieval, and verifier behavior.

## Formal PDF Experiment Example

```powershell
$env:QWEN_CHAT_MODEL='qwen3.7-plus'
$env:SEARCH_RESULTS_PER_QUERY='15'
$env:MAX_PAPERS='10'
$env:PDF_CANDIDATE_LIMIT='15'
$env:MAX_CHUNKS_PER_QUERY='10'
$env:MAX_EVIDENCE_CHUNKS='0'
$env:READER_BATCH_SIZE='10'
$env:USE_PDF='true'
$env:PDF_FAILURE_POLICY='skip'
$env:MIN_PDF_PAPERS='6'
$env:EXPERIMENT_ID='rag-evaluation-methods-pdf-formal-001'
uv run python -m research_agent "What are the main methods for evaluating retrieval-augmented generation systems?"
```

## Resume From Saved Evidence Chunks

If a run has already finished retrieval but fails during `reader`, `verifier`, or `writer`,
reuse its saved chunks without rerunning search, PDF ingest, Chroma indexing, or embedding:

```powershell
$env:QWEN_CHAT_MODEL='qwen3.7-plus'
$env:MAX_EVIDENCE_CHUNKS='0'
$env:READER_BATCH_SIZE='10'
$env:VERIFIER_BATCH_SIZE='8'
uv run python -m research_agent --from-run <run_id>
```

Set `READER_BATCH_SIZE` lower if the chat API disconnects during reader.

If raw reader claims were already saved and only verification/report writing needs to be retried:

```powershell
$env:QWEN_CHAT_MODEL='deepseek-v4-flash'
$env:VERIFIER_BATCH_SIZE='8'
uv run python -m research_agent --verify-from-run <run_id>
```

## Future Work

- Enable PDF ingestion by default with page-level citation metadata.
- Add BM25 + Chroma hybrid retrieval.
- Add benchmark evaluation with LitSearch and BEIR/SciFact.
- Add a small Web UI after the CLI workflow is stable.
