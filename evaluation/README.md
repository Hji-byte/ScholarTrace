# CS Literature Review Evaluation Set

`datasets/cs_questions_v2.jsonl` is the default fixed benchmark for the CS literature-review agent. Its natural-language questions do not contain explicit years; arXiv first-submission windows are stored separately in `year_from` and `year_to`, matching a user interface with independent date controls.
`datasets/cs_questions_v1.jsonl` preserves the original date-bearing question wording for provenance and is not used by default.
`datasets/key_papers_v2.jsonl` contains the current curated paper-level relevance judgments for those questions. Version 2 corrects official arXiv metadata and replaces one non-existent arXiv record; version 1 is retained so earlier experiment summaries remain reproducible.

## Scope

- 30 English research questions across seven CS domains.
- Publication windows end in 2025 so later runs evaluate against the same literature cutoff.
- Questions include survey, comparison, evaluation, limitations, and trend-analysis tasks.
- Difficulty labels describe the expected synthesis burden, not the reading level.

## Distribution

| Domain | Questions |
| --- | ---: |
| Natural language processing | 5 |
| Machine learning | 5 |
| Computer vision | 4 |
| Systems and networking | 4 |
| Security and privacy | 4 |
| Software engineering | 4 |
| Data management and information retrieval | 4 |

| Question type | Questions |
| --- | ---: |
| Survey | 7 |
| Comparison | 7 |
| Evaluation | 7 |
| Limitations | 5 |
| Trend | 4 |

| Difficulty | Questions |
| --- | ---: |
| Medium | 11 |
| Hard | 19 |

## Recommended First Pass

1. Run all 30 questions with one fixed baseline configuration.
2. Save the run ID, complete configuration, report path, latency, papers, chunks, claims, and verifier decisions for every question.
3. Use the required papers in `key_papers_v2.jsonl` as the strict gold set for Paper Recall@K. Use all papers, including supplemental entries, for a broader coverage score.
4. Select questions `csq001` through `csq010` for deeper annotation first. Label roughly 10 atomic claims and their supporting passages per question.
5. Freeze the resulting annotations as dataset version `v1`; do not edit them after comparing retrieval or verifier variants.

The first pass should use `USE_PDF=false` to reduce runtime and API failures. Use the same model, paper limit, retrieval limit, and prompts for every question. After the baseline is complete, rerun the selected 10-question subset with PDF evidence before committing to a full 30-question PDF experiment.

## Run the arXiv paper-retrieval evaluation

The paper-level baseline runs only the planner and arXiv search stages. It does not download PDFs, build Chroma indexes, generate claims, or write literature reviews.

Run a one-question smoke test first:

```powershell
.\.venv\Scripts\python.exe -m research_agent --paper-eval --eval-limit 1
```

Then run the fixed 30-question benchmark:

```powershell
.\.venv\Scripts\python.exe -m research_agent --paper-eval
```

The default `--eval-top-k` is 20. The runner writes an append-only per-question JSONL file, a raw-search JSONL file, and a summary JSON file under `evaluation/results/`. Each raw-search line preserves the structured search intent, the exact query sent to arXiv, and its original arXiv ranking so RRF and other ranking variants can be rerun offline without calling the planner or arXiv again. Failed arXiv requests are recorded as errors and excluded from macro metrics instead of being counted as retrieval misses.

All benchmark years use arXiv's first-submission year. For every intent, the arXiv adapter compiles required synonym groups into quoted all-field Boolean clauses, includes an inclusive `submittedDate` range, and requests the configured number of relevant results (15 by default). The runner then repeats the range check locally as an audit before applying RRF. Candidate and gold years remain in the result files for auditing.

## JSONL Fields

- `question_id`: stable identifier used to join runs and annotations.
- `domain`: broad CS area.
- `topic`: narrower research topic.
- `question_type`: reasoning and synthesis pattern.
- `difficulty`: expected synthesis difficulty.
- `year_from`, `year_to`: inclusive publication window.
- `question`: exact prompt supplied to the agent.

The question files contain prompts and structured metadata only. Key-paper relevance judgments and claim/passage annotations should live in separate JSONL files so they can be versioned without changing the questions.

## Key-paper labels

Each question has three `required` papers and usually one supplemental paper. The required set is intentionally small and method-diverse: it measures whether the search stage retrieves the minimum literature needed to answer the question, not whether it reproduces an exhaustive bibliography. Supplemental papers reward broader coverage without making strict recall unrealistically difficult.

Paper-level matching should use `source_id` first and normalized title second. Report both:

- `strict_recall@k`: recall over `required=true` papers.
- `broad_recall@k`: recall over all listed papers.

The collection is a manually curated benchmark seed rather than an exhaustive or community-adjudicated ground truth. Before publishing benchmark numbers, a second reviewer should inspect the paper choices.
