# Manual relevance audit

Source experiment: `arxiv-paper-eval-20260723-120305`

This is a first-pass title-level human screening of the 20 final papers returned for each question. It is intended to diagnose retrieval quality, not to serve as a second gold-standard annotation. Ambiguous papers should be checked from their full abstracts or PDFs before these labels are used as formal precision judgments.

Labels:

- **Direct**: substantively addresses a central method, comparison, benchmark, limitation, or trend requested by the question.
- **Partial**: useful adjacent background or a domain-specific instance, but does not directly answer the whole question.
- **Irrelevant**: keyword collision or a different task/domain.

| Question | Direct | Partial | Irrelevant | Assessment |
|---|---:|---:|---:|---|
| csq001 | 5 | 10 | 5 | Mixed: Ragas and several RAG evaluation/benchmark papers are useful, but many application and augmentation papers do not define evaluation methods. |
| csq002 | 14 | 1 | 5 | Good: speculative decoding, quantization, and KV-cache papers are all represented; unrelated mathematical optimization papers remain. |
| csq003 | 9 | 5 | 6 | Mixed: several factuality metrics and reliability studies are useful, with unrelated general human/LLM evaluation results. |
| csq004 | 3 | 5 | 12 | Poor: generic “bias” terms retrieve social-bias and gravitational-wave papers rather than biases of LLM judges. |
| csq005 | 10 | 3 | 7 | Mixed: long-context LLM work is present, but video, time-series, and other long-sequence domains introduce noise. |
| csq006 | 7 | 5 | 8 | Mixed: PEFT surveys, prefix tuning, adapters, and decomposition methods appear, mixed with generic fine-tuning results. |
| csq007 | 11 | 8 | 1 | Good: nearly all papers concern federated heterogeneity, non-IID data, client selection, or adjacent FL assumptions. |
| csq008 | 6 | 1 | 13 | Poor: “calibration” heavily collides with camera, measurement, physics, and unrelated calibration tasks. |
| csq009 | 13 | 2 | 5 | Good: over-smoothing, over-squashing, curvature, rewiring, and spectral methods dominate the list. |
| csq010 | 7 | 13 | 0 | Broadly useful: all papers concern self-supervised learning, though many are domain-specific examples rather than an evolution of objectives. |
| csq011 | 12 | 4 | 4 | Good: NeRF, Gaussian splatting, view synthesis, compression, and rendering efficiency are well represented. |
| csq012 | 10 | 5 | 5 | Good/mixed: the exact comparison paper appears at rank 15, while medical imaging and deepfake papers add noise. |
| csq013 | 3 | 6 | 11 | Poor: only a few papers specifically evaluate VLM hallucination; many are generic multimodal benchmarks or unrelated hallucination domains. |
| csq014 | 12 | 2 | 6 | Good: accuracy/robustness trade-offs and adversarial-training limitations are represented, with some GAN keyword collisions. |
| csq015 | 7 | 3 | 10 | Mixed/poor: several Byzantine-consensus papers are relevant, but generic consensus, coding, astronomy, and optimization papers dilute the list. |
| csq016 | 7 | 0 | 13 | Poor: valid serverless cold-start papers coexist with many recommender-system and gravitational-wave “cold start” collisions. |
| csq017 | 13 | 3 | 4 | Good: cluster schedulers, traces, workload characterization, utilization, and evaluation infrastructure are well covered. |
| csq018 | 5 | 2 | 13 | Poor: a few programmable data-plane and in-network computing papers are present, but most results are generic distributed computing or networking. |
| csq019 | 5 | 2 | 13 | Poor: private training papers are present, but generic deep learning, uncertainty, and non-training DP papers dominate. |
| csq020 | 14 | 0 | 6 | Good: black-box/white-box membership inference, threat models, defenses, and benchmarks dominate. |
| csq021 | 0 | 4 | 16 | Very poor: generic dataset/benchmark terms overwhelm malicious-package detection and supply-chain evaluation. |
| csq022 | 11 | 9 | 0 | Good: all papers are at least adjacent to FL attacks, privacy, poisoning, secure aggregation, or defenses. |
| csq023 | 15 | 4 | 1 | Good: automated program repair, patch assessment, fault localization, testing, and LLM repair dominate. |
| csq024 | 11 | 4 | 5 | Good: execution, tests, pass@k, leakage, judges, and code-generation benchmarks are well represented. |
| csq025 | 2 | 3 | 15 | Very poor: “dataset”, “leakage”, and “vulnerability” retrieve unrelated medical, information-theory, remote-sensing, and physics papers. |
| csq026 | 14 | 3 | 3 | Good: repository context, SWE-bench, testing, localization, and reproducibility are strongly represented. |
| csq027 | 4 | 1 | 15 | Very poor: ALEX and a few learned indexes appear, but insertion/deletion coding and unrelated learning papers dominate. |
| csq028 | 18 | 0 | 2 | Very good: almost the entire list concerns ANN graph indexes, product quantization, recall, construction, or search cost. |
| csq029 | 13 | 6 | 1 | Good: BEIR, sparse/dense/hybrid retrieval, robustness, domain transfer, and evaluation dominate. |
| csq030 | 16 | 0 | 4 | Very good: schema linking, Spider, neural parsers, LLM methods, and cross-domain evaluation dominate. |

## Aggregate screening result

- Direct: 277/600 (46.2%)
- Partial: 114/600 (19.0%)
- Irrelevant: 209/600 (34.8%)
- Direct or partial: 391/600 (65.2%)
- Good result sets: 15/30
- Mixed result sets: 6/30
- Poor result sets: 9/30

The dominant failure mode is ambiguous broad vocabulary passed to arXiv full-field search, including `bias`, `calibration`, `cold start`, `dataset`, `leakage`, `insertion/deletion`, and `consensus`. RRF cannot repair a candidate pool in which several query lists already contain systematic keyword collisions.

## Metrics recalculated with gold v2

Reusing the saved Top-20 results with `key_papers_v2.jsonl` produces the same aggregate metrics as the original run: strict Recall@20 is 8.89%, broad Recall@20 is 9.17%, and MRR is 0.1059. The corrected csq012 replacement paper was not present in that question's Top 20, so the metadata correction does not artificially improve or reduce this run's score.
