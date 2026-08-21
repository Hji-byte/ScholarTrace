# ScholarTrace

An Evidence-Grounded Research Agent for Computer Science Literature Reviews

[English](README.md) | 简体中文

**想快速了解一个 CS 问题，又不想在海量论文里迷失方向？**

把问题交给 `ScholarTrace`：它会查找并筛选出最相关的论文，检索并提炼关键内容，最后生成一篇带引用、可追溯的文献综述报告。

![ScholarTrace 运行效果](./images/scholartrace_entry.png)

![ScholarTrace 工作流程](./images/scholartrace_workflow.png)

## 目录

- [项目概述](#项目概述)
- [安装](#安装)
- [Quick Start](#quick-start)
- [整体流程](#整体流程)
- [技术栈](#技术栈)
- [Evaluation](#evaluation)
- [补充用法](#补充用法)
- [输出汇总](#输出汇总)



## 项目概述

本项目是一个面向计算机科学领域的 Research Agent，可以根据用户输入的研究问题，生成文献综述报告。

与直接搜索并生成答案不同，本项目可分为三个阶段：论文查询阶段、证据检索阶段和报告写作阶段。论文查询阶段会根据输入的问题生成特定的 arXiv 搜索语句，对返回结果重排序后下载 PDF，并进行文本切分和向量化。证据检索阶段会围绕原问题和生成的多个子问题，对论文片段进行混合检索和重排序，并从中提取、验证 Claims。报告写作阶段会根据验证后的 Claims，输出带有 IEEE 参考文献格式的文献综述报告。

项目配套构建了计算机科学领域的 30 题测评集，并对论文搜索、证据检索、Claim 生成和最终报告进行了分阶段评估，具体结果见下文 [Evaluation](#evaluation) 部分。



## 安装

```powershell
uv sync --extra dev
Copy-Item .env.example .env
```

项目默认使用 Qwen 系列模型，请在 `.env` 中填写 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_WORKSPACE_ID`，其余配置可以沿用默认值，也可以根据个人需求更改。

如需更换其他模型，可以在 `.env` 中配置相关参数。



## Quick Start

输入问题即可启动完整的文献调研流程。例如：

```powershell
uv run python -m scholar_trace "What are the main methods for evaluating retrieval-augmented generation systems?"
```

该命令会依次完成论文搜索、PDF 下载、证据检索、观点提取、验证以及报告生成。

运行过程中，终端会实时显示当前节点、已用时间和各阶段的结果摘要。

文献综述保存在 `outputs/reports/`，各阶段的中间结果会写入 SQLite，便于断点续跑。

默认仅从 arXiv 检索论文，如需使用自己的 PDF 论文库，请参阅 [补充用法](#补充用法)。

如需限制论文的 arXiv 首次上传年份，可以参考以下命令：

```powershell
uv run python -m scholar_trace `
  "What are the main methods for evaluating retrieval-augmented generation systems?" `
  --year-from 2020 `
  --year-to 2025
```



## 整体流程

项目由七个节点构成：

```text
Planner → Search → Ingest → Retrieve → Reader → Verifier → Writer
```

| 节点 | 输入 | 具体行为 | 主要输出 |
| --- | --- | --- | --- |
| **Planner** | 用户提出的研究问题 | 将用户提出的问题拆成 3–5 个更具体的子问题，用于检索和写作；同时整理出 3–5 个搜索方向，供后续查找论文使用。 | 一份研究计划，包含子问题和搜索方向 |
| **Search** | 研究问题、搜索方向、可选的论文来源和年份范围 | 把每个搜索方向转换成 arXiv 能够执行的查询语句，每次搜索返回 15 篇论文。全部搜索完成后，根据论文标题和摘要与原问题的相关程度，使用 Qwen3-Rerank 重新排序，最终保留 Top 20。若想使用本地论文库，详见 [补充用法](#补充用法)。 | Top 20 论文候选 |
| **Ingest** | Search 生成的 Top 20 论文候选 | 按排名下载 10 篇论文，PDF 下载或解析失败时继续尝试下一篇。对下载后的论文进行文本切分，并将向量存入 Chroma。 | 可用论文片段及对应向量 |
| **Retrieve** | 原问题、子问题和论文片段 | 分别围绕原问题和各个子问题检索论文片段：Dense 负责语义匹配，BM25 负责关键词匹配。系统通过 RRF 合并多路结果，再用 Qwen3-Rerank 选出与原问题最相关的 Top 30 证据片段。 | Top 30 证据片段 |
| **Reader** | 原问题、子问题和 Top 30 证据片段 | 从证据片段中提取 Claims，并标记它们引用的片段和每条 Claim 对应的子问题。若某个子问题对应的 Claim 不足 2 条，则定向重新提取。 | Claims、引用的证据片段和对应的子问题 |
| **Verifier** | Reader 生成的 Claims 和对应证据片段 | 逐条检查 Claim 是否得到所引片段的支持，只保留真正提供证据的片段和有充分证据支持的 Claims。 | Verified Claims 和有效证据片段 |
| **Writer** | 原问题、子问题、Verified Claims、有效证据片段和论文信息 | 围绕原问题组织综述，参考文献格式参照 IEEE。 | Markdown 文献综述 |



## 技术栈

- **Agent 架构：** LangGraph、LangChain、SQLite Checkpointer
- **大模型：** Qwen Chat、Qwen Embedding、Qwen3-Rerank
- **论文获取与解析：** arXiv API、PyPDF
- **混合检索：** Chroma、Dense、BM25、RRF
- **数据处理：** Pydantic、SQLite



## Evaluation

### 30 题计算机科学测评集

下文所有实验均基于该评测集进行，相关性与质量评分由 Codex 和本人共同完成。

- 30 个计算机科学领域的研究问题；
- 覆盖 NLP、机器学习、计算机视觉、系统与网络、安全与隐私、软件工程、数据管理与信息检索 7 个方向；
- 包含 Survey、Comparison、Evaluation、Limitations 和 Trend 等问题类型；
- 年份条件独立存储，统一采用 arXiv 首次上传年份；
- 测评问题位于 [`evaluation/datasets/cs_questions.jsonl`](evaluation/datasets/cs_questions.jsonl)，对应生成的 30 篇文献综述位于 [`evaluation/reports`](evaluation/reports)。



### 1. Planner：子问题质量评估

对 30 题生成的 119 个 Subquestions 进行评估，检查子问题是否清晰有效，以及是否涵盖原问题需要回答的各个方面。

| 指标 | 说明 | 结果 |
| --- | --- | ---: |
| Valid Subquestions | 子问题含义明确、彼此独立，并且适用于检索 | 94.96% |
| Acceptable Subquestions | 子问题能够帮助回答原问题，但允许存在轻微冗余或范围较为宽泛 | 100.00% |
| Question Coverage | 一道题生成的全部子问题涵盖了原问题需要回答的各个方面 | 100.00% |



### 2. Search：候选论文相关性评估

Search 节点评估的是“系统找到的论文是否值得继续下载和阅读”。

标注时只查看研究问题、论文标题和摘要，不使用正文。

标注分为三级：

- 直接相关：论文的主要工作能够直接帮助回答问题；
- 部分相关：论文与问题有实质关联，可以提供背景或侧面证据；
- 无关：只有关键词重合，不能实际帮助回答问题。

Precision@K 统计直接相关和部分相关的论文。

| 论文排序方式 | 直接相关 | 部分相关 | 无关 | Precision@5 | Precision@10 | Precision@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RRF | 441 | 122 | 37 | 96.0% | 95.3% | 93.8% |
| **Qwen3-Rerank** | **497** | **85** | **18** | **99.3%** | **98.3%** | **97.0%** |

项目最终采用 Qwen3-Rerank 的论文排序方式。



### 3. Retrieve：证据检索方案对比

对 Top 30 Evidence Chunks 进行三级相关性标注：直接相关、部分相关和无关。

Precision@K 统计直接相关和部分相关的 chunks，Direct@K 只统计直接相关的 chunks。

| 方法 | Precision@5 | Precision@10 | Precision@20 | Direct@5 | Direct@10 | Direct@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense + RRF | 84.0% | 80.3% | 82.0% | 78.0% | 75.0% | 76.7% |
| BM25 + RRF | 95.3% | 94.0% | 93.5% | 80.0% | 78.3% | 76.0% |
| Hybrid + RRF | 92.7% | 90.7% | 88.5% | 84.7% | 81.0% | 78.2% |
| **Hybrid + RRF + Qwen3-Rerank** | **96.7%** | **96.3%** | **94.5%** | **90.0%** | **90.0%** | **86.2%** |

项目最终采用 Hybrid + RRF + Qwen3-Rerank 的检索方式。



### 4. Reader 与 Verifier 结果测评

对 30 道题生成的 661 条 Claims 进行标注：

| Reader 指标 | 说明 | 结果 |
| --- | --- | ---: |
| Full Support | Claim 的全部内容都能从引用证据中得到支持 | 91.5% |
| Partial Support | Claim 至少有部分内容得到引用证据支持 | 99.5% |
| Direct Relevance | Claim 能够直接帮助回答研究问题 | 95.5% |
| Partial Relevance | Claim 与研究问题至少部分相关 | 99.7% |
| Exact Mapping | Claim 与所标注的 Subquestion 完全匹配，没有遗漏或多标 | 89.6% |
| Acceptable Mapping | Claim 至少匹配一个所标注的 Subquestion，允许少量遗漏或多标 | 97.1% |

| Verifier指标 | 说明 | 结果 |
| --- | --- | ---: |
| Verified Full Support | Verifier 保留的 Claim 得到证据完整支持 | 91.8% |
| Verified Partial Support | Verifier 保留的 Claim 至少得到部分支持 | 99.5% |
| Exact Evidence Selection | 保留的 Evidence Chunks 都必要且有效，没有遗漏或冗余 | 86.2% |
| Acceptable Evidence Selection | 保留的 Evidence Chunks 能够支撑结论，但允许少量冗余 | 99.4% |



### 5. 最终报告测评

对 30 篇报告中的 1,035 条陈述进行了支持性、引用和写作质量评估：

| 指标 | 结果 |
| --- | ---: |
| Fully Supported Statement Rate | 90.58% |
| Partially Supported Statement Rate | 8.93% |
| Unsupported Statement Rate | 0.49% |
| Citation Precision | 95.84% |
| Citation Completeness | 85.63% |

写作质量采用 1–5 分量表，由 Codex 评分：

| 维度 | 分数 |
| --- | ---: |
| 回答相关性 | 4.53 |
| 结构组织 | 5.00 |
| 综合归纳 | 3.93 |
| 非冗余性 | 3.97 |
| 谨慎与校准 | 4.30 |
| 可读性 | 4.97 |
| **总体** | **4.45** |



## 补充用法

**1. 选择论文来源**

ScholarTrace 支持以下三种模式：

| 模式 | 参数 | 说明 |
| --- | --- | --- |
| 仅使用 arXiv | 无需额外参数 | 默认模式，系统根据研究问题搜索并下载 arXiv 论文。 |
| 仅使用本地论文库 | `--source library --library-path <路径>` | 从指定的 PDF 文件或文件夹中读取论文，不搜索 arXiv。 |
| 本地论文库 + arXiv | `--source hybrid --library-path <路径>` | 读取本地论文，并使用 arXiv 搜索结果补充候选论文。 |

`--library-path` 可以指向单个 PDF，也可以指向文件夹；文件夹内的 PDF 会被递归读取。例如：

```powershell
# 仅使用本地论文库
uv run python -m scholar_trace "Research question" --source library --library-path "D:\papers"

# 同时使用本地论文库和 arXiv
uv run python -m scholar_trace "Research question" --source hybrid --library-path "D:\papers"
```



**2. 断点续跑**

每次启动新任务时，终端都会显示 `Run ID`。请保留该编号，以便任务中断后继续运行。

1）从 Evidence Chunks 继续

```powershell
uv run python -m scholar_trace --from-run <run_id>
```

2）从 Raw Claims 继续

```powershell
uv run python -m scholar_trace --verify-from-run <run_id>
```



## 输出汇总

| 内容 | 默认位置 |
| --- | --- |
| 文献综述报告 | `outputs/reports/` |
| 下载的论文 PDF | `data/pdfs/` |
| Chroma Collection | `data/chroma/<run-id>/` |
| Metadata、Papers、Chunks、Claims | `data/scholar_trace.db` |
| LangGraph Checkpointer | `data/checkpoints.sqlite` |
| 公开测评集 | `evaluation/datasets/cs_questions.jsonl` |
| 30 篇测评报告 | `evaluation/reports/` |
