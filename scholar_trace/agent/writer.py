import re
from collections.abc import Iterable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from scholar_trace.agent.llm_retry import invoke_with_retry
from scholar_trace.schema import EvidenceChunk, Paper, ResearchState, VerifiedClaim


SYSTEM = """You are the report writer for a computer science literature-review agent.
Write a concise, evidence-grounded Markdown literature review that answers the original research question.

Evidence policy:
- Treat the verified claims as candidate findings and the verified evidence records as the authoritative factual source.
- You may narrow, qualify, split, or omit a claim to make the report match its evidence precisely.
- Do not omit details that are necessary for the claim to match its verified evidence.
- Do not introduce a new finding that is absent from the verified claims.
- Do not use outside knowledge or infer a broader conclusion than the evidence supports.
- If evidence is insufficient for an aspect of the question, state the limitation instead of filling the gap.

Citation policy:
- Cite papers with the supplied numeric paper citations, such as [1] or [1, 3].
- Every substantive statement about a method, result, trend, limitation, or comparison must have a paper citation.
- Cite only papers linked to that claim's verified evidence records.
- Never output paper ids, chunk ids, claim numbers, evidence-record labels, or other internal identifiers.
- Do not write a References section. The application will generate it deterministically from the paper citations you use.

Organization:
- Start with '# Literature Review' and a '## Summary'.
- Add two to four natural, topic-specific sections based on the evidence and the research question.
- End with '## Limitations and Open Problems'.
- Use the supplied subquestions as a coverage checklist, not as a mandatory outline.
- Do not mechanically create one section per subquestion. Combine overlapping subquestions and use reader-friendly headings.
- Group related verified claims into coherent paragraphs instead of presenting each claim separately.
- Keep the tone careful and academic, and avoid repeating the same finding across sections.
- If the evidence base is small, write a shorter report and say that the available evidence is limited.
- If there are no supported claims, do not fabricate a review; briefly state that the retrieved evidence was insufficient.
- Do not mention internal planner, reader, verifier, retrieval, claim-count, or coverage machinery."""


CITATION_RE = re.compile(r"\[((?:\d+\s*,\s*)*\d+)\]")
REFERENCE_HEADING_RE = re.compile(r"^##\s+References\s*$", re.IGNORECASE | re.MULTILINE)
EVIDENCE_LABEL_RE = re.compile(r"\bEvidence(?:\s+record)?\s+E\d+\b", re.IGNORECASE)
SUMMARY_SECTION_RE = re.compile(
    r"^##\s+Summary\s*$\n(.*?)(?=^##\s+|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
LEVEL_ONE_HEADING_RE = re.compile(r"\A#\s+[^\r\n]*(?:\r?\n|\Z)")


def _paper_key(chunk: EvidenceChunk) -> str:
    return chunk.paper_id or f"title:{' '.join(chunk.title.lower().split())}"


def _format_ieee_author(author: str) -> str:
    parts = author.split()
    if len(parts) < 2:
        return author.strip()
    initials = " ".join(
        f"{piece[0].upper()}."
        for part in parts[:-1]
        for piece in part.replace("-", " ").split()
        if piece
    )
    return f"{initials} {parts[-1]}".strip()


def _format_authors(authors: list[str]) -> str:
    formatted = [_format_ieee_author(author) for author in authors if author.strip()]
    if not formatted:
        return ""
    if len(formatted) > 6:
        return f"{formatted[0]} et al."
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return f"{', '.join(formatted[:-1])}, and {formatted[-1]}"


def _format_reference(number: int, paper: Paper | None, chunk: EvidenceChunk) -> str:
    title = " ".join((paper.title if paper else chunk.title).split()) or "Untitled paper"
    authors = _format_authors(paper.authors) if paper else ""
    year = str(paper.year) if paper and paper.year is not None else ""
    venue = " ".join(paper.venue.split()) if paper and paper.venue else ""
    url = (paper.url if paper and paper.url else chunk.url).strip()
    source_id = paper.source_id.strip() if paper else ""

    prefix = f"[{number}] "
    if authors:
        prefix += f"{authors}, "
    reference = f'{prefix}“{title},”'
    if source_id.lower().startswith("arxiv:"):
        publication_parts = [f"arXiv preprint {source_id}"]
    else:
        publication_parts = [venue] if venue else []
    if year:
        publication_parts.append(year)
    publication = ", ".join(publication_parts)
    if publication:
        reference += f" {publication}."
    if url:
        reference += f" [Online]. Available: {url}"
    return reference


def _citation_numbers(text: str) -> set[int]:
    numbers: set[int] = set()
    for match in CITATION_RE.finditer(text):
        numbers.update(int(value.strip()) for value in match.group(1).split(","))
    return numbers


def validate_report_body(
    body: str,
    valid_citation_numbers: set[int],
    internal_identifiers: Iterable[str],
    require_citation: bool,
) -> set[int]:
    if REFERENCE_HEADING_RE.search(body):
        raise ValueError("Do not write a References section")
    if EVIDENCE_LABEL_RE.search(body):
        raise ValueError("The report exposes an internal evidence-record label")
    leaked = [identifier for identifier in internal_identifiers if identifier and identifier in body]
    if leaked:
        raise ValueError(f"The report exposes internal identifiers: {sorted(set(leaked))}")

    used = _citation_numbers(body)
    unknown = used - valid_citation_numbers
    if unknown:
        raise ValueError(f"The report uses unknown paper citations: {sorted(unknown)}")
    if require_citation and not used:
        raise ValueError("The report contains supported findings but no paper citations")
    if require_citation:
        summary = SUMMARY_SECTION_RE.search(body)
        if summary is None or not _citation_numbers(summary.group(1)):
            raise ValueError("The Summary section must contain a paper citation")
    return used


def append_references(body: str, used: set[int], references: dict[int, str]) -> str:
    entries = [references[number] for number in sorted(used)]
    reference_text = "\n".join(entries) if entries else "No references were cited."
    return f"{body.rstrip()}\n\n## References\n\n{reference_text}"


def insert_question_after_title(body: str, question: str) -> str:
    """Place the original question below the report's fixed Markdown title."""
    normalized_body = body.strip()
    normalized_question = " ".join(question.split())
    if not normalized_question:
        raise ValueError("The research question cannot be empty")
    if LEVEL_ONE_HEADING_RE.match(normalized_body):
        remainder = LEVEL_ONE_HEADING_RE.sub("", normalized_body, count=1).lstrip()
    else:
        remainder = normalized_body
    return f"# Literature Review\n\n{normalized_question}\n\n{remainder}".rstrip()


def write_report(llm: BaseChatModel, validation_attempts: int = 2):
    def node(state: ResearchState) -> ResearchState:
        supported = [claim for claim in state.get("verified_claims", []) if claim.supported]
        cited_chunk_ids = {
            chunk_id
            for claim in supported
            for chunk_id in claim.verified_evidence_chunk_ids
        }
        selected_chunks = [
            chunk
            for chunk in state.get("evidence_chunks", [])
            if chunk.chunk_id in cited_chunk_ids
        ]
        chunks_by_id = {chunk.chunk_id: chunk for chunk in selected_chunks}
        papers_by_id = {paper.paper_id: paper for paper in state.get("papers", [])}

        paper_number_by_key: dict[str, int] = {}
        representative_chunk_by_number: dict[int, EvidenceChunk] = {}
        for chunk in selected_chunks:
            key = _paper_key(chunk)
            if key not in paper_number_by_key:
                number = len(paper_number_by_key) + 1
                paper_number_by_key[key] = number
                representative_chunk_by_number[number] = chunk

        chunk_citation_number = {
            chunk.chunk_id: paper_number_by_key[_paper_key(chunk)]
            for chunk in selected_chunks
        }
        evidence_label_by_chunk_id = {
            chunk.chunk_id: f"E{index}"
            for index, chunk in enumerate(selected_chunks, start=1)
        }
        references = {
            number: _format_reference(
                number,
                papers_by_id.get(chunk.paper_id),
                chunk,
            )
            for number, chunk in representative_chunk_by_number.items()
        }

        def claim_text(index: int, claim: VerifiedClaim) -> str:
            evidence_ids = [
                chunk_id
                for chunk_id in claim.verified_evidence_chunk_ids
                if chunk_id in chunks_by_id
            ]
            paper_numbers = sorted({chunk_citation_number[chunk_id] for chunk_id in evidence_ids})
            evidence_labels = [evidence_label_by_chunk_id[chunk_id] for chunk_id in evidence_ids]
            citations = ", ".join(f"[{number}]" for number in paper_numbers) or "none"
            labels = ", ".join(evidence_labels) or "none"
            return (
                f"Candidate finding {index}:\n"
                f"Finding: {claim.claim}\n"
                f"Topic: {claim.category}\n"
                f"Relevant subquestion numbers: "
                f"{', '.join(map(str, claim.subquestion_ids)) or 'overall'}\n"
                f"Allowed paper citations: {citations}\n"
                f"Verified evidence records: {labels}"
            )

        claims_text = "\n\n".join(
            claim_text(index, claim) for index, claim in enumerate(supported, start=1)
        ) or "(No supported findings.)"
        evidence_text = "\n\n".join(
            (
                f"Evidence record {evidence_label_by_chunk_id[chunk.chunk_id]} "
                f"(internal; never output this label):\n"
                f"Paper citation: [{chunk_citation_number[chunk.chunk_id]}]\n"
                f"Paper title: {chunk.title}\n"
                f"Page: {chunk.page if chunk.page is not None else 'not recorded'}\n"
                f"Section: {chunk.section or 'not recorded'}\n"
                f"Text: {chunk.text}"
            )
            for chunk in selected_chunks
        ) or "(No verified evidence records.)"
        reference_candidates = "\n".join(references.values()) or "(No reference candidates.)"

        subquestions = list(state["plan"].subquestions) if state.get("plan") else []
        subquestion_text = "\n".join(
            f"{index}. {subquestion}"
            for index, subquestion in enumerate(subquestions, start=1)
        ) or "(No explicit subquestions.)"
        coverage = state.get("verified_claim_coverage", state.get("claim_coverage", {}))
        coverage_text = "\n".join(
            (
                f"Subquestion {item['subquestion_id']}: "
                f"{item['claim_count']} supported findings, "
                f"covered={item['covered']}"
            )
            for item in coverage.get("subquestions", [])
        ) or "(Coverage was not recorded.)"

        messages = [
            SystemMessage(content=SYSTEM),
            HumanMessage(
                content=(
                    f"Original research question: {state['question']}\n\n"
                    f"Subquestions for coverage planning:\n{subquestion_text}\n\n"
                    f"Verified coverage diagnostics:\n{coverage_text}\n\n"
                    f"Verified candidate findings:\n{claims_text}\n\n"
                    f"Verified evidence records:\n{evidence_text}\n\n"
                    f"Permitted paper references (the application will append the entries you cite):\n"
                    f"{reference_candidates}"
                )
            ),
        ]
        internal_identifiers = [
            *(chunk.chunk_id for chunk in selected_chunks),
            *(chunk.paper_id for chunk in selected_chunks),
        ]
        attempts = max(1, validation_attempts)
        for attempt in range(1, attempts + 1):
            response = invoke_with_retry(llm, messages)
            body = insert_question_after_title(str(response.content), state["question"])
            try:
                used_citations = validate_report_body(
                    body,
                    set(references),
                    internal_identifiers,
                    require_citation=bool(supported),
                )
                report = append_references(body, used_citations, references)
                break
            except ValueError as exc:
                if attempt >= attempts:
                    raise
                messages.extend(
                    [
                        response,
                        HumanMessage(
                            content=(
                                f"The previous report violated the output contract: {exc}. "
                                "Return the complete corrected report body. Use only the supplied "
                                "numeric paper citations and do not write a References section."
                            )
                        ),
                    ]
                )

        trace = state.get("trace", []) + [
            f"Generated final Markdown report with {len(used_citations)} cited papers."
        ]
        return {"report_markdown": report, "trace": trace}

    return node
