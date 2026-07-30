from __future__ import annotations

from research_agent.schema import SearchIntent


def _quote_term(term: str) -> str:
    normalized = " ".join(term.split())
    escaped = normalized.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _all_fields_clause(term: str) -> str:
    return f"all:{_quote_term(term)}"


def compile_arxiv_query(
    intent: SearchIntent,
    year_from: int | None = None,
    year_to: int | None = None,
) -> str:
    """Compile a search intent and optional upload-year range to arXiv syntax."""
    required_groups = []
    for group in intent.must_groups:
        alternatives = " OR ".join(_all_fields_clause(term) for term in group)
        required_groups.append(f"({alternatives})")
    query = " AND ".join(required_groups)

    if year_from is not None or year_to is not None:
        start_year = year_from if year_from is not None else 1991
        end_year = year_to if year_to is not None else 9999
        if start_year > end_year:
            raise ValueError("year_from must be less than or equal to year_to")
        query = (
            f"{query} AND submittedDate:[{start_year:04d}01010000 "
            f"TO {end_year:04d}12312359]"
        )
    return query
