from __future__ import annotations

import xml.etree.ElementTree as ET
from hashlib import sha256
import re
import time

import requests

from scholar_trace.schema import Paper

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_SORT_BY = "relevance"
ARXIV_SORT_ORDER = "descending"
ARXIV_TIMEOUT_SECONDS = 20
ARXIV_MAX_RETRIES = 3


def hashed_arxiv_paper_id(value: str) -> str:
    digest = sha256(value.strip().encode("utf-8")).hexdigest()[:16]
    return f"arxiv-{digest}"


def arxiv_source_id(value: str) -> str:
    """Return a stable arXiv identifier without the version suffix."""
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#]+)", value, flags=re.IGNORECASE)
    if not match:
        return ""
    identifier = re.sub(r"\.pdf$", "", match.group(1), flags=re.IGNORECASE)
    identifier = re.sub(r"v\d+$", "", identifier, flags=re.IGNORECASE)
    return f"arXiv:{identifier}"


def search_arxiv(
    provider_query: str,
    max_results: int = 10,
    timeout: int = ARXIV_TIMEOUT_SECONDS,
    max_retries: int = ARXIV_MAX_RETRIES,
    retry_backoff_seconds: float = 2.0,
) -> list[Paper]:
    params = {
        "search_query": provider_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": ARXIV_SORT_BY,
        "sortOrder": ARXIV_SORT_ORDER,
    }
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(ARXIV_API, params=params, timeout=timeout)
            status_code = getattr(response, "status_code", 200)
            if status_code != 429 and status_code < 500:
                response.raise_for_status()
                break
            if attempt == max_retries:
                response.raise_for_status()
        except (requests.Timeout, requests.ConnectionError):
            if attempt == max_retries:
                raise
        except requests.RequestException:
            raise
        time.sleep(retry_backoff_seconds * (2**attempt))
    root = ET.fromstring(response.text)
    papers: list[Paper] = []
    for entry in root.findall(f"{ATOM}entry"):
        title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        if not title:
            continue
        url = entry.findtext(f"{ATOM}id") or ""
        if not url:
            continue

        abstract = " ".join((entry.findtext(f"{ATOM}summary") or "").split())
        published = entry.findtext(f"{ATOM}published") or ""
        authors = [
            name.text or ""
            for author in entry.findall(f"{ATOM}author")
            for name in [author.find(f"{ATOM}name")]
            if name is not None
        ]
        pdf_url = ""
        for link in entry.findall(f"{ATOM}link"):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break
        year = int(published[:4]) if published[:4].isdigit() else None
        papers.append(
            Paper(
                paper_id=hashed_arxiv_paper_id(url),
                source_id=arxiv_source_id(url),
                title=title.strip(),
                authors=authors,
                year=year,
                abstract=abstract.strip(),
                url=url.strip(),
                pdf_url=pdf_url.strip(),
                source="arxiv",
            )
        )
    return papers
