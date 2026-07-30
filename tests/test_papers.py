import requests

from research_agent.schema import SearchIntent
from research_agent.search_providers.arxiv_adapter import compile_arxiv_query
from research_agent.tools.arxiv_search import hashed_arxiv_paper_id, search_arxiv


ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <title> Fast Inference from Transformers via Speculative Decoding </title>
    <summary> Speculative decoding uses a draft model to reduce latency. </summary>
    <published>2024-01-01T00:00:00Z</published>
    <author><name>Ada Lovelace</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v2" type="application/pdf" />
  </entry>
</feed>
"""


class FakeResponse:
    text = ARXIV_XML

    def raise_for_status(self):
        return None


def test_search_arxiv_parses_atom_response(monkeypatch):
    def fake_get(url, params, timeout):
        assert params["search_query"] == "all:speculative decoding"
        assert params["sortBy"] == "relevance"
        return FakeResponse()

    monkeypatch.setattr("research_agent.tools.arxiv_search.requests.get", fake_get)

    papers = search_arxiv("all:speculative decoding", max_results=1)

    assert papers[0].paper_id == hashed_arxiv_paper_id("http://arxiv.org/abs/2401.12345v2")
    assert papers[0].title == "Fast Inference from Transformers via Speculative Decoding"
    assert papers[0].authors == ["Ada Lovelace"]
    assert papers[0].year == 2024
    assert papers[0].pdf_url == "http://arxiv.org/pdf/2401.12345v2"
    assert papers[0].source == "arxiv"


def test_arxiv_adapter_adds_submitted_date_range():
    intent = SearchIntent(
        purpose="speculative decoding",
        must_groups=[["speculative decoding"]],
    )

    assert compile_arxiv_query(intent, year_from=2020, year_to=2025) == (
        '(all:"speculative decoding") AND '
        "submittedDate:[202001010000 TO 202512312359]"
    )


def test_search_arxiv_sends_provider_query_unchanged(monkeypatch):
    provider_query = (
        '(all:"speculative decoding") AND '
        "submittedDate:[202001010000 TO 202512312359]"
    )

    def fake_get(url, params, timeout):
        assert params["search_query"] == provider_query
        assert params["max_results"] == 10
        assert params["sortBy"] == "relevance"
        return FakeResponse()

    monkeypatch.setattr("research_agent.tools.arxiv_search.requests.get", fake_get)

    search_arxiv(provider_query)


def test_arxiv_adapter_compiles_short_all_field_phrases_with_and_or_groups():
    intent = SearchIntent(
        purpose="serverless cold starts",
        must_groups=[
            ["serverless computing", "function as a service"],
            ["cold start", "startup latency"],
        ],
    )

    assert compile_arxiv_query(intent) == (
        '(all:"serverless computing" OR all:"function as a service") AND '
        '(all:"cold start" OR all:"startup latency")'
    )


def test_search_arxiv_uses_compiled_intent_before_date_filter(monkeypatch):
    intent = SearchIntent(
        purpose="malicious package benchmarks",
        must_groups=[
            ["malicious package", "software supply chain"],
            ["dataset", "benchmark"],
        ],
    )

    def fake_get(url, params, timeout):
        assert params["search_query"] == (
            '(all:"malicious package" OR all:"software supply chain") AND '
            '(all:"dataset" OR all:"benchmark") AND '
            'submittedDate:[201901010000 TO 202512312359]'
        )
        return FakeResponse()

    monkeypatch.setattr("research_agent.tools.arxiv_search.requests.get", fake_get)

    search_arxiv(compile_arxiv_query(intent, year_from=2019, year_to=2025))


def test_search_arxiv_does_not_retry_non_rate_limit_4xx(monkeypatch):
    calls = []
    sleeps = []

    class BadRequestResponse:
        status_code = 400
        text = ""

        def raise_for_status(self):
            raise requests.HTTPError("400 Bad Request")

    def fake_get(url, params, timeout):
        calls.append(params["search_query"])
        return BadRequestResponse()

    monkeypatch.setattr("research_agent.tools.arxiv_search.requests.get", fake_get)
    monkeypatch.setattr("research_agent.tools.arxiv_search.time.sleep", sleeps.append)

    try:
        search_arxiv("invalid query", max_retries=3)
    except requests.HTTPError:
        pass
    else:
        raise AssertionError("400 responses must fail immediately")

    assert calls == ["invalid query"]
    assert sleeps == []


def test_search_arxiv_retries_rate_limit(monkeypatch):
    responses = []
    sleeps = []

    class RateLimitedResponse:
        status_code = 429
        text = ""

        def raise_for_status(self):
            raise requests.HTTPError("429 Too Many Requests")

    def fake_get(url, params, timeout):
        responses.append(params["search_query"])
        return RateLimitedResponse() if len(responses) == 1 else FakeResponse()

    monkeypatch.setattr("research_agent.tools.arxiv_search.requests.get", fake_get)
    monkeypatch.setattr("research_agent.tools.arxiv_search.time.sleep", sleeps.append)

    papers = search_arxiv("all:test", retry_backoff_seconds=0.25)

    assert len(papers) == 1
    assert responses == ["all:test", "all:test"]
    assert sleeps == [0.25]


def test_hashed_arxiv_paper_id_is_stable_and_file_safe():
    paper_id = hashed_arxiv_paper_id("http://arxiv.org/abs/cs/9901001v1")

    assert paper_id == hashed_arxiv_paper_id("http://arxiv.org/abs/cs/9901001v1")
    assert paper_id != hashed_arxiv_paper_id("http://arxiv.org/abs/2401.12345v2")
    assert paper_id.startswith("arxiv-")
    assert "/" not in paper_id
