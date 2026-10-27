"""Tests for the TypeSafe Jev (System One) comparers and window oracle."""
import json
from unittest import mock

import httpx
import pytest

from blitzrank.engine.comparer.jev_api import (
    JevApiComparer,
    JevApiWindowComparer,
    JevClient,
)
from blitzrank.engine.algorithms.tournament_graph.experimental_interface import (
    ContentItem, SingleContent,
)
from blitzrank.engine.algorithms.tournament_graph.jev_oracle import JevWindowOracle
from blitzrank.engine.config import ComparerConfig, ComparerType

PAIRWISE_RESPONSE = {
    "answers": {"pair_1_2": {"type": "choice", "choice": "second", "confidence": 0.97,
                             "probabilities": {"first": 0.03, "second": 0.97}}},
    "usage": {"input_tokens": 428, "output_tokens": 0},
}


def _window_response(k, answers_by_key):
    answers = {key: {"type": "choice", "choice": choice,
                     "confidence": 1.0,
                     "probabilities": {"left": 1.0 if choice == "left" else 0.0,
                                       "right": 1.0 if choice == "right" else 0.0}}
               for key, choice in answers_by_key.items()}
    return {"answers": answers, "usage": {"input_tokens": 10, "output_tokens": 0}}


@pytest.fixture
def pairwise_comparer(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    comparer = JevApiComparer(model="jev-latest", max_doc_tokens=1024)
    return comparer


@pytest.mark.asyncio
async def test_pairwise_first_and_second(pairwise_comparer):
    for choice, expected_index in (("first", 0), ("second", 1)):
        response = {"answers": {"pair_1_2": {"choice": choice,
                                             "probabilities": {"first": 0.6, "second": 0.4}}},
                    "usage": {"input_tokens": 42, "output_tokens": 0}}
        with mock.patch.object(JevClient, "_post", return_value=response) as post:
            result = await pairwise_comparer.compare("capital of France", ["doc one", "doc two"])
        assert result.winner_index == expected_index
        assert result.probabilities == {"first": 0.6, "second": 0.4}
        assert result.input_tokens == 42 and result.output_tokens == 0
        assert result.latency_ms > 0
        assert result.model == "jev-latest"
        assert result.raw_prompt == post.call_args.args[0]
        assert result.raw_response == str(response)


@pytest.mark.asyncio
async def test_pairwise_doc_trimming(pairwise_comparer):
    long_doc = "word " * 5000
    with mock.patch.object(JevClient, "_post", return_value=PAIRWISE_RESPONSE) as post:
        await pairwise_comparer.compare("q", [long_doc, "short doc"])
    criteria = post.call_args.args[0]["questions"]["pair_1_2"]["criteria"]
    expected = pairwise_comparer._prepare_doc(long_doc)
    assert criteria["first"] == expected and criteria["second"] == "short doc"
    assert len(criteria["first"]) < len(long_doc)


@pytest.mark.asyncio
async def test_pairwise_missing_api_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    comparer = JevApiComparer(model="jev-latest", max_doc_tokens=1024)
    with pytest.raises(ValueError, match="Please set TYPESAFE_API_KEY"):
        await comparer.compare("q", ["a", "b"])


@pytest.mark.asyncio
async def test_pairwise_http_error_propagates(pairwise_comparer):
    request = httpx.Request("POST", "https://api.typesafe.ai/v1/systemone")
    response = httpx.Response(400, request=request)
    error = httpx.HTTPStatusError("bad request", request=request, response=response)
    with mock.patch.object(JevClient, "_post", side_effect=error):
        with pytest.raises(httpx.HTTPStatusError):
            await pairwise_comparer.compare("q", ["a", "b"])


@pytest.mark.asyncio
async def test_window_comparer_all_pairs():
    comparer = JevApiWindowComparer(model="jev-latest", max_doc_tokens=1024)
    # doc 2 beats everyone; doc 1 beats doc 3..5; etc. — a known total order: 2,1,3,5,4
    beats = {("2", "1"), ("2", "3"), ("2", "4"), ("2", "5"),
             ("1", "3"), ("1", "4"), ("1", "5"), ("3", "5"), ("4", "5"), ("3", "4")}
    answers = {}
    for i in range(1, 6):
        for j in range(i + 1, 6):
            answers[f"pair_{i}_{j}"] = "left" if (str(i), str(j)) in beats else "right"
    with mock.patch.object(JevClient, "_post") as post:
        post.return_value = _window_response(5, answers)
        result = await comparer.compare("q", [f"doc {i}" for i in range(1, 6)])

    request = post.call_args.args[0]
    assert len(request["questions"]) == 10
    assert set(request["questions"]) == {f"pair_{i}_{j}" for i in range(1, 5) for j in range(i + 1, 6)}
    state_lines = request["state"].split("\n\n")
    assert state_lines[0] == "Query: q"
    assert [line.split(":")[0] for line in state_lines[1:]] == [f"Doc {i}" for i in range(1, 6)]

    expected_edges = {(1, 0), (1, 2), (1, 4), (1, 3), (0, 2), (0, 4), (0, 3), (2, 4), (2, 3), (4, 3)}
    assert result.edges == [
        (i - 1, j - 1) if (str(i), str(j)) in beats else (j - 1, i - 1)
        for i in range(1, 6) for j in range(i + 1, 6)
    ]
    # order follows combinations over the input order; all edges winner-first
    assert all((str(w + 1), str(l + 1)) in beats for w, l in result.edges)


def _content_item(i, query):
    content = SingleContent(content=f"doc {i}", qid=0, docid=str(i), rank=i + 1,
                            score=5 - i, orig_idx=i)
    return ContentItem(content, query)


@pytest.mark.asyncio
async def test_jev_window_oracle_integration():
    oracle = JevWindowOracle(5, ComparerConfig(type=ComparerType.JEV_API, model="jev-latest"))
    items = [_content_item(i, "q") for i in range(5)]
    with mock.patch.object(JevApiWindowComparer, "compare") as compare:
        from blitzrank.engine.comparer.jev_api import JevWindowResult
        compare.return_value = JevWindowResult(
            edges=[(1, 0), (0, 2)], input_tokens=10, output_tokens=0,
            latency_ms=1.5, model="jev-latest", raw_prompt={}, raw_response="",
        )
        result = await oracle.compare_k_async(items)
        compare.assert_awaited_once_with(query="q", docs=[item.content.content for item in items])

    assert oracle.calls == 1
    expected = [(items[1], items[0]), (items[0], items[2])]
    for edge in result.edges:
        assert edge in expected
    assert len(oracle.call_logs) == 1
    log = oracle.call_logs[0]
    assert log["num_pairs_decided"] == 2
    for field in ("input_tokens", "output_tokens", "thought_tokens", "latency_ms",
                  "missing_indices", "duplicate_indices", "num_trimmed_docs"):
        assert field in log
