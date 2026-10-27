"""
TypeSafe Jev (System One) comparers: structured pairwise decisions via the
/v1/systemone API.

LitellmClient is bypassed on purpose: Jev is not an OpenAI-compatible chat
endpoint, so (like CtxlApiComparer) we own the HTTP call.
"""
import os
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Tuple

import httpx

from .base import BaseComparer, CompareResultBase
from ..utils.retry_utils import async_retry

API_TIMEOUT = 60.0
API_URL = "https://api.typesafe.ai/v1/systemone"


def _doc_label(idx: int) -> str:
    return f"Doc {idx}"


class JevClient:
    """Thin async HTTP wrapper around the TypeSafe systemone endpoint."""

    def __init__(self):
        self._api_key = None

    def _get_api_key(self) -> str:
        if self._api_key is None:
            self._api_key = os.getenv("TYPESAFE_API_KEY")
            if self._api_key is None:
                raise ValueError("Please set TYPESAFE_API_KEY")
        return self._api_key

    async def _post(self, body: Dict) -> Dict:
        headers = {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.post(API_URL, json=body, headers=headers)
        response.raise_for_status()
        return response.json()


@dataclass
class JevPairwiseResult(CompareResultBase):
    winner_index: int  # 0 or 1, relative to the order of docs passed in
    probabilities: Dict[str, float] = field(default_factory=dict)


class JevApiComparer(BaseComparer):
    """Pairwise relevance decisions: one Choice question per doc pair, all
    pairs of the (2-doc) comparison batched into a single systemone call."""

    def __init__(self, model: str, max_doc_tokens: int, max_query_tokens: int = 1024):
        super().__init__(model, max_doc_tokens, max_query_tokens)
        self._client = JevClient()

    @async_retry()
    async def compare(self, query: str, docs: List[str]) -> JevPairwiseResult:
        assert len(docs) == 2, f"Pairwise comparer requires exactly 2 docs, got {len(docs)}"
        request = {
            "model": self.model,
            "state": f"Query: {self._prepare_query(query)}",
            "questions": {
                "pair_1_2": {
                    "type": "choice",
                    "instructions": (
                        "Two passages are shown below. Which passage is more "
                        "relevant to the query in the state?"
                    ),
                    "criteria": {
                        "first": self._prepare_doc(docs[0]),
                        "second": self._prepare_doc(docs[1]),
                    },
                }
            },
        }

        start_time = time.perf_counter()
        data = await self._client._post(request)
        latency_ms = (time.perf_counter() - start_time) * 1000

        answer = data["answers"]["pair_1_2"]
        choice = answer["choice"]
        assert choice in ("first", "second"), f"Unexpected choice value: {choice!r}"
        usage = data.get("usage", {})

        return JevPairwiseResult(
            winner_index=0 if choice == "first" else 1,
            probabilities=dict(answer.get("probabilities", {})),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=latency_ms,
            model=self.model,
            raw_prompt=request,
            raw_response=str(data),
        )


@dataclass
class JevWindowResult(CompareResultBase):
    """All pairwise edges from one window's batched Choice questions."""
    edges: List[Tuple[int, int]]  # (winner_idx, loser_idx) into the input docs
    probabilities: Dict[str, Dict[str, float]] = field(default_factory=dict)


class JevApiWindowComparer(BaseComparer):
    """One systemone call per window; one Choice question per doc pair."""

    def __init__(self, model: str, max_doc_tokens: int, max_query_tokens: int = 1024):
        super().__init__(model, max_doc_tokens, max_query_tokens)
        self._jev = JevClient()

    @async_retry()
    async def compare(self, query: str, docs: List[str]) -> JevWindowResult:
        k = len(docs)
        prepared = [self._prepare_doc(d) for d in docs]
        state_lines = [f"Query: {self._prepare_query(query)}"] + [
            f"{_doc_label(i)}: {prepared[i - 1]}" for i in range(1, k + 1)
        ]
        request = {
            "model": self.model,
            "state": "\n\n".join(state_lines),
            "questions": {
                f"pair_{i}_{j}": {
                    "type": "choice",
                    "instructions": (
                        f"Between {_doc_label(i)} and {_doc_label(j)} "
                        "(both shown in the state), which doc is more relevant "
                        "to the query?"
                    ),
                    "criteria": {"left": _doc_label(i), "right": _doc_label(j)},
                }
                for i, j in combinations(range(1, k + 1), 2)
            },
        }

        start_time = time.perf_counter()
        data = await self._jev._post(request)
        latency_ms = (time.perf_counter() - start_time) * 1000

        edges, probs = [], {}
        answers = data["answers"]
        for i, j in combinations(range(1, k + 1), 2):
            answer = answers[f"pair_{i}_{j}"]
            probs[f"pair_{i}_{j}"] = dict(answer.get("probabilities", {}))
            winner, loser = (i, j) if answer["choice"] == "left" else (j, i)
            edges.append((winner - 1, loser - 1))  # convert to 0-based doc indices

        usage = data.get("usage", {})
        return JevWindowResult(
            edges=edges,
            probabilities=probs,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=latency_ms,
            model=self.model,
            raw_prompt=request,
            raw_response=str(data),
        )
