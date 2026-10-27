"""Batched-Jev compare oracle: all pairwise window edges from one API call."""
from typing import List

from ...config import ComparerConfig
from ...comparer.jev_api import JevApiWindowComparer
from .compare_oracle import CompareOracle, OracleResult
from .experimental_interface import ContentItem


class JevWindowOracle(CompareOracle):
    def __init__(self, k: int, comparer_config: ComparerConfig):
        super().__init__(k)
        self.comparer = JevApiWindowComparer(
            model=comparer_config.model,
            max_doc_tokens=comparer_config.max_doc_tokens,
            max_query_tokens=comparer_config.max_query_tokens,
        )
        self.input_tokens_total = 0
        self.output_tokens_total = 0
        self.thought_tokens_total = 0  # always 0 for Jev; kept for log-schema parity
        self.llm_latency_ms_total = 0.0
        self.num_trimmed_docs_total = 0  # stays 0; kept for log-schema parity
        self.call_logs: list[dict] = []

    async def _compare_k_async(self, items: List[ContentItem]) -> OracleResult:
        if len(items) < 2:
            # Small-corpus datasets (e.g. nfcorpus) return 0- or 1-hit queries;
            # the API 422s on zero-pair windows, and there is nothing to compare.
            return OracleResult(edges=[], metadata={"num_pairs_decided": 0})
        results = await self.comparer.compare(
            query=items[0].query,
            docs=[item.content.content for item in items],
        )
        self.input_tokens_total += results.input_tokens
        self.output_tokens_total += results.output_tokens
        self.thought_tokens_total += 0
        self.llm_latency_ms_total += results.latency_ms
        call_log = {
            "input_tokens": results.input_tokens,
            "output_tokens": results.output_tokens,
            "thought_tokens": 0,
            "latency_ms": results.latency_ms,
            "missing_indices": [],
            "duplicate_indices": [],
            "num_trimmed_docs": 0,
            "num_pairs_decided": len(results.edges),
        }
        self.call_logs.append(call_log)

        edges = [(items[w], items[l]) for w, l in results.edges]
        return OracleResult(edges=edges, metadata=call_log)
