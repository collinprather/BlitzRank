"""Run the Jev-based BlitzRank on a single query and print everything:
the final reranking plus every tournament round's window and edge decisions.

Examples:
  # first DL19 query (auto-loads the host-mounted .cache dataset cache)
  python scripts/demo_jev_query.py

  # a specific query by its qid, top 5
  python scripts/demo_jev_query.py --qid 264014 --topk 5
"""
import argparse
import asyncio
import os
from pathlib import Path

import orjson

from blitzrank.engine.algorithms.tournament_graph.experimental_interface import (
    ContentItem,
    RerankTask,
    SingleContent,
    TournamentGraphRerankConfig,
    rerank_with_tournament_graph,
)
from blitzrank.engine.algorithms.tournament_graph.jev_oracle import JevWindowOracle
from blitzrank.engine.algorithms.tournament_graph.types import (
    TournamentGraphSortConfig,
)
from blitzrank.engine.config import ComparerConfig, ComparerType


def load_cached_dataset(dataset_name: str, index: str, k: int = 100) -> dict:
    cache = Path(".cache") / "datasets" / dataset_name / f"{index}_bm25flat_k{k}.json"
    if not cache.exists():
        raise SystemExit(
            f"{cache} not found; run the dataset download first "
            f"(e.g. bash scripts/run_beir_grid.sh docker, or any single benchmark run)"
        )
    return orjson.loads(cache.read_bytes())["data"]


async def run(query: str, docs: list[str], topk: int, window_size: int, model: str):
    task = RerankTask(
        query=query,
        contents=[
            SingleContent(content=d, qid="0", docid=str(i), rank=i + 1,
                          score=len(docs) - i, orig_idx=i)
            for i, d in enumerate(docs)
        ],
        hits=[],
    )
    events = []

    def on_round_complete(round_log: dict):
        round_log = dict(round_log)
        stats = round_log.pop("oracle_metadata") or {}
        events.append({"nodes": round_log.pop("nodes_for_match", []), **stats, **round_log})

    oracle = JevWindowOracle(window_size, ComparerConfig(type=ComparerType.JEV_API, model=model))
    result = await rerank_with_tournament_graph(
        task,
        TournamentGraphRerankConfig(top_m=topk, oracle=oracle),
        TournamentGraphSortConfig(on_round_complete=on_round_complete),
    )
    return result, oracle, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="dl19-passage", help="dataset name (e.g. dl20-passage)")
    parser.add_argument("--index", default="msmarco-v1-passage", help="pyserini index")
    parser.add_argument("--k", type=int, default=100, help="BM25 retrieval depth; must match the cache")
    parser.add_argument("--qid", default=None, help="pick query by qid; default = first query")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--model", default="jev-latest")
    args = parser.parse_args()

    data = load_cached_dataset(args.name, args.index, args.k)
    entry = next((e for e in data["results"] if e["hits"] and e["hits"][0]["qid"] == args.qid),
                 data["results"][0])
    docs = [hit["content"] for hit in entry["hits"]]
    print(f"Query [{entry['hits'][0]['qid']}]: {entry['query']}\n")

    result, oracle, events = asyncio.run(run(entry["query"], docs, args.topk, args.window_size, args.model))

    print(f"Tournament rounds: {len(events)}  |  Jev calls: {oracle.calls}"
          f"  |  input tokens: {oracle.input_tokens_total:,}"
          f"  |  pairs decided: {sum(e['num_pairs_decided'] for e in events):,}\n")
    for i, ev in enumerate(events, 1):
        print(f"--- Round {i}: window={[f'#{n}' for n in ev['nodes']]} ---")
        print(f"    {ev['num_pairs_decided']} pairwise decisions,"
              f"  {ev['input_tokens']:,}/{ev['output_tokens']:,} in/out tokens:"
              f"  {(ev.get('latency_ms') or 0) / 1000:.2f}s")

    print(f"\nFinal top {args.topk} (Jev reranking):")
    print(f"{'BM25':>4}  {'docid':>9}  doc snippet")
    for item in result.results:
        doc = item.content
        snippet = " ".join(doc.content.split())[:90]
        print(f"{doc.rank:>4}  {doc.docid:>9}  {snippet}")

    print(f"\ntotals: {oracle.input_tokens_total:,} input / {oracle.output_tokens_total:,} output tokens"
          f"  |  latency {oracle.llm_latency_ms_total / 1000:.1f}s")
    qid = str(entry["hits"][0]["qid"])
    if qid in data["qrels"]:
        gold = data["qrels"][qid]
        print(f"qrels for this query: {len(gold)} judged docs"
              f"  |  qid lookup: blitzrank docids are the BM25 hit docids")


if __name__ == "__main__":
    main()
