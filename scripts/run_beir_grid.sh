#!/usr/bin/env bash
# Run the Jev- and GPT-6-Luna-based BlitzRank sweeps on the 8 BEIR paper
# benchmarks. Each run also caches its BM25 hits under .cache/datasets for
# reuse. Plot the results afterwards with scripts/plot_pareto.py. Usage:
#   bash scripts/run_beir_grid.sh docker   # runs inside the blitzrank image
#   bash scripts/run_beir_grid.sh native   # runs via uv on the host
set -euo pipefail

# BEIR dataset name / pyserini index
# (bash 3.2 on stock macOS has no associative arrays; plain lists below)
BEIR_DATASETS="
trec-covid        beir-v1.0.0-trec-covid-test        beir-v1.0.0-trec-covid.flat
nfcorpus          beir-v1.0.0-nfcorpus-test          beir-v1.0.0-nfcorpus.flat
signal1m          beir-v1.0.0-signal1m-test          beir-v1.0.0-signal1m.flat
trec-news         beir-v1.0.0-trec-news-test         beir-v1.0.0-trec-news.flat
robust04          beir-v1.0.0-robust04-test          beir-v1.0.0-robust04.flat
webis-touche2020  beir-v1.0.0-webis-touche2020-test  beir-v1.0.0-webis-touche2020.flat
dbpedia-entity    beir-v1.0.0-dbpedia-entity-test    beir-v1.0.0-dbpedia-entity.flat
scifact           beir-v1.0.0-scifact-test           beir-v1.0.0-scifact.flat
"
CONFIG=${CONFIG:-configs/jev_tournament_graph.yaml}

# (display-name, comparer type, model) — one BlitzRank sweep per oracle
COMPARERS="
jev            jev_api           jev-latest
gpt6-luna      listwise_rank_gpt openai/gpt-6-luna
"

run_one() {
  local ds_name=$1 ds_index=$2 cmp_type=$3 cmp_model=$4
  echo "=== $ds_name [$cmp_model] ==="
  if [ "$RUNNER" = "docker" ]; then
    docker run --rm --env-file .env \
      -v "$(pwd)/blitzrank_outputs:/app/blitzrank_outputs" \
      -v "$(pwd)/.cache/pyserini:/root/.cache/pyserini" \
      -v "$(pwd)/.cache/datasets:/app/.cache/datasets" \
      blitzrank python -m blitzrank.engine.run "$CONFIG" \
      --set "dataset.name=$ds_name" --set "dataset.index=$ds_index" \
      --set "reranking.comparer.type=$cmp_type" \
      --set "reranking.comparer.model=$cmp_model" \
      --set logging.output_dir=blitzrank_outputs
  else
    uv run --python 3.11 --extra all python -m blitzrank.engine.run "$CONFIG" \
      --set "dataset.name=$ds_name" --set "dataset.index=$ds_index" \
      --set "reranking.comparer.type=$cmp_type" \
      --set "reranking.comparer.model=$cmp_model"
  fi
}

RUNNER=${1:-docker}
echo "$BEIR_DATASETS" | while read -r _ ds_name ds_index; do
  [ -n "$ds_name" ] || continue
  echo "$COMPARERS" | while read -r cmp_name cmp_type cmp_model; do
    [ -n "$cmp_name" ] || continue
    ok=""
    for attempt in 1 2 3; do
      if run_one "$ds_name" "$ds_index" "$cmp_type" "$cmp_model"; then ok=1; break; fi
      echo "### $cmp_name for $ds_name failed (attempt $attempt/3); retrying in $((attempt * 30))s" >&2
      sleep $((attempt * 30))
    done
    [ -n "$ok" ] || echo "### GIVING UP on $ds_name [$cmp_name]; continuing with the rest" >&2
  done
done
