#!/usr/bin/env bash
# Compares cmd/eval with trec_eval, query by query, on one run file.
# The qrels reach trec_eval through awk, not through our reader.
# usage: scripts/check-eval.sh TREC_EVAL EVAL DATASET_DIR RUN
set -euo pipefail
te=${1:?usage: scripts/check-eval.sh TREC_EVAL EVAL DATASET_DIR RUN}
ev=$2 data=$3 run=$4
qrels=$(mktemp)
trap 'rm -f "$qrels"' EXIT
awk -F'\t' 'NR > 1 { sub(/\r$/, ""); print $1, 0, $2, $3 }' "$data/qrels/test.tsv" >"$qrels"
diff <("$te" -c -q -m ndcg_cut.10 -m recall.100 "$qrels" "$run") <("$ev" -data "$data" -run "$run")
