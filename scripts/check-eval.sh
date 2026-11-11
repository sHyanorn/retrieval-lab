#!/usr/bin/env bash
# Compares cmd/eval with trec_eval, query by query, on one run file. Fails
# if either tool fails or prints nothing, or if their outputs differ.
# The qrels reach trec_eval through awk, not through our reader.
# usage: scripts/check-eval.sh TREC_EVAL EVAL DATASET_DIR RUN
set -euo pipefail
te=${1:?usage: scripts/check-eval.sh TREC_EVAL EVAL DATASET_DIR RUN}
ev=$2 data=$3 run=$4
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
awk -F'\t' 'NR > 1 { sub(/\r$/, ""); print $1, 0, $2, $3 }' "$data/qrels/test.tsv" >"$tmp/qrels"
"$te" -c -q -m ndcg_cut.10 -m recall.100 "$tmp/qrels" "$run" >"$tmp/trec_eval"
"$ev" -data "$data" -run "$run" >"$tmp/eval"
[ -s "$tmp/trec_eval" ] || { echo "check-eval: trec_eval printed nothing" >&2; exit 1; }
diff "$tmp/trec_eval" "$tmp/eval"
