#!/usr/bin/env bash
# Builds trec_eval v10.0 into DIR/trec_eval. It is used only to check
# cmd/eval, never by the pipeline itself.
# usage: scripts/trec_eval.sh DIR
set -euo pipefail
dir=${1:?usage: scripts/trec_eval.sh DIR}
commit=f4253652c8efd0d86ddffd0d163cc0a0f813111a
[ -x "$dir/trec_eval" ] && exit 0
git -c advice.detachedHead=false clone -q --depth 1 --branch v10.0 https://github.com/usnistgov/trec_eval "$dir"
[ "$(git -C "$dir" rev-parse HEAD)" = "$commit" ] || { echo "trec_eval: v10.0 is not $commit" >&2; exit 1; }
make -s -C "$dir" >/dev/null
