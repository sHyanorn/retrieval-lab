#!/bin/sh
# The CPU reference vectors of check 1 (docs/PREREGISTRATION.md, Section 5):
# each model of the study encodes the first 1,000 documents and all test
# queries of SciFact on the CPU, with encode/encode.py, into OUT/<model>.
# A model whose summary OUT/<model>.json exists is skipped, and encode.py
# resumes a model after its last whole chunk, so an interrupted run goes on
# where it stopped. Then the SHA-256 of every vector file is printed.
#
# usage: scripts/reference.sh PYTHON SCIFACT_DIR OUT
set -eu
py=$1 data=$2 out=$3
mkdir -p "$out"
for m in sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 \
	BAAI/bge-base-en-v1.5 intfloat/e5-base-v2 thenlper/gte-base \
	Snowflake/snowflake-arctic-embed-m-v1.5 BAAI/bge-large-en-v1.5 \
	mixedbread-ai/mxbai-embed-large-v1; do
	d=$out/$(echo "$m" | tr / _)
	[ -s "$d.json" ] && continue
	"$py" encode/encode.py --model "$m" --data "$data" --out "$d" --limit 1000 --device cpu --chunk 100 >"$d.json.tmp"
	mv "$d.json.tmp" "$d.json"
done
cd "$out" && sha256sum -- */index */docid */queries/index */queries/docid
