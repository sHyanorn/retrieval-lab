#!/usr/bin/env bash
# Downloads into DIR what the BGE-base runs need: the ONNX model Anserini's
# BgeBaseEn15Encoder uses, the BAAI tokenizer.json, ONNX Runtime 1.31.1, and,
# only to compare vectors, Pyserini's prebuilt faiss flat index of each NAME.
# Everything is checked against scripts/dense.sha256.
# usage: scripts/dense.sh DIR [NAME...]
set -euo pipefail
dir=${1:?usage: scripts/dense.sh DIR [NAME...]}
shift
sums=$(cd "$(dirname "$0")" && pwd)/dense.sha256
get() { [ -f "$2" ] || { curl -fsSL --retry 5 --retry-all-errors --connect-timeout 30 --max-time 1800 -o "$2.part" "$1" && mv "$2.part" "$2"; }; }
check() { grep " $1\$" "$sums" | sha256sum -c -; }
mkdir -p "$dir/prebuilt"
cd "$dir"
get https://huggingface.co/castorini/bge-base-en-v1.5-onnx/resolve/c8a4305d344e3f74e43710ae814e5b168a751313/bge-base-en-v1.5-optimized.onnx bge-base-en-v1.5-optimized.onnx
get https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/tokenizer.json tokenizer.json
get https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-linux-x64-1.31.1.tgz onnxruntime-linux-x64-1.31.1.tgz
for f in bge-base-en-v1.5-optimized.onnx tokenizer.json onnxruntime-linux-x64-1.31.1.tgz; do check "$f"; done
[ -d onnxruntime-linux-x64-1.31.1 ] || tar xzf onnxruntime-linux-x64-1.31.1.tgz
for n in "$@"; do
	f=prebuilt/$n.tar.gz
	get "https://rgw.cs.uwaterloo.ca/pyserini/indexes/faiss/faiss-flat.beir-v1.0.0-$n.bge-base-en-v1.5.20240107.tar.gz" "$f"
	check "$f"
	[ -d "prebuilt/$n" ] || { mkdir "prebuilt/$n" && tar xzf "$f" -C "prebuilt/$n" --strip-components 1; }
done
