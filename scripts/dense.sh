#!/usr/bin/env bash
# Downloads into DIR what the BGE-base runs need: the ONNX model Anserini's
# BgeBaseEn15Encoder uses, the BAAI tokenizer.json, ONNX Runtime 1.31.1 (its
# PyPI wheel, of which only the shared library is unpacked), and, only to
# compare vectors, Pyserini's prebuilt faiss flat index of each NAME.
# Everything is checked against scripts/dense.sha256. With PREBUILT_ONLY=1
# only the prebuilt indexes are downloaded. An archive counts as unpacked once
# its .ok file exists; without it the directory is unpacked again. Owners in
# the archives are not kept, as a network volume refuses the change.
# usage: scripts/dense.sh DIR [NAME...]
set -euo pipefail
dir=${1:?usage: scripts/dense.sh DIR [NAME...]}
shift
sums=$(cd "$(dirname "$0")" && pwd)/dense.sha256
get() { [ -f "$2" ] || { curl -fsSL --retry 5 --retry-all-errors --connect-timeout 30 --max-time 1800 -o "$2.part" "$1" && mv "$2.part" "$2"; }; }
check() { grep " $1\$" "$sums" | sha256sum -c -; }
mkdir -p "$dir/prebuilt"
cd "$dir"
if [ "${PREBUILT_ONLY:-0}" != 1 ]; then
get https://huggingface.co/castorini/bge-base-en-v1.5-onnx/resolve/c8a4305d344e3f74e43710ae814e5b168a751313/bge-base-en-v1.5-optimized.onnx bge-base-en-v1.5-optimized.onnx
get https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/tokenizer.json tokenizer.json
whl=onnxruntime-1.31.1-cp313-cp313-manylinux_2_28_x86_64.whl
get https://files.pythonhosted.org/packages/fc/0a/c887bcb84a9455b2723825381a77af38611a80f996f804e20fde5f671b63/$whl $whl
for f in bge-base-en-v1.5-optimized.onnx tokenizer.json $whl; do check "$f"; done
[ -f onnxruntime-1.31.1.ok ] || { rm -rf onnxruntime-1.31.1 && unzip -q $whl onnxruntime/capi/libonnxruntime.so.1.31.1 -d onnxruntime-1.31.1 && touch onnxruntime-1.31.1.ok; }
fi
for n in "$@"; do
	f=prebuilt/$n.tar.gz
	get "https://rgw.cs.uwaterloo.ca/pyserini/indexes/faiss/faiss-flat.beir-v1.0.0-$n.bge-base-en-v1.5.20240107.tar.gz" "$f"
	check "$f"
	[ -f "prebuilt/$n.ok" ] || { rm -rf "prebuilt/$n" && mkdir "prebuilt/$n" && tar xzf "$f" --no-same-owner -C "prebuilt/$n" --strip-components 1 && touch "prebuilt/$n.ok"; }
done
