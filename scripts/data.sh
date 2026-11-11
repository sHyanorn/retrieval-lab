#!/usr/bin/env bash
# Downloads BEIR datasets into DIR, verifies the zips against
# scripts/data.sha256 and unpacks them into DIR/<name>/.
# usage: scripts/data.sh DIR [NAME...]   (default: all six)
set -euo pipefail
dir=${1:?usage: scripts/data.sh DIR [NAME...]}
shift
names=("$@")
[ ${#names[@]} -gt 0 ] || names=(scifact nfcorpus arguana scidocs fiqa trec-covid)
base=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets
sums=$(cd "$(dirname "$0")" && pwd)/data.sha256
mkdir -p "$dir"
cd "$dir"
for n in "${names[@]}"; do
	[ -f "$n.zip" ] || { curl -fsSL --retry 5 -o "$n.zip.part" "$base/$n.zip" && mv "$n.zip.part" "$n.zip"; }
	grep " $n.zip\$" "$sums" | sha256sum -c -
	[ -d "$n" ] || unzip -q "$n.zip"
done
