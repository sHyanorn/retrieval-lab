#!/usr/bin/env bash
# Sets up the step-5 CPU pod and runs step5/job.py on the network volume;
# called by scripts/pod5-start.sh with the pod's start time. Everything it
# prints also goes to VOL/step5/out/pod-<pod>-<start>.log. At exit, whatever
# the outcome, it prints the small files of step5/out with step5/report.py,
# so that they leave the pod in its log, then waits GRACE_MIN minutes
# (default 45), within the pod's time limit, for the log to be read.
# Environment: HARD_LIMIT_MIN, STEP4 (step 4's committed files, a directory
# of the repository); VOL (default /workspace/rl), WORKERS (default the CPUs),
# GRACE_MIN, DEADLINE_H (default the pod's time limit less the grace and a
# quarter of an hour), SKIP_SETUP=1 where Go 1.27.3 is installed. The
# binaries, trec_eval and the BM25 indexes stay on the pod's own disk; only
# run files, per-query measures and records go to VOL.
set -euo pipefail
start=${1:-$(date +%s)}
repo=$(cd "$(dirname "$0")/.." && pwd)
vol=${VOL:-/workspace/rl}
out=$vol/step5/out
mkdir -p "$out"
exec > >(tee -a "$out/pod-${RUNPOD_POD_ID:-local}-$start.log") 2>&1
cd "$repo"
finish() {
	code=$?
	python3 step5/report.py "$out" || echo "step5/report.py failed"
	wait=$((${GRACE_MIN:-45} * 60))
	if [ -n "${HARD_LIMIT_MIN:-}" ]; then
		left=$((start + HARD_LIMIT_MIN * 60 - $(date +%s)))
		if [ "$left" -lt "$wait" ]; then wait=$left; fi
	fi
	if [ "$wait" -gt 0 ]; then
		echo "exit $code in $wait s"
		sleep "$wait"
	fi
	exit "$code"
}
trap finish EXIT
echo "pod ${RUNPOD_POD_ID:-local}, commit $(git rev-parse HEAD), started $(date -u -d "@$start" +%FT%TZ), $(nproc) CPUs"
free -g
df -h "$vol" /root
if [ "${SKIP_SETUP:-0}" != 1 ]; then
	# Go 1.27.3 (D001), checked against go.dev's SHA-256.
	curl -fsSL --retry 5 --retry-all-errors -o /tmp/go.tgz https://go.dev/dl/go1.27.3.linux-amd64.tar.gz
	echo "65efbc80fdc1fa2cd2631cad5b481777a68e8725f34e89f50fc6ac714e379b40  /tmp/go.tgz" | sha256sum -c -
	rm -rf /usr/local/go && tar -C /usr/local -xzf /tmp/go.tgz --no-same-owner
	command -v unzip >/dev/null || { apt-get update -qq && apt-get install -y -qq unzip; }
fi
export PATH=/usr/local/go/bin:$PATH GOTOOLCHAIN=local
go version
bin=${BIN:-/root/bin}
go build -o "$bin/" ./cmd/dense ./cmd/eval ./cmd/bm25
scripts/trec_eval.sh "${TREC_EVAL_DIR:-/root/trec_eval}"
step4=${STEP4:?STEP4 names the committed files of step 4}
[[ $step4 = /* ]] || step4=$repo/$step4
deadline=${DEADLINE_H:-$(awk "BEGIN { print (${HARD_LIMIT_MIN:?} - ${GRACE_MIN:-45} - 15) / 60 }")}
set +e
python3 step5/job.py --vol "$vol" --step4 "$step4" --bin "$bin" --trec-eval "${TREC_EVAL_DIR:-/root/trec_eval}/trec_eval" \
	--local "${LOCAL:-/root}" --workers "${WORKERS:-$(nproc)}" --deadline-h "$deadline"
code=$?
echo "step5/job.py exited with $code"
exit $code
