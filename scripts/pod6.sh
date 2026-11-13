#!/usr/bin/env bash
# Sets up the step-6 CPU pod and runs step6/job.py on the network volume;
# called by scripts/pod6-start.sh with the pod's start time. Everything it
# prints also goes to VOL/step6/out/pod-<pod>-<start>.log. At exit, whatever
# the outcome, it prints two reports of step6/report.py, the label-free
# checksum lists and then the small label-free files, so that they leave the
# pod in its log, then waits GRACE_MIN minutes (default 45), within the pod's
# time limit, for the log to be read. The files that hold measures stay on
# the volume for another route.
# Environment: HARD_LIMIT_MIN; VOL_QUOTA_GB, MEM_GB and VCPU, the volume's
# and the pod's sizes as created; VOL (default /workspace/rl), GRACE_MIN,
# DEADLINE_H (hours from the pod's start; default its time limit less the
# grace and a quarter of an hour), WORK (default /root), SKIP_SETUP=1 where
# Go 1.27.3 is installed, JOB_ARGS (more arguments of step6/job.py, for a dry
# run). The binaries, trec_eval, the runs and the copies of the inputs stay
# on the pod's own disk; VOL gets the per-query files, records and logs. The
# summary of an earlier pod is removed first; the job writes it again.
set -euo pipefail
start=${1:-$(date +%s)}
repo=$(cd "$(dirname "$0")/.." && pwd)
vol=${VOL:-/workspace/rl}
work=${WORK:-/root}
out=$vol/step6/out
mkdir -p "$out"
rm -f "$out/summary.json"
exec > >(tee -a "$out/pod-${RUNPOD_POD_ID:-local}-$start.log") 2>&1
cd "$repo"
finish() {
	code=$?
	python3 step6/report.py "$out" --sums || echo "step6/report.py --sums failed"
	python3 step6/report.py "$out" || echo "step6/report.py failed"
	wait=$((${GRACE_MIN:-45} * 60))
	if [ -n "${HARD_LIMIT_MIN:-}" ]; then
		left=$(awk "BEGIN { printf \"%d\", $start + ${HARD_LIMIT_MIN} * 60 - $(date +%s) - 60 }")
		if [ "$left" -lt "$wait" ]; then wait=$left; fi
	fi
	if [ "$wait" -gt 0 ]; then
		echo "exit $code in $wait s"
		sleep "$wait"
	fi
	exit "$code"
}
trap finish EXIT
echo "pod ${RUNPOD_POD_ID:-local}, commit $(git rev-parse HEAD), started $(date -u -d "@$start" +%FT%TZ), nproc $(nproc)"
free -g
df -h "$work"
if [ "${SKIP_SETUP:-0}" != 1 ]; then
	# Go 1.27.3 (D001), checked against go.dev's SHA-256.
	curl -fsSL --retry 5 --retry-all-errors -o /tmp/go.tgz https://go.dev/dl/go1.27.3.linux-amd64.tar.gz
	echo "65efbc80fdc1fa2cd2631cad5b481777a68e8725f34e89f50fc6ac714e379b40  /tmp/go.tgz" | sha256sum -c -
	rm -rf /usr/local/go && tar -C /usr/local -xzf /tmp/go.tgz --no-same-owner
	export PATH=/usr/local/go/bin:$PATH
fi
export GOTOOLCHAIN=local
go version
bin=${BIN:-$work/bin}
go build -o "$bin/" ./cmd/dense ./cmd/eval ./cmd/fuse ./cmd/bm25 ./cmd/study
scripts/trec_eval.sh "${TREC_EVAL_DIR:-$work/trec_eval}"
deadline=${DEADLINE_H:-$(awk "BEGIN { print (${HARD_LIMIT_MIN:?} - ${GRACE_MIN:-45} - 15) / 60 }")}
set +e
python3 step6/job.py --vol "$vol" --bin "$bin" --trec-eval "${TREC_EVAL_DIR:-$work/trec_eval}/trec_eval" --go "$(command -v go)" \
	--vol-quota-gb "${VOL_QUOTA_GB:?}" --mem-gb "${MEM_GB:?}" --vcpu "${VCPU:?}" --local "$work/step6" --start "$start" \
	--deadline-h "$deadline" ${JOB_ARGS:-}
code=$?
echo "step6/job.py exited with $code"
exit $code
