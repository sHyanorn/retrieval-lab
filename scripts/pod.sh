#!/usr/bin/env bash
# Sets up the step-4 pod and runs encode/job.py on the network volume; called
# by scripts/pod-start.sh with the pod's start time. Everything it prints also
# goes to VOL/step4/out/pod-<pod>-<start>.log. At exit, whatever the outcome,
# it prints the small files of step4/out with encode/report.py, so that they
# leave the pod in its log, then waits GRACE_MIN minutes (default 5), within
# the pod's time limit, for the log to be read while the pod lives.
# Environment: HARD_LIMIT_MIN, PRICE; VOL (default /workspace/rl), LINE
# (default 17.9), DEADLINE_H (the job's time limit, default an hour less than
# the pod's), GRACE_MIN, DRY_RUN=1 to skip the set-up and run job.py --dry-run.
set -euo pipefail
start=${1:-$(date +%s)}
repo=$(cd "$(dirname "$0")/.." && pwd)
vol=${VOL:-/workspace/rl}
out=$vol/step4/out
mkdir -p "$out"
exec > >(tee -a "$out/pod-${RUNPOD_POD_ID:-local}-$start.log") 2>&1
cd "$repo"
finish() {
	code=$?
	python3 encode/report.py "$out" || echo "encode/report.py failed"
	wait=$((${GRACE_MIN:-5} * 60))
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
echo "pod ${RUNPOD_POD_ID:-local}, commit $(git rev-parse HEAD), started $(date -u -d "@$start" +%FT%TZ)"
deadline=${DEADLINE_H:-$(awk "BEGIN { print $HARD_LIMIT_MIN / 60 - 1 }")}
args=(--vol "$vol" --price "$PRICE" --line "${LINE:-17.9}" --deadline-h "$deadline" --pod-start "$start" --agree "$repo/bin/agree")

if [ "${DRY_RUN:-0}" = 1 ]; then
	py=${PYTHON:-python3}
	args+=(--dry-run)
else
	nvidia-smi
	# Go 1.27.3 (D001), checked against go.dev's SHA-256.
	curl -fsSL --retry 5 --retry-all-errors -o /tmp/go.tgz https://go.dev/dl/go1.27.3.linux-amd64.tar.gz
	echo "65efbc80fdc1fa2cd2631cad5b481777a68e8725f34e89f50fc6ac714e379b40  /tmp/go.tgz" | sha256sum -c -
	rm -rf /usr/local/go && tar -C /usr/local -xzf /tmp/go.tgz
	export PATH=/usr/local/go/bin:$PATH GOTOOLCHAIN=local
	# Python 3.13; torch 2.15.0 built for CUDA 13.2, then the pinned packages.
	python3 -c 'import sys; assert sys.version_info[:2] == (3, 13), sys.version'
	python3 -m venv /root/venv
	py=/root/venv/bin/python
	"$py" -m pip install -q --index-url https://download.pytorch.org/whl/cu132 torch==2.15.0
	"$py" -m pip install -q -r encode/requirements.txt
	"$py" -c 'import torch; assert torch.__version__ == "2.15.0+cu132" and torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
	command -v unzip >/dev/null || { apt-get update -qq && apt-get install -y -qq unzip; }
	args+=(--device cuda)
fi
go build -o bin/agree ./cmd/agree
export HF_HOME=$vol/hf
set +e
"$py" encode/job.py "${args[@]}" --python "$py"
code=$?
echo "job.py exited with $code"
exit $code
