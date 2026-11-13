# The container start command of the pod that tests the log route: a log as
# large as what step 6 would carry out in it (scripts/logtest.py), passed
# verbatim to bash -c. As scripts/pod5-start.sh, before anything else it sets
# the two deletions made inside the pod: at HARD_LIMIT_MIN minutes, and when
# this shell exits, whatever the outcome. Then it fetches the repository at
# COMMIT, prints the log, and waits GRACE_MIN minutes (default 30), ending at
# least a minute before the time limit, for the log to be read.
# With PHASES set it prints the paced log instead (logtest.py paced), with
# REPRINT, LEAD_S and MAX_PRINT_MIN if set.
# Environment: COMMIT, HARD_LIMIT_MIN, GRACE_MIN; PHASES, REPRINT, LEAD_S,
# MAX_PRINT_MIN; WORK (default /root) and REPO_URL; LOGTEST_ARGS (more
# arguments of logtest.py print or paced, for tests).
# Runpod sets RUNPOD_POD_ID and RUNPOD_API_KEY, the pod's own key. Without
# HARD_LIMIT_MIN the pod deletes itself at once. With DRY_RUN=1 the deletion
# is printed.
set -u
start=$(date +%s)
del() {
	if [ "${DRY_RUN:-0}" = 1 ]; then
		echo "dry run: curl -fsS -X DELETE -H 'Authorization: Bearer \$RUNPOD_API_KEY' https://api.runpod.io/v2/pods/${RUNPOD_POD_ID:-}"
		return
	fi
	for i in 1 2 3; do
		curl -fsS -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" && return
		sleep 20
	done
	runpodctl pod delete "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID"
}
trap 'echo "exit: deleting the pod"; del' EXIT
[ -n "${HARD_LIMIT_MIN:-}" ] || { echo "HARD_LIMIT_MIN is not set"; exit 1; }
(sleep "${HARD_LIMIT_MIN}m" && echo "time limit: deleting the pod" && del) &
work=${WORK:-/root}
rm -rf "$work/rl" "$work/logtest"
echo "pod ${RUNPOD_POD_ID:-local}, started $(date -u -d "@$start" +%FT%TZ)"
git clone -q "${REPO_URL:-https://github.com/sHyanorn/retrieval-lab}" "$work/rl" &&
	git -C "$work/rl" checkout -q "$COMMIT" &&
	echo "commit $(git -C "$work/rl" rev-parse HEAD), $(python3 --version)" &&
	if [ -n "${PHASES:-}" ]; then
		python3 "$work/rl/scripts/logtest.py" paced --phases "$PHASES" ${REPRINT:+--reprint "$REPRINT"} \
			--lead-s "${LEAD_S:-0}" --max-print-min "${MAX_PRINT_MIN:-45}" --start "$start" --limit-min "$HARD_LIMIT_MIN" \
			--grace-min "${GRACE_MIN:-30}" ${LOGTEST_ARGS:-}
	else
		python3 "$work/rl/scripts/logtest.py" print --tmp "$work/logtest" --start "$start" --limit-min "$HARD_LIMIT_MIN" \
			--grace-min "${GRACE_MIN:-30}" ${LOGTEST_ARGS:-}
	fi
