# The container start command of the pod that carries step 6's files out by
# its log (scripts/carry6.py print), passed verbatim to bash -c. As
# scripts/logtest-start.sh, before anything else it sets the two deletions
# made inside the pod: at HARD_LIMIT_MIN minutes, and when this shell exits,
# whatever the outcome. Then it fetches the repository at COMMIT onto the
# pod's own disk and prints the files that VOL/step6/out/manifest.tsv names,
# after checking each one's size and SHA-256, and waits GRACE_MIN minutes
# (default 5), ending at least a minute before the time limit, for the log to
# be read. The volume is only read.
# Environment: COMMIT, HARD_LIMIT_MIN, GRACE_MIN; VOL (default /workspace/rl);
# CHUNK_KIB (256), PAUSE (0.6), LEAD_S (60), REPRINT, MAX_PRINT_MIN (45);
# WORK (default /root) and REPO_URL; CARRY_ARGS (more arguments of carry6.py
# print, for tests). Runpod sets RUNPOD_POD_ID and RUNPOD_API_KEY, the pod's
# own key. Without HARD_LIMIT_MIN the pod deletes itself at once. With
# DRY_RUN=1 the deletion is printed.
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
rm -rf "$work/rl"
echo "pod ${RUNPOD_POD_ID:-local}, started $(date -u -d "@$start" +%FT%TZ)"
git clone -q "${REPO_URL:-https://github.com/sHyanorn/retrieval-lab}" "$work/rl" &&
	git -C "$work/rl" checkout -q "$COMMIT" &&
	echo "commit $(git -C "$work/rl" rev-parse HEAD), $(python3 --version)" &&
	python3 "$work/rl/scripts/carry6.py" print --out "${VOL:-/workspace/rl}/step6/out" --chunk-kib "${CHUNK_KIB:-256}" \
		--pause "${PAUSE:-0.6}" --lead-s "${LEAD_S:-60}" ${REPRINT:+--reprint "$REPRINT"} \
		--max-print-min "${MAX_PRINT_MIN:-45}" --start "$start" --limit-min "$HARD_LIMIT_MIN" \
		--grace-min "${GRACE_MIN:-5}" ${CARRY_ARGS:-}
