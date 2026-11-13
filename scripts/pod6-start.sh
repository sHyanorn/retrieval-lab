# The container start command of the step-6 pod, passed verbatim to bash -c.
# Before anything else it sets the two deletions made inside the pod: at
# HARD_LIMIT_MIN minutes, and when this shell exits, whatever the outcome.
# Then it fetches the repository at COMMIT and runs scripts/pod6.sh.
# Environment: COMMIT, HARD_LIMIT_MIN and those of scripts/pod6.sh; WORK
# (default /root) and REPO_URL. Runpod sets RUNPOD_POD_ID and RUNPOD_API_KEY,
# the pod's own key. Without HARD_LIMIT_MIN the pod deletes itself at once.
# With DRY_RUN=1 the deletion is printed.
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
rm -rf "${WORK:-/root}/rl"
git clone -q "${REPO_URL:-https://github.com/sHyanorn/retrieval-lab}" "${WORK:-/root}/rl" &&
	git -C "${WORK:-/root}/rl" checkout -q "$COMMIT" &&
	"${WORK:-/root}/rl/scripts/pod6.sh" "$start"
