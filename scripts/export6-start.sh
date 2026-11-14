# The container start command of the pod that exports the files kept from the
# network volume (scripts/export6.py), passed verbatim to bash -c. As
# scripts/carry6-start.sh, before anything else it sets the two deletions made
# inside the pod: at HARD_LIMIT_MIN minutes, and when this shell exits,
# whatever the outcome. Then it fetches the repository at COMMIT onto the
# pod's own disk and runs one of three modes. The volume is only read.
#   MODE=du: prints the bytes and file counts of every part of VOL
#   (export6.py du), then waits GRACE_MIN minutes for the log to be read.
#   MODE=export: packs, on the pod's own disk under EXPORT, the files that
#   results/step6/perquery.sha256 names under VOL/step6 (step6-perquery) and
#   those that results/step5/runs.sha256 names under VOL/step5 (step5-runs),
#   each file checked against its list, into parts of at most PART_MIB MiB,
#   writes their manifest and prints them by scripts/carry6.py print, then
#   waits GRACE_MIN minutes.
#   MODE=serve: packs as MODE=export does, prints the manifest and its
#   SHA-256, and serves the manifest and its files on port PORT
#   (export6.py serve) under the path SERVE_TOKEN, until the pod is deleted.
# Environment: MODE, COMMIT, HARD_LIMIT_MIN, GRACE_MIN (default 5); VOL
# (default /workspace/rl); SETS (default "step6 step5"); PART_MIB (1945);
# EXPORT (default WORK/export); CHUNK_KIB (256), PAUSE (0.6), LEAD_S (60),
# REPRINT, MAX_PRINT_MIN (45); SERVE_TOKEN, 16 or more letters and digits,
# and PORT (8000); WORK (default /root) and REPO_URL; LIST6 and LIST5, other
# checksum lists, for tests only. Runpod sets RUNPOD_POD_ID and
# RUNPOD_API_KEY, the pod's own key. Without HARD_LIMIT_MIN the pod deletes
# itself at once. With DRY_RUN=1 the deletion is printed.
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
case "${MODE:-}" in du | export | serve) ;; *) echo "MODE must be du, export or serve"; exit 1 ;; esac
if [ "$MODE" = serve ] && ! printf %s "${SERVE_TOKEN:-}" | grep -qxE '[A-Za-z0-9]{16,}'; then
	echo "SERVE_TOKEN must be 16 or more letters and digits"
	exit 1
fi
work=${WORK:-/root}
vol=${VOL:-/workspace/rl}
rm -rf "$work/rl"
echo "pod ${RUNPOD_POD_ID:-local}, started $(date -u -d "@$start" +%FT%TZ), mode $MODE"
git clone -q "${REPO_URL:-https://github.com/sHyanorn/retrieval-lab}" "$work/rl" &&
	git -C "$work/rl" checkout -q "$COMMIT" &&
	echo "commit $(git -C "$work/rl" rev-parse HEAD), $(python3 --version)" || exit 1
x="$work/rl/scripts/export6.py"
if [ "$MODE" = du ]; then
	python3 "$x" du --vol "$vol" || exit 1
	wait=$(awk "BEGIN { w = ${GRACE_MIN:-5} * 60; l = $start + $HARD_LIMIT_MIN * 60 - 60 - $(date +%s); printf \"%d\", (l < w ? l : w) }")
	if [ "$wait" -gt 0 ]; then
		echo "waiting $wait s for the log to be read"
		sleep "$wait"
	fi
	exit 0
fi
out=${EXPORT:-$work/export}
rm -rf "$out"
part=$((${PART_MIB:-1945} * 1048576))
for s in ${SETS:-step6 step5}; do
	case "$s" in
	step6) python3 "$x" pack --base "$vol/step6" --list "${LIST6:-$work/rl/results/step6/perquery.sha256}" --name step6-perquery \
		--out "$out" --part-bytes "$part" || exit 2 ;;
	step5) python3 "$x" pack --base "$vol/step5" --list "${LIST5:-$work/rl/results/step5/runs.sha256}" --name step5-runs \
		--out "$out" --part-bytes "$part" || exit 2 ;;
	*) echo "SETS: unknown set $s"; exit 1 ;;
	esac
done
python3 "$x" manifest --out "$out" || exit 1
df -h "$out" | tail -1
if [ "$MODE" = serve ]; then
	echo "=== export6 manifest ==="
	cat "$out/manifest.tsv"
	echo "=== end export6 manifest ==="
	(cd "$out" && sha256sum manifest.tsv)
	python3 "$x" serve --out "$out" --token "$SERVE_TOKEN" --port "${PORT:-8000}"
	exit 1
fi
python3 "$work/rl/scripts/carry6.py" print --out "$out" --chunk-kib "${CHUNK_KIB:-256}" \
	--pause "${PAUSE:-0.6}" --lead-s "${LEAD_S:-60}" ${REPRINT:+--reprint "$REPRINT"} \
	--max-print-min "${MAX_PRINT_MIN:-45}" --start "$start" --limit-min "$HARD_LIMIT_MIN" \
	--grace-min "${GRACE_MIN:-5}"
