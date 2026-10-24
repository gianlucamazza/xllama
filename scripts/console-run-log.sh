#!/usr/bin/env bash
# Shared by the console harnesses. Source after SCRIPT_DIR and TMPDIR_LOCAL
# are set. Transport uses the caller's Device Portal connection, never secrets
# stored in the repository. Snapshot AFTER checked stop and BEFORE starting;
# never delete the log, since an accepted unlink may still be pending in WDP.

_download_run_log() {
	local dest="$1" code
	: >"$dest"
	code=$(curl "${CURL_AUTH[@]}" -o "$dest" -w "%{http_code}" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=xllama.log" 2>/dev/null) || return 1
	printf '%s' "$code"
}

reset_run_log() {
	: >"${TMPDIR_LOCAL}/xllama-log-ready"
}

snapshot_run_log() {
	local before="${TMPDIR_LOCAL}/xllama-before.log" code
	# Invalidate the last run before trying the next snapshot (including when a
	# caller suppresses errexit, as the autopilot command substitutions do).
	reset_run_log
	code=$(_download_run_log "$before") || {
		echo "  FAIL: could not snapshot the pre-run log" >&2
		return 1
	}
	case "$code" in
	200) ;;
	404) : >"$before" ;;
	*)
		echo "  FAIL: pre-run log returned HTTP ${code}" >&2
		return 1
		;;
	esac
	printf 'ready' >"${TMPDIR_LOCAL}/xllama-log-ready"
}

fetch_log() {
	local full="${TMPDIR_LOCAL}/xllama-full.log" run="${TMPDIR_LOCAL}/xllama.log" code
	: >"$run"
	if [[ ! -f "${TMPDIR_LOCAL}/xllama-log-ready" ]] ||
		[[ "$(cat "${TMPDIR_LOCAL}/xllama-log-ready")" != "ready" ]]; then
		echo "  FAIL: no verified pre-run log snapshot" >&2
		return 1
	fi
	code=$(_download_run_log "$full") || {
		echo "  FAIL: could not download this run's log" >&2
		return 1
	}
	if [[ "$code" != "200" ]]; then
		echo "  FAIL: run log returned HTTP ${code}" >&2
		return 1
	fi
	python3 "${SCRIPT_DIR}/console_run_log.py" \
		"${TMPDIR_LOCAL}/xllama-before.log" "$full" "$run" || return 1
	echo "$run"
}
