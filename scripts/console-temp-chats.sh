#!/usr/bin/env bash
# Run-owned fixture journal. Assumes exclusive console access while validating;
# WDP has no atomic compare-and-swap for chats/index.json. Never use error bodies
# as an empty index. Failed cleanup stays pending for a checked retry.
CHAT_JOURNAL="${XLLAMA_CHAT_JOURNAL:-${TMPDIR_LOCAL}/chat-cleanup.pending}"

register_temp_chat() {
	local suffix cid
	suffix=$(python3 -c 'import uuid; print(uuid.uuid4().hex)') || return 1
	cid="${1}-${suffix}"
	printf '%s\n' "$cid" >>"$CHAT_JOURNAL" || return 1
	printf '%s\n' "$cid"
}

_chat_request() {
	local method="$1" name="$2" out="$3"
	local args=(--connect-timeout 5 --max-time 20 -X "$method" -o "$out" -w "%{http_code}")
	if [[ "$method" != GET ]]; then
		args+=(-H "X-CSRF-Token:${CSRF_TOKEN}")
	fi
	if [[ "$method" == POST ]]; then
		args+=(-F "file=@${4};filename=${name};type=application/octet-stream")
	fi
	curl "${CURL_AUTH[@]}" "${args[@]}" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState%5Cchats&filename=${name}" 2>/dev/null
}

fetch_chat_index() {
	local dest="$1" code
	stop_app || return 1
	code=$(_chat_request GET index.json "$dest") || return 1
	case "$code" in
	200)
		python3 - "$dest" <<'PY' || return 1
import json, sys
from pathlib import Path
try:
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    # Runtime validation must survive PYTHONOPTIMIZE / python -O.
    if not isinstance(data, list) or not all(isinstance(e, dict) for e in data):
        raise ValueError("chat index must be a list of objects")
except (OSError, ValueError):
    print("  FAIL: chat index is unreadable; refusing fixture injection", file=sys.stderr)
    raise SystemExit(1)
PY
		;;
	404) printf '[]' >"$dest" ;;
	*)
		echo "  FAIL: chat index HTTP ${code}; refusing fixture injection" >&2
		return 1
		;;
	esac
}

_remove_temp_chat() {
	local cid="$1" code index="${TMPDIR_LOCAL}/cleanup-index.json"
	local filtered="${TMPDIR_LOCAL}/cleanup-filtered.json" changed="${TMPDIR_LOCAL}/cleanup-changed"
	local reply="${TMPDIR_LOCAL}/cleanup-reply" check="${TMPDIR_LOCAL}/cleanup-check"
	code=$(_chat_request GET index.json "$index") || return 1
	case "$code" in
	200)
		python3 - "$index" "$filtered" "$changed" "$cid" <<'PY' || return 1
import json, sys
from pathlib import Path
src, out, changed, cid = sys.argv[1:]
try:
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(e, dict) for e in data):
        raise ValueError("chat index must be a list of objects")
    kept = [e for e in data if e.get("id") != cid]
    Path(out).write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    Path(changed).write_text("1" if len(kept) != len(data) else "0")
except (OSError, ValueError) as exc:
    print(f"  FAIL: refusing to replace unreadable chat index: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
		;;
	404) printf '0' >"$changed" ;;
	*) return 1 ;;
	esac
	code=$(_chat_request DELETE "${cid}.json" "$reply") || return 1
	[[ "$code" == 200 || "$code" == 404 ]] || return 1
	# A DELETE success is not proof of absence (same WDP defect as #347).
	code=$(_chat_request GET "${cid}.json" "$check") || return 1
	[[ "$code" == 404 ]] || return 1
	if [[ "$(cat "$changed")" == 1 ]]; then
		# Detect an intervening writer before upload. This is not atomic: the
		# documented exclusive-access requirement still applies after this read.
		code=$(_chat_request GET index.json "$check") || return 1
		[[ "$code" == 200 ]] && cmp -s "$index" "$check" || return 1
		code=$(_chat_request POST index.json "$reply" "$filtered") || return 1
		[[ "$code" == 200 ]] || return 1
		code=$(_chat_request GET index.json "$check") || return 1
		[[ "$code" == 200 ]] || return 1
		python3 - "$filtered" "$check" <<'PY' || return 1
import json, sys
from pathlib import Path
try:
    expected = json.loads(Path(sys.argv[1]).read_text())
    actual = json.loads(Path(sys.argv[2]).read_text())
    if expected != actual:
        raise ValueError("chat index read-back does not match the requested update")
except (OSError, ValueError):
    raise SystemExit(1)
PY
	fi
}

remove_chat() {
	local cid="$1"
	[[ "$cid" =~ ^[a-zA-Z0-9_-]+$ ]] || return 1
	[[ -f "$CHAT_JOURNAL" ]] || return 0
	local found=0
	grep -qxF "$cid" "$CHAT_JOURNAL" || found=$?
	[[ "$found" == 1 ]] && return 0
	[[ "$found" == 0 ]] || return 1
	# Do not race the app's SaveConversation/SaveIndex with a host rewrite.
	if ! stop_app || ! _remove_temp_chat "$cid"; then
		echo "  FAIL: temporary chat cleanup pending: ${cid} (journal: ${CHAT_JOURNAL})" >&2
		return 1
	fi
	python3 - "$CHAT_JOURNAL" "$cid" <<'PY'
import sys
from pathlib import Path
p, cid = Path(sys.argv[1]), sys.argv[2]
pending = [line for line in p.read_text().splitlines() if line != cid]
tmp = p.with_name(p.name + ".next")
tmp.write_text("".join(line + "\n" for line in pending))
tmp.replace(p)
PY
}

cleanup_temp_chats() {
	local cid rc=0
	[[ -s "$CHAT_JOURNAL" ]] || return 0
	# Read a snapshot: remove_chat updates the journal only on verified success.
	local pending
	pending=$(cat "$CHAT_JOURNAL") || return 1
	while IFS= read -r cid; do
		[[ -z "$cid" ]] && continue
		remove_chat "$cid" || rc=1
	done <<<"$pending"
	return "$rc"
}
