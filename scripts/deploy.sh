#!/usr/bin/env bash
# deploy.sh — interact with the Xbox Device Portal
#
# Sub-commands:
#   deploy.sh <package.msix>                             Upload and install .msix (+ auto-install .cer)
#   deploy.sh install-cert <cert.cer>                    Install a trust certificate on the console
#   deploy.sh upload-file <local> <pfn> [remote-dir] [remote-name]  Upload to LocalFolder (auto-creates subdir)
#   deploy.sh upload-dir <local-dir> <pfn> <remote-dir>  Upload all files in a directory to LocalFolder/<remote-dir>/
#   deploy.sh mkdir-localstate <pfn> <relpath>           Create directory in LocalState (e.g. models\Phi-3.5)
#   deploy.sh pfn   (highest registered version; warns on stderr if several are live)                                        Print installed xllama package full name
#   deploy.sh get-log [pfn]                              Print LocalState/xllama.log
#   deploy.sh fetch-file <pfn> <name> <local-out> [subdir]  Download a LocalState file
#   deploy.sh delete-file <pfn> <name> [subdir]          Delete a LocalState file (best-effort)
#   deploy.sh list-localstate [pfn]                      List app LocalState files
#   deploy.sh list-dumps                                 List user-mode crash dumps
#   deploy.sh start-app [pfn]                            Launch xllama through WDP
#   deploy.sh stop-app [pfn]                             Stop xllama through WDP
#   deploy.sh diagnose-startup [pfn]                     Start app and print startup diagnostics
#
# Required env vars: XBOX_IP, XBOX_USER, XBOX_PASS

set -euo pipefail

: "${XBOX_IP:?XBOX_IP not set}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

BASE_URL="https://${XBOX_IP}:11443"
CURL_AUTH=(--basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS)
APP_ID="GianlucaMazza.xllama"
# Pre-rebrand package identity (≤ v1.4.x installs). Kept so pfn/uninstall/log
# sub-commands still find an old package during the migration window.
APP_ID_LEGACY="VenereLabs.xllama"

# Xbox WDP requires X-CSRF-Token on all POST/DELETE requests; extract it from
# the Set-Cookie header of a plain GET.
#
# This used to be one `CSRF_TOKEN=$(curl ... 2>/dev/null | sed | head)`. Under
# `set -euo pipefail` an unreachable console made curl fail inside that
# substitution, so the script exited on THIS line with curl's bare rc (7) and
# printed nothing — the warnings below could not run, because there was no
# later line to run them on. Every caller then reported a silent failure whose
# only clue was an exit code, which is precisely the failure mode the truthful
# stop-app/start-app return values exist to remove.
#
# So separate the two questions. "Cannot reach the device" is fatal and gets a
# message; "reached it but no token" stays a warning, because read-only
# subcommands (pfn, get-log, list-*) genuinely work without one.
if ! CSRF_HEADERS="$(curl "${CURL_AUTH[@]}" "${BASE_URL}/" -o /dev/null -D - 2>&1)"; then
	echo "Error: cannot reach Device Portal at ${BASE_URL}" >&2
	[[ -n "$CSRF_HEADERS" ]] && echo "  ${CSRF_HEADERS}" >&2
	echo "  Check XBOX_IP, that the console is powered on, and that Dev Mode is active." >&2
	exit 1
fi
CSRF_TOKEN=$(printf '%s' "$CSRF_HEADERS" |
	sed -n 's/.*[Cc][Ss][Rr][Ff]-[Tt]oken=\([^;[:space:]]*\).*/\1/p' |
	tr -d '\r' | head -n 1)
if [[ -z "$CSRF_TOKEN" ]]; then
	echo "Warning: reached ${BASE_URL} but found no CSRF token —" \
		"POST/DELETE will fail with 403" >&2
fi

get_pfn() {
	curl "${CURL_AUTH[@]}" "${BASE_URL}/api/app/packagemanager/packages" |
		APP_ID="$APP_ID" APP_ID_LEGACY="$APP_ID_LEGACY" python3 -c '
import json
import os
import re
import sys

# An MSIX upgrade can leave two versions of the same family registered, and the
# WDP listing order is not defined. Taking the first match therefore silently
# targets an arbitrary version — a bench run would upload to one package and read
# results back from whichever the API happened to list first. Pick the highest
# version, and say so on stderr when there is more than one.
app_ids = (os.environ["APP_ID"], os.environ["APP_ID_LEGACY"])
data = json.load(sys.stdin)
matches = [p for p in data.get("InstalledPackages", [])
           if any(a in p.get("PackageRelativeId", "") for a in app_ids)]
if not matches:
    sys.exit(0)


def version_key(package):
    name = package.get("PackageFullName", "")
    m = re.search(r"_(\d+(?:\.\d+)*)_", name)
    return tuple(int(x) for x in m.group(1).split(".")) if m else ()


matches.sort(key=version_key, reverse=True)
if len(matches) > 1:
    others = ", ".join(p.get("PackageFullName", "") for p in matches[1:])
    sys.stderr.write(
        "Warning: %d xllama packages registered; using the highest version "
        "%s (also present: %s)\n"
        % (len(matches), matches[0].get("PackageFullName", ""), others))
print(matches[0].get("PackageFullName", ""))
'
}

require_pfn() {
	local pfn="${1:-}"
	if [[ -z "$pfn" ]]; then
		pfn="$(get_pfn)"
	fi
	if [[ -z "$pfn" ]]; then
		echo "Error: xllama package not found on Xbox." >&2
		exit 1
	fi
	printf '%s\n' "$pfn"
}

aumid_for_pfn() {
	local pfn="$1"
	local pfamily
	# shellcheck disable=SC2001  # regex uses [^_] class, not expressible with bash ${//}
	pfamily=$(echo "$pfn" | sed 's/_[0-9][0-9.]*_[^_]*__/_/')
	printf '%s!xllama' "$pfamily" | base64 -w0
}

print_log() {
	local pfn
	pfn="$(require_pfn "${1:-}")"
	curl "${CURL_AUTH[@]}" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${pfn}&path=\\LocalState&filename=xllama.log" ||
		true
}

list_localstate() {
	local pfn
	pfn="$(require_pfn "${1:-}")"
	curl "${CURL_AUTH[@]}" \
		"${BASE_URL}/api/filesystem/apps/files?knownfolderid=LocalAppData&packagefullname=${pfn}&path=\\LocalState" ||
		true
}

# Download a LocalState file to a local path (fails loud on HTTP errors so a
# missing remote file doesn't leave a stale/HTML local copy behind).
# subdir: optional path under LocalState, backslash-separated (e.g. "models\\x").
fetch_file() {
	local pfn="$1" name="$2" out="$3" subdir="${4:-}"
	local path='\LocalState'
	if [[ -n "$subdir" ]]; then
		path="\\LocalState\\${subdir}"
	fi
	curl "${CURL_AUTH[@]}" --fail -o "$out" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${pfn}&path=${path}&filename=${name}"
	echo "Fetched ${name} -> ${out}"
}

# Delete a file from LocalState (WDP DELETE; requires the CSRF token).
delete_file() {
	local pfn="$1" name="$2" subdir="${3:-}"
	local path='\LocalState'
	if [[ -n "$subdir" ]]; then
		path="\\LocalState\\${subdir}"
	fi
	curl "${CURL_AUTH[@]}" --fail \
		-H "X-CSRF-Token:${CSRF_TOKEN}" \
		-X DELETE \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${pfn}&path=${path}&filename=${name}" >/dev/null
	echo "Deleted ${name}."
}

# Create a directory inside LocalState.
# relpath: e.g. "models" or "models\\Phi-3.5-mini-instruct-onnx-gpu"
# WDP requires path=<parent> + newfoldername=<leaf> as separate query params.
mkdir_localstate() {
	local pfn="$1"
	local relpath="$2"

	# Create every path component from root downward so parent dirs always exist.
	# relpath uses backslash as separator: e.g. "models\\Phi-3.5-mini-instruct-onnx-gpu"
	# tr '\134' splits on backslash (octal 134) without triggering SC2141/SC1003.
	local parent_param="%5CLocalState"
	local accumulated=""
	local part
	# `|| [[ -n "$part" ]]`: the here-string from printf '%s' has no trailing
	# newline, so a bare `while read` DROPS the last path component — the model
	# dir itself was never created and every subsequent file POST failed with
	# "cannot find the path" (root cause of the silent 1.7B upload loss).
	while IFS= read -r part || [[ -n "$part" ]]; do
		[[ -z "$part" ]] && continue
		echo "Creating remote dir LocalState\\${accumulated:+${accumulated}\\}${part} ..."
		RESP=$(curl "${CURL_AUTH[@]}" \
			-H "X-CSRF-Token:${CSRF_TOKEN}" \
			-X POST \
			-d "" \
			"${BASE_URL}/api/filesystem/apps/folder?knownfolderid=LocalAppData&packagefullname=${pfn}&path=${parent_param}&newfoldername=${part}" 2>/dev/null || echo "")
		if [[ -n "$RESP" ]]; then
			echo "  mkdir response: $RESP"
		fi
		parent_param="${parent_param}%5C${part}"
		accumulated="${accumulated:+${accumulated}\\}${part}"
	done < <(printf '%s' "$relpath" | tr '\134' '\n')
}

list_dumps() {
	curl "${CURL_AUTH[@]}" "${BASE_URL}/api/debug/dump/usermode/dumps" || true
}

print_process_status() {
	curl "${CURL_AUTH[@]}" "${BASE_URL}/api/resourcemanager/processes" |
		python3 -c '
import json
import sys

data = json.load(sys.stdin)
matches = [
    p for p in data.get("Processes", [])
    if "xllama" in p.get("ImageName", "").lower()
    or "GianlucaMazza.xllama" in p.get("PackageFullName", "")
    or "VenereLabs.xllama" in p.get("PackageFullName", "")
]
if not matches:
    print("xllama process: not running")
else:
    for p in matches:
        print(
            "xllama process: pid={pid} image={image} running={running} ws={ws}".format(
                pid=p.get("ProcessId", ""),
                image=p.get("ImageName", ""),
                running=p.get("IsRunning", ""),
                ws=p.get("WorkingSetSize", ""),
            )
        )
'
}

start_app() {
	local pfn
	pfn="$(require_pfn "${1:-}")"
	local aumid
	aumid="$(aumid_for_pfn "$pfn")"
	# curl exits 0 on an HTTP error unless -f is given, and this discarded the
	# body, so "Started ${pfn}." used to print whatever the device answered —
	# including 400 {"ErrorMessage":"Failed to launch the application."} for a
	# package that is not installed. Same defect stop_app had; report the truth.
	local body code
	body="$(curl "${CURL_AUTH[@]}" \
		-H "X-CSRF-Token:${CSRF_TOKEN}" \
		-X POST \
		-d "" -w '\n%{http_code}' \
		"${BASE_URL}/api/taskmanager/app?appid=${aumid}")"
	code="${body##*$'\n'}"
	body="${body%$'\n'*}"
	if [[ "$code" == "200" ]]; then
		echo "Started ${pfn}."
	else
		echo "Error: start-app returned HTTP ${code}: ${body}" >&2
		return 1
	fi
}

stop_app() {
	local pfn
	pfn="$(require_pfn "${1:-}")"
	# WDP wants `package` BASE64-ENCODED. It was passed raw here, so the device
	# answered 400 "Failed to decode expected base64 encoded parameter: package"
	# and the app kept running — invisibly, because the response went to
	# /dev/null behind `|| true` and "Stopped" was printed unconditionally. Every
	# caller that stops before uploading was uploading into a live app.
	# start_app has always encoded correctly (aumid_for_pfn), which is why that
	# half worked.
	#
	# Measured on Series S, 2026-08-10:
	#   running     -> 200, empty body, process gone
	#   not running -> 400 {"ErrorMessage":"Failed to terminate the application."}
	#                  (an UNINSTALLED package answers identically — the device
	#                   cannot tell "stopped" from "not there", so neither can we)
	#   raw pfn     -> 400 {"ErrorMessage":"Failed to decode expected base64 ..."}
	# Not-running is success for a caller whose intent is "make sure it is not
	# running"; a decode error is not, and must not be swallowed again.
	local pkg64 body code
	pkg64="$(printf '%s' "$pfn" | base64 -w0)"
	body="$(curl "${CURL_AUTH[@]}" \
		-H "X-CSRF-Token:${CSRF_TOKEN}" \
		-X DELETE -w '\n%{http_code}' \
		"${BASE_URL}/api/taskmanager/app?package=${pkg64}")"
	code="${body##*$'\n'}"
	body="${body%$'\n'*}"
	case "$code" in
	200)
		echo "Stopped ${pfn}."
		;;
	400)
		if [[ "$body" == *"Failed to terminate"* ]]; then
			echo "${pfn} is not running."
		else
			echo "Error: stop-app rejected by the device: ${body}" >&2
			return 1
		fi
		;;
	*)
		echo "Error: stop-app returned HTTP ${code}: ${body}" >&2
		return 1
		;;
	esac
}

# -----------------------------------------------------------------------
# Sub-command: install-cert
#   install-cert <cert.cer>
#   Installs a trust certificate so signed-with-that-cert packages deploy.
# -----------------------------------------------------------------------
if [[ "${1:-}" == "pfn" ]]; then
	require_pfn "${2:-}"
	exit 0
fi

if [[ "${1:-}" == "get-log" ]]; then
	print_log "${2:-}"
	exit 0
fi

if [[ "${1:-}" == "list-localstate" ]]; then
	list_localstate "${2:-}"
	exit 0
fi

if [[ "${1:-}" == "fetch-file" ]]; then
	PFN="$(require_pfn "${2:-}")"
	NAME="${3:-}"
	OUT="${4:-}"
	SUBDIR="${5:-}"
	if [[ -z "$NAME" || -z "$OUT" ]]; then
		echo "Usage: $0 fetch-file <pfn> <name> <local-out> [subdir]" >&2
		exit 1
	fi
	fetch_file "$PFN" "$NAME" "$OUT" "$SUBDIR"
	exit 0
fi

if [[ "${1:-}" == "delete-file" ]]; then
	PFN="$(require_pfn "${2:-}")"
	NAME="${3:-}"
	SUBDIR="${4:-}"
	if [[ -z "$NAME" ]]; then
		echo "Usage: $0 delete-file <pfn> <name> [subdir]" >&2
		exit 1
	fi
	delete_file "$PFN" "$NAME" "$SUBDIR"
	exit 0
fi

if [[ "${1:-}" == "list-dumps" ]]; then
	list_dumps
	exit 0
fi

if [[ "${1:-}" == "start-app" ]]; then
	start_app "${2:-}"
	exit 0
fi

if [[ "${1:-}" == "stop-app" ]]; then
	stop_app "${2:-}"
	exit 0
fi

if [[ "${1:-}" == "diagnose-startup" ]]; then
	PFN="$(require_pfn "${2:-}")"
	echo "PFN: ${PFN}"
	echo "--- starting app ---"
	start_app "$PFN"
	sleep 5
	echo "--- process ---"
	print_process_status
	echo "--- xllama.log ---"
	print_log "$PFN"
	echo ""
	echo "--- LocalState ---"
	list_localstate "$PFN"
	echo ""
	echo "--- crash dumps ---"
	list_dumps
	echo ""
	exit 0
fi

if [[ "${1:-}" == "mkdir-localstate" ]]; then
	PFN="${2:-}"
	RELPATH="${3:-}"
	if [[ -z "$PFN" || -z "$RELPATH" ]]; then
		echo "Usage: $0 mkdir-localstate <package-full-name> <relpath>" >&2
		exit 1
	fi
	mkdir_localstate "$PFN" "$RELPATH"
	exit 0
fi

if [[ "${1:-}" == "install-cert" ]]; then
	CER="${2:-}"
	if [[ -z "$CER" || ! -f "$CER" ]]; then
		echo "Usage: $0 install-cert <path/to/cert.cer>" >&2
		exit 1
	fi
	echo "Installing certificate $(basename "$CER") on Xbox at ${XBOX_IP} ..."
	RESP=$(curl "${CURL_AUTH[@]}" \
		-H "X-CSRF-Token:${CSRF_TOKEN}" \
		-X POST \
		-F "file=@${CER};type=application/octet-stream" \
		"${BASE_URL}/api/app/packagemanager/certificate?package=$(basename "$CER")")
	echo "Response: $RESP"
	echo "Certificate installed."
	exit 0
fi

# -----------------------------------------------------------------------
# Sub-command: upload-file
#   upload-file <local-path> <package-full-name> [remote-dir] [remote-name]
#   remote-dir: e.g. "models\\Phi-3.5-mini-instruct-onnx-directml" → LocalFolder\models\...\
#               ""  → LocalFolder\
#   Auto-creates the remote-dir hierarchy before uploading.
# -----------------------------------------------------------------------
if [[ "${1:-}" == "upload-file" ]]; then
	LOCAL_PATH="${2:-}"
	PFN="${3:-}"
	REMOTE_DIR="${4:-}"
	REMOTE_NAME="${5:-}"

	if [[ -z "$LOCAL_PATH" || -z "$PFN" ]]; then
		echo "Usage: $0 upload-file <local-path> <package-full-name> [remote-dir] [remote-name]" >&2
		exit 1
	fi
	if [[ ! -f "$LOCAL_PATH" ]]; then
		echo "Error: file not found: $LOCAL_PATH" >&2
		exit 1
	fi

	# Auto-create the target subdirectory (WDP fails with 500 if it doesn't exist)
	if [[ -n "$REMOTE_DIR" ]]; then
		mkdir_localstate "$PFN" "$REMOTE_DIR"
	fi

	# Build path parameter (URL-encoded backslashes required by WDP).
	if [[ -n "$REMOTE_DIR" ]]; then
		PATH_PARAM="%5CLocalState%5C${REMOTE_DIR//\\/%5C}"
	else
		PATH_PARAM="%5CLocalState"
	fi

	# WDP names the remote file after the multipart filename; default is the
	# local basename, override with [remote-name] (e.g. upload x-v2.onnx as
	# model.onnx without a local rename).
	REMOTE_NAME="${REMOTE_NAME:-$(basename "$LOCAL_PATH")}"
	printf 'Uploading %s → LocalState\\%s\\%s ...\n' "$(basename "$LOCAL_PATH")" "$REMOTE_DIR" "$REMOTE_NAME"
	RESP=$(curl "${CURL_AUTH[@]}" \
		-H "X-CSRF-Token:${CSRF_TOKEN}" \
		-X POST \
		-F "file=@${LOCAL_PATH};type=application/octet-stream;filename=${REMOTE_NAME}" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=${PATH_PARAM}" 2>/dev/null || echo "")
	if [[ -n "$RESP" ]]; then
		echo "$RESP"
		# WDP returns JSON with "Success":false on failure
		if echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('Success',True) else 1)" 2>/dev/null; then
			: # success or non-JSON response (old WDP versions return empty on success)
		else
			echo "  ERROR: WDP upload failed" >&2
			exit 1
		fi
	fi
	echo "Done."
	exit 0
fi

# -----------------------------------------------------------------------
# Sub-command: upload-dir
#   upload-dir <local-dir> <package-full-name> <remote-dir>
#   Uploads every file inside local-dir to LocalFolder\remote-dir\.
#   Creates the remote dir first.
#   Usage: deploy.sh upload-dir ./Phi-3.5-onnx-directml/ $PFN "models\\Phi-3.5-mini-instruct-onnx-directml"
# -----------------------------------------------------------------------
if [[ "${1:-}" == "upload-dir" ]]; then
	LOCAL_DIR="${2:-}"
	PFN="${3:-}"
	REMOTE_DIR="${4:-}"

	if [[ -z "$LOCAL_DIR" || -z "$PFN" || -z "$REMOTE_DIR" ]]; then
		echo "Usage: $0 upload-dir <local-dir> <package-full-name> <remote-dir>" >&2
		exit 1
	fi
	if [[ ! -d "$LOCAL_DIR" ]]; then
		echo "Error: directory not found: $LOCAL_DIR" >&2
		exit 1
	fi

	mkdir_localstate "$PFN" "$REMOTE_DIR"

	# Confirm the target dir actually exists before uploading — WDP folder creation
	# can fail silently, after which every file POST returns Success:false with
	# "The system cannot find the path specified" (a batch that then reports
	# "Uploaded N" while landing nothing). Fail loudly here instead.
	PARENT_PARAM="%5CLocalState"
	LEAF="${REMOTE_DIR##*\\}"
	if [[ "$REMOTE_DIR" == *\\* ]]; then
		PARENT_PARAM="%5CLocalState%5C${REMOTE_DIR%\\*}"
		PARENT_PARAM="${PARENT_PARAM//\\/%5C}"
	fi
	DIR_CHECK=$(curl "${CURL_AUTH[@]}" \
		"${BASE_URL}/api/filesystem/apps/files?knownfolderid=LocalAppData&packagefullname=${PFN}&path=${PARENT_PARAM}" 2>/dev/null || echo "")
	if ! echo "$DIR_CHECK" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if any(i.get('Name')=='${LEAF}' for i in d.get('Items',[])) else 1)" 2>/dev/null; then
		echo "Error: remote dir LocalState\\${REMOTE_DIR} was not created (WDP mkdir failed); aborting upload." >&2
		exit 1
	fi

	PATH_PARAM="%5CLocalState%5C${REMOTE_DIR//\\/%5C}"
	total=0
	failed=0
	while IFS= read -r -d '' f; do
		fname=$(basename "$f")
		echo "  uploading $fname ..."
		RESP=$(curl "${CURL_AUTH[@]}" \
			-H "X-CSRF-Token:${CSRF_TOKEN}" \
			-X POST \
			-F "file=@${f};type=application/octet-stream" \
			"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=${PATH_PARAM}" 2>/dev/null || echo "")
		# WDP returns 200 with {"Success": false} on failure; empty body = success.
		if [[ -n "$RESP" ]] && ! echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('Success',True) else 1)" 2>/dev/null; then
			echo "    ERROR: upload of $fname failed: $RESP" >&2
			((failed++)) || true
		else
			((total++)) || true
		fi
	done < <(find "$LOCAL_DIR" -maxdepth 1 -type f -print0)
	echo "Uploaded $total file(s) to LocalState\\${REMOTE_DIR}.${failed:+ ($failed FAILED)}"
	[[ "$failed" -eq 0 ]] || exit 1
	exit 0
fi

# -----------------------------------------------------------------------
# Default: deploy an .msix/.appx
# -----------------------------------------------------------------------
APPX="${1:-}"
if [[ -z "$APPX" ]]; then
	echo "Usage:" >&2
	echo "  $0 <path/to/xllama.msix>                              (deploy package)" >&2
	echo "  $0 install-cert <path/to/cert.cer>                    (trust certificate)" >&2
	echo "  $0 upload-file <local> <pfn> [remote-dir] [remote-name] (upload file; auto-creates subdir)" >&2
	echo "  $0 upload-dir <local-dir> <pfn> <remote-dir>          (upload all files in dir)" >&2
	echo "  $0 mkdir-localstate <pfn> <relpath>                   (create dir in LocalState)" >&2
	echo "  $0 pfn                                                (print installed package full name)" >&2
	echo "  $0 get-log [pfn]                                      (print LocalState/xllama.log)" >&2
	echo "  $0 list-localstate [pfn]                              (list LocalState files)" >&2
	echo "  $0 list-dumps                                         (list user-mode crash dumps)" >&2
	echo "  $0 start-app [pfn]                                    (launch xllama)" >&2
	echo "  $0 stop-app [pfn]                                     (stop xllama)" >&2
	echo "  $0 diagnose-startup [pfn]                             (run startup diagnostics)" >&2
	exit 1
fi
if [[ ! -f "$APPX" ]]; then
	echo "Error: file not found: $APPX" >&2
	exit 1
fi

APPX_NAME=$(basename "$APPX")
APPX_DIR=$(dirname "$APPX")

# Auto-install companion .cer if present alongside the .msix
CER_PATH="${APPX_DIR}/../xllama-test.cer"
if [[ ! -f "$CER_PATH" ]]; then
	CER_PATH="${APPX_DIR}/../../xllama-test.cer"
fi
if [[ -f "$CER_PATH" ]]; then
	echo "Found companion certificate: $(readlink -f "$CER_PATH")"
	"$0" install-cert "$(readlink -f "$CER_PATH")" || true
	echo ""
fi

# Collect framework dependencies (.appx) from the sibling Dependencies/x64/ folder.
# WDP accepts multiple -F file= entries in the same POST for a bundle install.
DEPS=()
DEPS_DIR="${APPX_DIR}/Dependencies/x64"
if [[ -d "$DEPS_DIR" ]]; then
	while IFS= read -r -d '' dep; do
		DEPS+=("-F" "file=@${dep};type=application/octet-stream")
		echo "  + dependency: $(basename "$dep")"
	done < <(find "$DEPS_DIR" -name "*.appx" -print0)
fi

echo "Deploying ${APPX_NAME} to Xbox at ${XBOX_IP} ..."

# NOTE: Xbox Device Portal requires ?package=<filename> query parameter.
RESPONSE=$(curl "${CURL_AUTH[@]}" \
	-H "X-CSRF-Token:${CSRF_TOKEN}" \
	-X POST \
	-F "file=@${APPX};type=application/octet-stream" \
	"${DEPS[@]}" \
	"${BASE_URL}/api/app/packagemanager/package?package=${APPX_NAME}")

echo "Response: $RESPONSE"

if echo "$RESPONSE" | grep -qi '"Reason".*error\|failed'; then
	echo "Error: Device Portal returned an error." >&2
	exit 1
fi

echo "Upload complete. Monitor installation at: https://${XBOX_IP}:11443/#apps"

if command -v jq &>/dev/null; then
	echo "Polling installation status ..."
	for i in $(seq 1 24); do
		sleep 5
		STATUS=$(curl "${CURL_AUTH[@]}" "${BASE_URL}/api/app/packagemanager/state" 2>/dev/null || echo "{}")
		CODE=$(echo "$STATUS" | jq -r '.Code // -1' 2>/dev/null || echo "-1")
		REASON=$(echo "$STATUS" | jq -r '.Reason // "unknown"' 2>/dev/null || echo "unknown")
		SUCCESS=$(echo "$STATUS" | jq -r '.Success // false' 2>/dev/null || echo "false")
		echo "  [${i}] code=${CODE} reason='${REASON}' success=${SUCCESS}"
		if [[ "$SUCCESS" == "true" && "$CODE" == "0" ]]; then
			echo "Installation succeeded."
			break
		fi
		if [[ "$SUCCESS" == "false" ]]; then
			echo "Installation failed: ${REASON}" >&2
			exit 1
		fi
	done
else
	echo "Tip: install jq to poll installation status automatically."
fi
