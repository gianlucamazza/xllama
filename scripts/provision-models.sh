#!/usr/bin/env bash
# provision-models.sh — push catalogue models onto the Xbox LocalState over WDP.
#
# The app normally downloads models on demand from inside the UI. In Dev Mode
# (no text input) the headless validators (validate-console.sh routing/taesd,
# validate-logit-parity.sh) need specific models pre-seeded. This script fetches
# a model's files from the manifest's hf_base_url and uploads them to
# LocalState\models\<name>\..., reusing deploy.sh's WDP upload-file (which auto-
# creates the subdir hierarchy and verifies each POST).
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/provision-models.sh <model-name> [<model-name> ...]
#   ./scripts/provision-models.sh --list          # print catalogue names
#   ./scripts/provision-models.sh --all-test       # the full live-test set
#
# Idempotent: a model whose files already exist on the device is skipped
# (pass --force to re-upload). Downloads are cached under a temp dir for the run.
#
# Requires: gh not needed; curl, python3, and the Device Portal env
# (XBOX_IP/USER/PASS). Files come from the bundled manifest uwp/models/manifest.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY="${SCRIPT_DIR}/deploy.sh"
MANIFEST="${REPO_ROOT}/uwp/models/manifest.json"

: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

[[ -f "$MANIFEST" ]] || {
	echo "manifest not found: $MANIFEST" >&2
	exit 2
}

# Models the full-functionality live test needs pre-seeded (chat GGUF, DML
# routing / logit-parity reference, and diffusion).
ALL_TEST=(lfm25-350m smollm2-360m-dml-fp16 sd-turbo-fp16)

FORCE=false
MODELS=()
for arg in "$@"; do
	case "$arg" in
	--list)
		python3 -c 'import json,sys
m=json.load(open(sys.argv[1]))
for e in m["models"]:
    if "name" not in e: continue
    src = "auto" if e.get("hf_base_url") else "USB-only"
    print("  %-24s %-10s %-8s %s" % (e["name"], e.get("kind","ort-genai"), src, e.get("display","")))' "$MANIFEST"
		exit 0
		;;
	--all-test) MODELS+=("${ALL_TEST[@]}") ;;
	--force) FORCE=true ;;
	-*)
		echo "unknown flag: $arg" >&2
		exit 2
		;;
	*) MODELS+=("$arg") ;;
	esac
done

[[ ${#MODELS[@]} -eq 0 ]] && {
	echo "no models given (see --list / --all-test)" >&2
	exit 2
}

PFN=$("${DEPLOY}" pfn 2>/dev/null)
[[ -z "$PFN" ]] && {
	echo "xllama not installed — deploy it first" >&2
	exit 2
}
echo "==> device PFN: $PFN"

CURL_AUTH=(--basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS -m 20)
BASE_URL="https://${XBOX_IP}:11443"

# Disk-backed work dir: model files are up to ~1.7 GB, so the default /tmp
# (often tmpfs / RAM-backed) fills up mid-download (curl error 23). Override with
# PROVISION_WORK_DIR; default to a cache under $HOME (real disk).
WORK_ROOT="${PROVISION_WORK_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/xllama-provision}"
mkdir -p "$WORK_ROOT"
WORK_DIR="$(mktemp -d "${WORK_ROOT}/run.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

# Emit "filename<TAB>remote" lines for a model, plus hf_base_url on the first line
# prefixed with "BASE\t". Exits 3 if the model is USB-only / unknown.
model_files() {
	python3 -c '
import json, sys
name = sys.argv[2]
m = json.load(open(sys.argv[1]))
e = next((x for x in m["models"] if x.get("name") == name), None)
if e is None:
    print("ERR: unknown model", file=sys.stderr); sys.exit(3)
base = e.get("hf_base_url")
if not base:
    print("ERR: USB-only (no hf_base_url)", file=sys.stderr); sys.exit(3)
print("BASE\t" + base)
for f in e["files"]:
    fn = f["filename"]
    print(fn + "\t" + f.get("remote", fn))
' "$MANIFEST" "$1"
}

# Is a file already present on the device (non-zero size)? subdir uses backslashes.
remote_has() {
	local subdir="$1" fname="$2"
	local path="%5CLocalState%5C${subdir//\\/%5C}"
	curl "${CURL_AUTH[@]}" \
		"${BASE_URL}/api/filesystem/apps/files?knownfolderid=LocalAppData&packagefullname=${PFN}&path=${path}" 2>/dev/null |
		python3 -c "import sys,json
try: d=json.load(sys.stdin)
except Exception: sys.exit(1)
for i in d.get('Items',[]):
    if i.get('Name')==sys.argv[1] and int(i.get('FileSize') or i.get('Size') or 0)>0: sys.exit(0)
sys.exit(1)" "$fname"
}

provision_one() {
	local model="$1"
	echo ""
	echo "=== provisioning: $model ==="
	local lines base
	if ! lines=$(model_files "$model"); then
		echo "  SKIP: $model not auto-provisionable (see stderr above)"
		return 1
	fi
	base=$(printf '%s\n' "$lines" | sed -n 's/^BASE\t//p')

	local rc=0
	while IFS=$'\t' read -r filename remote; do
		[[ "$filename" == "BASE" ]] && continue
		[[ -z "$filename" ]] && continue

		# Split filename into subdir (backslash) + basename for the device path.
		local base_name subdir remote_dir
		base_name="${filename##*/}"
		if [[ "$filename" == */* ]]; then
			subdir="${filename%/*}"
			remote_dir="models\\${model}\\${subdir//\//\\}"
		else
			remote_dir="models\\${model}"
		fi

		# Skip if already present on the device (idempotent), unless --force.
		if [[ "$FORCE" == false ]] && remote_has "$remote_dir" "$base_name"; then
			echo "  [skip] $filename (already on device)"
			continue
		fi

		# Unique per-asset temp dir keeping the correct basename (deploy.sh
		# upload-file derives the device filename from it). Multiple files share
		# the basename "model.onnx" (text_encoder/unet/vae_decoder), so a flat
		# temp name would collide; a fresh download each time (no -C - resume)
		# avoids reusing a stale partial from a different asset.
		local url="${base}/${remote}"
		local tmp_dir="${WORK_DIR}/${remote//\//_}.d"
		mkdir -p "$tmp_dir"
		local tmp="${tmp_dir}/${base_name}"
		rm -f "$tmp"
		echo "  [get ] $remote"
		if ! curl -fL --retry 3 --retry-delay 2 -o "$tmp" "$url"; then
			echo "  ERROR: download failed: $url" >&2
			rm -f "$tmp"
			rc=1
			continue
		fi
		echo "  [put ] $filename -> LocalState\\${remote_dir}"
		if ! "${DEPLOY}" upload-file "$tmp" "$PFN" "$remote_dir" >/dev/null; then
			echo "  ERROR: upload failed: $filename" >&2
			rc=1
		fi
		rm -f "$tmp"
	done <<<"$lines"

	if [[ $rc -eq 0 ]]; then
		echo "  OK: $model provisioned"
	else
		echo "  PARTIAL/FAIL: $model (see errors above)"
	fi
	return $rc
}

overall=0
for m in "${MODELS[@]}"; do
	provision_one "$m" || overall=1
done

echo ""
[[ $overall -eq 0 ]] && echo "provision: ALL OK" || echo "provision: some models failed"
exit $overall
