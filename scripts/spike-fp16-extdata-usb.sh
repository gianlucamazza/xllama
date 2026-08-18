#!/usr/bin/env bash
# spike-fp16-extdata-usb.sh — Fase 0 of the ">2 GB fp16 on GPU" unblock plan.
#
# Hypothesis: the weakly_canonical crash (docs/uwp-constraints.md §8) is specific to
# the walk over Q:\Users\UserMgr0 (LocalState). A model on USB
# (<usb>\xllama\models\<name>\, resolved via resolve_model_path's USB fallback,
# src/bridge/path_utils.cpp:167-203) does NOT traverse UserMgr0 — so loading an
# fp16 model WITH external .onnx.data from USB may avoid the crash entirely, with
# zero code changes.
#
# This script drives the load end-to-end via the headless bench path (bench.flag),
# then classifies the outcome from bench-result.csv + xllama.log:
#   LOADED  → a valid bench row appears           ⇒ spike PASS (fix at zero cost)
#   CRASH   → log shows weakly_canonical/ACCESS_DENIED/UserMgr0 ⇒ spike FAIL → Fase 1
#   OTHER   → OgaCreateModel failed w/o the above (e.g. OOM/budget) ⇒ investigate
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/spike-fp16-extdata-usb.sh <model-name-on-usb> [--prompt FILE] [--routing 0|1|2] [--yes]
#
# Precondition: the model dir is already on the USB stick and the stick is plugged
# into the console. Build/stage it first with scripts/build-fp16-dml-model.sh.
#
# Required env: XBOX_IP, XBOX_USER, XBOX_PASS

set -euo pipefail

MODEL_NAME="${1:?usage: spike-fp16-extdata-usb.sh <model-name-on-usb> [--prompt FILE] [--yes]}"
shift || true
PROMPT_FILE=""
ASSUME_YES=false
ROUTING=0 # 0=CpuOnly (int4 test model), 1=GpuOnly, 2=Auto — see routing_policy.h
while [[ $# -gt 0 ]]; do
	case "$1" in
	--prompt)
		PROMPT_FILE="${2:?--prompt requires a file}"
		shift 2
		;;
	--routing)
		ROUTING="${2:?--routing requires 0|1|2}"
		shift 2
		;;
	--yes)
		ASSUME_YES=true
		shift
		;;
	*)
		echo "Unknown argument: $1" >&2
		exit 1
		;;
	esac
done

: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY="${SCRIPT_DIR}/deploy.sh"
BENCH="${SCRIPT_DIR}/bench-xbox-ort.sh"
PROMPT_SRC="${PROMPT_FILE:-${REPO_ROOT}/bench/prompts/long-1k.txt}"
OUT_CSV="${REPO_ROOT}/bench/results/phase6-fp16-extdata.csv"
TMPDIR_LOCAL="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

echo "=== Fase 0 spike: fp16 + external data from USB ==="
echo "  Model (USB): $MODEL_NAME"
echo "  Prompt:      $PROMPT_SRC"
echo "  Xbox:        $XBOX_IP"

PFN="$("${DEPLOY}" pfn 2>/dev/null || true)"
[[ -z "$PFN" ]] && {
	echo "error: xllama not installed on the console — deploy it first" >&2
	exit 1
}
echo "  PFN:         $PFN"

if [[ "$ASSUME_YES" != "true" ]]; then
	echo ""
	printf 'Confirm the model dir is on the USB stick at:  <usb>\\xllama\\models\\%s\\\n' "$MODEL_NAME"
	echo "(model.onnx, model.onnx.data, genai_config.json, tokenizer files) and the stick is plugged in."
	read -r -p "Proceed? [y/N] " ans
	[[ "$ans" =~ ^[Yy]$ ]] || {
		echo "Aborted."
		exit 0
	}
fi

# --- Bootstrap USB discovery -------------------------------------------------
# usb_model_root.txt is written ONLY by EnsureModelNamedAsync (MainPage.cpp:1506-
# 1537) during an INTERACTIVE launch with the model SELECTED — it probes
# KnownFolders.RemovableDevices for xllama\models\<name>\genai_config.json and
# caches the drive root. The headless bench only READS that file. So we select the
# model via settings.json, launch the XAML app once to trigger the probe, then run
# the headless bench (which resolves the USB model via usb_model_root.txt).
echo ""
echo "--- Bootstrapping USB discovery (select model + interactive launch) ---"
cat >"${TMPDIR_LOCAL}/settings.json" <<EOF
{
  "system_prompt": "You are a helpful assistant.",
  "model": "${MODEL_NAME}",
  "kv_reuse": true,
  "routing": ${ROUTING},
  "sampling": { "temperature": 0.8, "top_p": 0.9, "top_k": 40, "repetition_penalty": 1.1, "n_predict": 64 }
}
EOF
# Clear the log so the verdict grep only sees this attempt.
curl --basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS -X DELETE \
	"https://${XBOX_IP}:11443/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=xllama.log" \
	>/dev/null 2>&1 || true
"${DEPLOY}" upload-file "${TMPDIR_LOCAL}/settings.json" "$PFN" >/dev/null 2>&1 || true
"${DEPLOY}" start-app "$PFN" >/dev/null || true
echo "  Waiting 20s for RemovableDevices enumeration + provisioning ..."
sleep 20
"${DEPLOY}" stop-app "$PFN" >/dev/null || true

if "${DEPLOY}" list-localstate "$PFN" 2>/dev/null | grep -qi "usb_model_root.txt"; then
	echo "  ✓ usb_model_root.txt present — USB drive discovered."
else
	echo "  ! usb_model_root.txt NOT written — the app did not find the model on USB."
	printf '    Check: stick plugged into the console? layout <usb>\\xllama\\models\\%s\\genai_config.json?\n' "$MODEL_NAME"
	echo "    Inspect the USB probe log lines:"
	"${DEPLOY}" get-log "$PFN" 2>/dev/null | grep -i "usb probe\|usb model" | tail -5 | sed 's/^/       /' || true
fi

# --- Load test via the headless bench path -----------------------------------
echo ""
echo "--- Load test + prefill bench (headless, DML) ---"
set +e
"${BENCH}" "$MODEL_NAME" --prompt "$PROMPT_SRC" --runs 2 --out "$OUT_CSV" --gpu-sample
BENCH_RC=$?
set -e

# --- Fetch the log tail for classification -----------------------------------
"${DEPLOY}" get-log "$PFN" >"${TMPDIR_LOCAL}/xllama.log" 2>/dev/null || true
LOG="${TMPDIR_LOCAL}/xllama.log"

echo ""
echo "=== VERDICT ==="
NEW_ROW=""
[[ -f "$OUT_CSV" ]] && NEW_ROW="$(tail -n1 "$OUT_CSV" 2>/dev/null || true)"

if [[ "$BENCH_RC" -eq 0 && -n "$NEW_ROW" && "$NEW_ROW" != *"model,quant,backend"* ]]; then
	echo "  ✅ LOADED — fp16 + external data loaded from USB without the weakly_canonical crash."
	echo "     Bench row: $NEW_ROW"
	echo ""
	echo "  ⇒ Spike PASS. >2 GB fp16 on GPU is unblocked at ZERO code cost via USB provisioning."
	echo "     Next: route large fp16 models to USB in uwp/models/manifest.json + docs/model-selection.md,"
	echo "     record the prefill crossover row in docs/benchmarks.md, close ROADMAP.md:163-167."
	echo "     Fase 1 (ORT patch) is NOT needed."
	exit 0
fi

if grep -qiE "weakly_canonical|ACCESS_DENIED|UserMgr0|escapes model directory" "$LOG" 2>/dev/null; then
	echo "  ❌ CRASH (weakly_canonical) — the path walk fails even from USB."
	echo "     Log evidence:"
	grep -iE "weakly_canonical|ACCESS_DENIED|UserMgr0|OgaCreateModel|escapes model directory" "$LOG" | tail -5 | sed 's/^/       /'
	echo ""
	echo "  ⇒ Spike FAIL. Proceed to Fase 1 (patch ORT ValidateExternalDataPath):"
	echo "       patches/onnxruntime-extdata-appcontainer.patch + scripts/vendor-ort-extdata-patch.ps1 -Build"
	exit 2
fi

if grep -qiE "OgaCreateModel failed|out of memory|E_OUTOFMEMORY|887A" "$LOG" 2>/dev/null; then
	echo "  ⚠ OTHER load failure (no weakly_canonical signal) — likely budget/OOM or a DML init issue."
	echo "     Log evidence:"
	grep -iE "OgaCreateModel|memory|887A|budget" "$LOG" | tail -5 | sed 's/^/       /'
	echo ""
	echo "  ⇒ The weakly_canonical hypothesis may be VALID (it got past parse) but the model is too big."
	echo "     Retry with a smaller fp16 model (build-fp16-dml-model.sh --smoke) to isolate the load path,"
	echo "     or check 'deploy.sh list-dumps' for a minidump."
	exit 3
fi

echo "  ❓ INCONCLUSIVE — no bench row and no clear log signal (bench rc=${BENCH_RC})."
echo "     Inspect: ${LOG}  and  ./scripts/deploy.sh list-dumps"
exit 4
