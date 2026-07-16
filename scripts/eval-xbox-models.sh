#!/usr/bin/env bash
# eval-xbox-models.sh — deterministic H9 capability checks through the LAN API.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/eval-xbox-models.sh --models model-a,model-b [--tasks FILE] [--out FILE]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TASKS="${REPO_ROOT}/bench/eval/phase7-h9.json"
OUT="${REPO_ROOT}/bench/results/phase7-h9.jsonl"
MODELS=""

while [[ $# -gt 0 ]]; do
	case "$1" in
	--models)
		MODELS="${2:?--models requires a comma-separated list}"
		shift 2
		;;
	--tasks)
		TASKS="${2:?--tasks requires a file}"
		shift 2
		;;
	--out)
		OUT="${2:?--out requires a file}"
		shift 2
		;;
	*)
		echo "Unknown argument: $1" >&2
		exit 1
		;;
	esac
done

: "${MODELS:?--models is required}"
: "${XBOX_IP:?XBOX_IP not set — source ~/.config/xllama/xbox-env}"
command -v curl >/dev/null
command -v jq >/dev/null
jq -e 'type == "array" and length > 0' "$TASKS" >/dev/null

API_URL="http://${XBOX_IP}:11434/v1/chat/completions"
mkdir -p "$(dirname "$OUT")"
: >"$OUT"

score_response() {
	local task="$1" content="$2" pass=true pattern max_words expected_items words items
	while IFS= read -r pattern; do
		if ! grep -Eiq "$pattern" <<<"$content"; then
			pass=false
		fi
	done < <(jq -r '.required_patterns[]?' <<<"$task")
	while IFS= read -r pattern; do
		if grep -Eiq "$pattern" <<<"$content"; then
			pass=false
		fi
	done < <(jq -r '.forbidden_patterns[]?' <<<"$task")

	max_words=$(jq -r '.max_words // 0' <<<"$task")
	words=$(wc -w <<<"$content")
	if ((max_words > 0 && words > max_words)); then
		pass=false
	fi

	expected_items=$(jq -r '.expected_list_items // 0' <<<"$task")
	if ((expected_items > 0)); then
		items=$(grep -Ec '^[[:space:]]*([-*]|[0-9]+[.)])[[:space:]]+' <<<"$content" || true)
		if ((items != expected_items)); then
			pass=false
		fi
	fi

	printf '%s' "$pass"
}

IFS=',' read -r -a model_list <<<"$MODELS"
for model in "${model_list[@]}"; do
	model="${model#"${model%%[![:space:]]*}"}"
	model="${model%"${model##*[![:space:]]}"}"
	echo "=== H9: $model ==="
	while IFS= read -r task; do
		id=$(jq -r '.id' <<<"$task")
		request=$(jq -cn --arg model "$model" --argjson task "$task" '{
			model: $model,
			messages: $task.messages,
			max_completion_tokens: ($task.max_tokens // 128),
			temperature: 0,
			seed: 42
		}')
		start_ns=$(date +%s%N)
		if ! response=$(curl -sS --max-time 600 -H 'Content-Type: application/json' -d "$request" "$API_URL"); then
			response='{"error":{"message":"request failed"}}'
		fi
		end_ns=$(date +%s%N)
		latency_ms=$(((end_ns - start_ns) / 1000000))
		content=$(jq -r '.choices[0].message.content // ("ERROR: " + (.error.message // "invalid response"))' <<<"$response")
		pass=$(score_response "$task" "$content")
		jq -cn \
			--arg model "$model" --arg task_id "$id" --arg content "$content" \
			--argjson pass "$pass" --argjson latency_ms "$latency_ms" \
			--argjson usage "$(jq '.usage // {}' <<<"$response")" \
			'{model: $model, task_id: $task_id, pass: $pass, latency_ms: $latency_ms, usage: $usage, content: $content}' \
			| tee -a "$OUT"
	done < <(jq -c '.[]' "$TASKS")
done

echo ""
echo "=== Summary ==="
jq -s -r 'group_by(.model)[] | .[0].model as $m | (map(select(.pass)) | length) as $p | "\($m): \($p)/\(length) pass"' "$OUT"
echo "Results: $OUT"
