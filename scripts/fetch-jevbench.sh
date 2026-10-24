#!/usr/bin/env bash
# fetch-jevbench.sh — download the 231 public JevBench items and record provenance.
#
# Default source (override with env, never a console host):
#   XLLAMA_JEVBENCH_REPO   https://github.com/fstandhartinger/jevbench
#   XLLAMA_JEVBENCH_COMMIT 609a664e81f46e03731600bd7bd9f1b7a61257c5
#   XLLAMA_JEVBENCH_DIR    <repo>/bench/jevbench/public
#
# Usage:
#   ./scripts/fetch-jevbench.sh
#   XLLAMA_JEVBENCH_DIR=/tmp/jevbench ./scripts/fetch-jevbench.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_URL="${XLLAMA_JEVBENCH_REPO:-https://github.com/fstandhartinger/jevbench}"
COMMIT="${XLLAMA_JEVBENCH_COMMIT:-609a664e81f46e03731600bd7bd9f1b7a61257c5}"
DEST="${XLLAMA_JEVBENCH_DIR:-${REPO_ROOT}/bench/jevbench/public}"

# SHA256 of the three public files at the default commit. A different commit
# must set XLLAMA_JEVBENCH_SKIP_HASH=1 after recording its own hashes.
EXPECTED_EASY="231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b"
EXPECTED_HARD="89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb"
EXPECTED_ORIG="5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180"

mkdir -p "${DEST}"
# github.com/foo/bar → raw.githubusercontent.com/foo/bar/COMMIT/path
if [[ "${REPO_URL}" == https://github.com/* ]]; then
	OWNER_REPO="${REPO_URL#https://github.com/}"
	OWNER_REPO="${OWNER_REPO%.git}"
	BASE="https://raw.githubusercontent.com/${OWNER_REPO}/${COMMIT}/datasets/public"
else
	echo "fetch-jevbench: set XLLAMA_JEVBENCH_BASE to a raw file prefix for non-GitHub repos" >&2
	BASE="${XLLAMA_JEVBENCH_BASE:?XLLAMA_JEVBENCH_BASE is required for non-GitHub sources}"
fi

sha256_of() {
	sha256sum "$1" | awk '{print $1}'
}

download() {
	local name="$1"
	local url="${BASE}/${name}"
	echo "GET ${url}"
	curl -fsSL -o "${DEST}/${name}" "${url}"
}

download easy.jsonl
download original.jsonl
download hard.jsonl

EASY_HASH="$(sha256_of "${DEST}/easy.jsonl")"
ORIG_HASH="$(sha256_of "${DEST}/original.jsonl")"
HARD_HASH="$(sha256_of "${DEST}/hard.jsonl")"
N_EASY="$(grep -cve '^[[:space:]]*$' "${DEST}/easy.jsonl" || true)"
N_ORIG="$(grep -cve '^[[:space:]]*$' "${DEST}/original.jsonl" || true)"
N_HARD="$(grep -cve '^[[:space:]]*$' "${DEST}/hard.jsonl" || true)"
N_TOTAL=$((N_EASY + N_ORIG + N_HARD))

if [[ "${XLLAMA_JEVBENCH_SKIP_HASH:-0}" != "1" ]]; then
	if [[ "${COMMIT}" == "609a664e81f46e03731600bd7bd9f1b7a61257c5" ]]; then
		[[ "${EASY_HASH}" == "${EXPECTED_EASY}" ]] || {
			echo "easy.jsonl SHA256 mismatch: ${EASY_HASH}" >&2
			exit 1
		}
		[[ "${ORIG_HASH}" == "${EXPECTED_ORIG}" ]] || {
			echo "original.jsonl SHA256 mismatch: ${ORIG_HASH}" >&2
			exit 1
		}
		[[ "${HARD_HASH}" == "${EXPECTED_HARD}" ]] || {
			echo "hard.jsonl SHA256 mismatch: ${HARD_HASH}" >&2
			exit 1
		}
	fi
fi

if [[ "${N_TOTAL}" -ne 231 ]]; then
	echo "expected 231 public items, got ${N_TOTAL} (${N_EASY}+${N_ORIG}+${N_HARD})" >&2
	exit 1
fi

cat >"${DEST}/provenance.json" <<EOF
{
  "source_url": "${REPO_URL}",
  "commit": "${COMMIT}",
  "files": {
    "easy.jsonl": {"sha256": "${EASY_HASH}", "n": ${N_EASY}},
    "original.jsonl": {"sha256": "${ORIG_HASH}", "n": ${N_ORIG}},
    "hard.jsonl": {"sha256": "${HARD_HASH}", "n": ${N_HARD}}
  },
  "n_public": ${N_TOTAL},
  "fetched_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "wrote ${DEST} (${N_TOTAL} items)"
echo "provenance: ${DEST}/provenance.json"
