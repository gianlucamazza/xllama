#!/usr/bin/env bash
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
# Shared WDP file-write status contract. Caller supplies CURL_AUTH/CSRF_TOKEN.

wdp_upload_file_request() {
	local url="$1" local_path="$2" filename="${3:-}" response form_entry code
	if [[ ! -f "$local_path" ]]; then
		echo "Error: WDP upload source does not exist" >&2
		return 1
	fi
	form_entry="${local_path};type=application/octet-stream"
	[[ -z "$filename" ]] || form_entry="${local_path};filename=${filename};type=application/octet-stream"
	# Never turn a curl failure into the empty-body success used by older WDP.
	if ! response=$(curl "${CURL_AUTH[@]}" --fail-with-body --connect-timeout 20 --max-time 600 \
		-H "X-CSRF-Token:${CSRF_TOKEN}" -X POST -F "file=@${form_entry}" \
		-w $'\n%{http_code}' "$url"); then
		echo "Error: WDP upload request failed" >&2
		return 1
	fi
	code="${response##*$'\n'}"
	if [[ "$response" != *$'\n'* || "$code" != 200 ]]; then
		printf 'Error: WDP upload returned HTTP %s\n' "$code" >&2
		return 1
	fi
	response="${response%$'\n'*}"
	# Old portals return no payload on success. Nonempty replies must explicitly
	# confirm success: HTTP 200 with Success:false is a normal WDP failure mode.
	if [[ ! "$response" =~ [^[:space:]] ]]; then
		return 0
	fi
	if ! python3 -c 'import json,sys
try:
    result = json.load(sys.stdin)
    good = isinstance(result, dict) and result.get("Success") is True
except (ValueError, TypeError):
    good = False
sys.exit(0 if good else 1)' <<<"$response"; then
		printf 'Error: WDP upload rejected: %s\n' "$response" >&2
		return 1
	fi
	return 0
}
