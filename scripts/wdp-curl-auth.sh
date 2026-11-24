#!/usr/bin/env bash
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
# Device Portal credentials for curl, never via process argv.
#
# Usage, after XBOX_USER and XBOX_PASS are in the environment:
#   source wdp-curl-auth.sh
#   wdp_setup_curl_auth            # CURL_AUTH=(--basic -K "$WDP_CURL_CONFIG" -k -sS)
#   wdp_setup_curl_auth -m 20      # extra curl flags appended
# The curl -K file lives in a 0700 mktemp directory (file mode 0600) and is
# removed on EXIT/INT/TERM. Do not set -x around this helper: it turns xtrace
# off while the password is expanded.

wdp_curl_config_escape() {
	# Escape \ and " for a curl --config double-quoted value.
	local s=$1
	s=${s//\\/\\\\}
	s=${s//\"/\\\"}
	printf '%s' "$s"
}

wdp_curl_auth_cleanup() {
	set +x
	if [[ -n "${WDP_CURL_CONFIG_DIR:-}" ]]; then
		rm -rf "${WDP_CURL_CONFIG_DIR}"
	fi
	unset WDP_CURL_CONFIG_DIR WDP_CURL_CONFIG
}

_wdp_trap_body() {
	local sig="$1" spec
	spec=$(trap -p "$sig" 2>/dev/null || true)
	[[ -n "$spec" ]] || return 0
	spec="${spec#trap -- \'}"
	local suffix="\' ${sig}"
	spec="${spec%"$suffix"}"
	printf '%s' "$spec"
}

wdp_setup_curl_auth() {
	local xtrace_restore="" old_umask="" user_esc pass_esc
	case "$-" in
	*x*) xtrace_restore="set -x" ;;
	esac
	set +x

	wdp_curl_auth_cleanup

	: "${XBOX_USER:?XBOX_USER not set}"
	: "${XBOX_PASS:?XBOX_PASS not set}"
	if [[ "$XBOX_USER" == *$'\n'* || "$XBOX_USER" == *$'\r'* ]]; then
		echo "Error: XBOX_USER must not contain a newline" >&2
		eval "$xtrace_restore"
		return 1
	fi
	if [[ "$XBOX_PASS" == *$'\n'* || "$XBOX_PASS" == *$'\r'* ]]; then
		echo "Error: XBOX_PASS must not contain a newline" >&2
		eval "$xtrace_restore"
		return 1
	fi

	old_umask=$(umask)
	umask 077
	if ! WDP_CURL_CONFIG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/wdp-curl.XXXXXX"); then
		umask "$old_umask"
		echo "Error: cannot create Device Portal curl config directory" >&2
		eval "$xtrace_restore"
		return 1
	fi
	WDP_CURL_CONFIG="${WDP_CURL_CONFIG_DIR}/config"
	user_esc=$(wdp_curl_config_escape "$XBOX_USER")
	pass_esc=$(wdp_curl_config_escape "$XBOX_PASS")
	printf 'user = "%s:%s"\n' "$user_esc" "$pass_esc" >"$WDP_CURL_CONFIG"
	chmod 600 "$WDP_CURL_CONFIG"
	umask "$old_umask"

	# Callers expand this as curl "${CURL_AUTH[@]}".
	# shellcheck disable=SC2034
	CURL_AUTH=(--basic -K "${WDP_CURL_CONFIG}" -k -sS "$@")

	_WDP_PREV_EXIT=$(_wdp_trap_body EXIT)
	_WDP_PREV_INT=$(_wdp_trap_body INT)
	_WDP_PREV_TERM=$(_wdp_trap_body TERM)
	# shellcheck disable=SC2064
	trap 'wdp_curl_auth_cleanup; eval "${_WDP_PREV_EXIT:-}"' EXIT
	# shellcheck disable=SC2064
	trap 'wdp_curl_auth_cleanup; if [[ -n "${_WDP_PREV_INT:-}" ]]; then eval "${_WDP_PREV_INT}"; else exit 130; fi' INT
	# shellcheck disable=SC2064
	trap 'wdp_curl_auth_cleanup; if [[ -n "${_WDP_PREV_TERM:-}" ]]; then eval "${_WDP_PREV_TERM}"; else exit 143; fi' TERM

	eval "$xtrace_restore"
}
