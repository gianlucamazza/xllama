#!/usr/bin/env bash
# check-uwp-host.sh - validate the Linux host prerequisites for a Windows UWP VM.

set -euo pipefail

ok=0

check_cmd() {
	local name="$1"
	if command -v "$name" >/dev/null 2>&1; then
		echo "ok: ${name} -> $(command -v "$name")"
	else
		echo "missing: ${name}" >&2
		ok=1
	fi
}

echo "=== xllama UWP host preflight ==="

check_cmd docker
check_cmd qemu-system-x86_64
check_cmd virt-install
check_cmd virsh

if docker info --format '{{.OSType}} {{.Architecture}} {{.ServerVersion}}' >/tmp/xllama-docker-info 2>/dev/null; then
	echo "docker: $(cat /tmp/xllama-docker-info)"
	if grep -q '^linux ' /tmp/xllama-docker-info; then
		echo "note: local Docker is Linux-only; Windows/UWP builds require a Windows VM or Windows CI."
	fi
else
	echo "warning: docker daemon is not reachable" >&2
fi
rm -f /tmp/xllama-docker-info

if systemctl is-active --quiet libvirtd 2>/dev/null || systemctl is-active --quiet virtqemud 2>/dev/null; then
	echo "ok: libvirt service is active"
else
	echo "warning: libvirt service is not active; start libvirtd or virtqemud before creating the VM" >&2
fi

for group_name in kvm libvirt; do
	if id -nG | tr ' ' '\n' | grep -qx "$group_name"; then
		echo "ok: current user is in ${group_name}"
	else
		echo "warning: current user is not in ${group_name}" >&2
	fi
done

if [[ -d /var/lib/libvirt/images ]]; then
	echo "ok: libvirt image directory exists: /var/lib/libvirt/images"
else
	echo "warning: /var/lib/libvirt/images does not exist" >&2
fi

exit "$ok"
