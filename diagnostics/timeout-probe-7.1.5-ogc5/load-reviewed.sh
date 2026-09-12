#!/bin/bash
set -euo pipefail
[[ $(id -u) == 0 ]] || { echo 'Workstation root is required to insert this reviewed module.' >&2; exit 1; }
[[ $(uname -r) == 7.1.5-ogc5.1.fc44.x86_64 ]] || exit 1
echo '8269e5b4a6f181c771e1f9ae0e8604f1de254df032e06a32ab47ab833e606f3a  /sys/kernel/btf/vmlinux' | sha256sum -c -
echo 'd1e6c9c5376d7d10888aecd75c0aea5754e932441c642095fb397c03c5b95908  /sys/kernel/btf/amdgpu' | sha256sum -c -
probe=/var/home/dylan/projects/inference/r9v/diagnostics/timeout-probe-7.1.5-ogc5/r9v_timeout_probe.ko
echo '5c87287a290664e4db96de2e10da7b1f45cc3a13082e7457cb25bcf81db33331  '"$probe" | sha256sum -c -
exec /usr/sbin/insmod "$probe"
