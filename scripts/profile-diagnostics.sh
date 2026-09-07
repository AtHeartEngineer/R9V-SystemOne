#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
action=${1:?Expected support, soak, or placement}
shift
case $action in
    support) tool=support_bundle.py ;;
    soak) tool=soak_runtime.py ;;
    placement) tool=trim_experts.py ;;
    *) printf 'Unknown diagnostics action: %s\n' "$action" >&2; exit 2 ;;
esac
if [[ -n ${R9V_CONFIG_FILE:-} ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$R9V_CONFIG_FILE"
    set +a
fi
set -a
# shellcheck disable=SC1090
source "${R9V_PROFILE:-$repo_root/profiles/qwen38-flash-next/dual-r9700/profile.env}"
set +a
exec python3 "$repo_root/tools/$tool" "$@"
