#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
base_image=${R9V_RETAINED_BASE_IMAGE:?Set R9V_RETAINED_BASE_IMAGE to the pinned retained base image; a public base is not configured yet}
runtime_image=${R9V_IMAGE:-r9v-qwen38-flash-next-mtp4:local}
expected=$(python3 - "$repo_root" <<'PYBASE'
import json,sys
from pathlib import Path
print(json.loads((Path(sys.argv[1])/'runtimes/qwen38-flash-next-gfx1201-v1/retained-mtp4/PROVENANCE.json').read_text())['base_image_digest'])
PYBASE
)
actual=$(docker image inspect "$base_image" --format '{{.Id}}')
[[ $actual == "$expected" ]] || {
    printf 'Retained base image differs: expected %s, found %s\n' "$expected" "$actual" >&2
    exit 2
}
docker buildx build --load --network=none --pull=false \
    --file "$repo_root/docker/Dockerfile.retained-mtp4" \
    --build-arg BASE_IMAGE="$base_image" \
    --build-arg R9V_BASE_IMAGE_ID="$actual" \
    --build-arg R9V_REVISION="$(git -C "$repo_root" rev-parse HEAD)" \
    --tag "$runtime_image" "$repo_root"
printf 'Built %s; exact image qualification is required before release.\n' "$runtime_image"
