#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the WMMA prefill overlay on the verified public image7 runtime.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
base_image=${R9V_WMMA_BASE_IMAGE:?Set R9V_WMMA_BASE_IMAGE to a local tag of the public image7 runtime}
runtime_image=${R9V_IMAGE:-r9v-qwen38-flash-next-wmma:local}
expected=sha256:46ab688af195643e61322a72b4e7b7fa0999c12299bffb2a4515f8363c59393c
actual=$(docker image inspect "$base_image" --format '{{.Id}}')
[[ $actual == "$expected" ]] || {
    printf 'Base image differs: expected %s, found %s\n' "$expected" "$actual" >&2
    exit 2
}
docker buildx build --load --network=none --pull=false \
    --file "$repo_root/docker/Dockerfile.wmma-prefill" \
    --build-arg BASE_IMAGE="$base_image" \
    --build-arg R9V_REVISION="$(git -C "$repo_root" rev-parse HEAD)" \
    --tag "$runtime_image" "$repo_root"
printf 'Built %s; exact image qualification is required before release.\n' "$runtime_image"
