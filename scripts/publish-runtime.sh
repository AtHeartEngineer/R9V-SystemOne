#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Explicit maintainer command: builds and publishes a candidate, never promotes it.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image=${1:?Usage: publish-runtime.sh REGISTRY/IMAGE:VERSION}
[[ $image == */*:* && $image != *:latest && $image != -* ]] || {
    printf 'Supply a registry image with an immutable release-version tag, not latest.\n' >&2
    exit 2
}
[[ -z $(git -C "$repo_root" status --porcelain) ]] || {
    printf 'Commit and review the release inputs before publishing.\n' >&2
    exit 1
}
"$repo_root/scripts/ci-static.sh"
R9V_IMAGE="$image" "$repo_root/scripts/build-image.sh"
docker push "$image"
docker image inspect "$image" --format '{{json .RepoDigests}}'
printf 'Candidate published. Qualify this digest on a clean dual-R9700 host before setting runtime distribution.image.\n'
