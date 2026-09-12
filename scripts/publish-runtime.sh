#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Explicit maintainer command: publish the exact qualified image, never rebuild.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image=${1:?Usage: publish-runtime.sh REGISTRY/IMAGE:VERSION LOCAL_IMAGE QUALIFICATION.json}
qualified_image=${2:?Supply the exact locally qualified image}
receipt=${3:?Supply the qualification receipt}
[[ $image == */*:* && $image != *:latest && $image != -* ]] || {
    printf 'Supply a registry image with an immutable release-version tag, not latest.\n' >&2
    exit 2
}
[[ -z $(git -C "$repo_root" status --porcelain) ]] || {
    printf 'Commit and review the release inputs before publishing.\n' >&2
    exit 1
}
"$repo_root/scripts/ci-static.sh"
image_id=$("${PYTHON:-python3}" "$repo_root/tools/release_gate.py" --image "$qualified_image" --receipt "$receipt")
docker image tag "$image_id" "$image"
docker push "$image"
docker image inspect "$image" --format '{{json .RepoDigests}}'
printf 'Qualified image published without rebuilding. Record the returned digest in runtime distribution.image.\n'
