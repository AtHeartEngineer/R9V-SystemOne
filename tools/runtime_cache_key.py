#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Namespace compile caches by the exact serving image, placement and workload."""

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

try:
    from tools.plan_experts import IGNORED
except ModuleNotFoundError:
    from plan_experts import IGNORED


def cache_key(image, manifest, config, driver):
    settings = {
        k: v
        for k, v in config.items()
        if k.startswith("R9V_")
        and k not in IGNORED
        and k not in {"R9V_CACHE_NAMESPACE", "R9V_CAPTURE_AUTO"}
    }
    settings["R9V_IMAGE"] = image
    for key, value in list(settings.items()):
        if (key.startswith("R9V_DEV_") or key == "R9V_RUNTIME_DESCRIPTOR") and value:
            settings[key] = hashlib.sha256(Path(value).read_bytes()).hexdigest()
    payload = {
        "image": image,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "config": settings,
        "driver": driver,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


if __name__ == "__main__":
    print(
        cache_key(
            sys.argv[1],
            Path(sys.argv[2]).read_bytes(),
            os.environ,
            {
                "kernel": platform.release(),
                "amdgpu": Path("/sys/module/amdgpu/srcversion").read_text().strip()
                if Path("/sys/module/amdgpu/srcversion").exists()
                else None,
            },
        )
    )
