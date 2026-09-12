#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify the exact tested image before publishing it; never rebuild on publish."""

import argparse
import json
import subprocess
from pathlib import Path

GATES = {"correctness", "headroom", "throughput", "setup", "restart", "worker_probes"}


def validate_receipt(receipt, image_id, revision):
    if receipt.get("schema") != "r9v.release-qualification.v1":
        raise ValueError("unsupported qualification receipt")
    if (
        receipt.get("image_id") != image_id
        or receipt.get("source_revision") != revision
    ):
        raise ValueError("qualification does not describe this exact image/source")
    gates = receipt.get("gates", {})
    missing = sorted(g for g in GATES if gates.get(g) is not True)
    if missing:
        raise ValueError("release gates not passed: " + ", ".join(missing))
    if not receipt.get("evidence") or not receipt.get("limitations"):
        raise ValueError(
            "qualification must record evidence and bounded-test limitations"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        inspected = json.loads(
            subprocess.check_output(
                ["docker", "image", "inspect", args.image], text=True, timeout=20
            )
        )[0]
        if (
            inspected["Config"]
            .get("Labels", {})
            .get("org.opencontainers.image.revision")
            != revision
        ):
            raise ValueError("image source label differs from the release checkout")
        validate_receipt(
            json.loads(args.receipt.read_text()), inspected["Id"], revision
        )
        print(inspected["Id"])
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Publication blocked: {error}\n")


if __name__ == "__main__":
    main()
