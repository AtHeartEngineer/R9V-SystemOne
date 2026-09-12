"""Host-only memory pressure evidence; MemAvailable alone is not admission proof."""

import os
import re
from pathlib import Path


def normal_zones(text, page_size=None):
    page_size = page_size or os.sysconf("SC_PAGE_SIZE")
    zones, current = [], None
    for line in text.splitlines():
        match = re.match(r"Node\s+(\d+), zone\s+(\S+)", line)
        if match:
            current = (
                {"node": int(match[1]), "zone": match[2]}
                if match[2] == "Normal"
                else None
            )
            if current is not None:
                zones.append(current)
        elif current is not None:
            match = re.match(r"\s*(pages free|min|low|high)\s+(\d+)\s*$", line)
            if match:
                current[match[1].replace("pages ", "") + "_bytes"] = (
                    int(match[2]) * page_size
                )
    return zones


def snapshot():
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        parts = line.split()
        if parts[0].rstrip(":") in (
            "MemFree",
            "MemAvailable",
            "Unevictable",
            "Mlocked",
            "SwapFree",
            "SwapTotal",
        ):
            mem[parts[0].rstrip(":") + "_bytes"] = int(parts[1]) * 1024
    return {
        "meminfo": mem,
        "normal_zones": normal_zones(Path("/proc/zoneinfo").read_text()),
    }


def exhausted(record):
    """Two successive observations are required by the caller before aborting."""
    zones = record.get("normal_zones", [])
    return bool(zones) and all(
        "free_bytes" in z and "min_bytes" in z and z["free_bytes"] < z["min_bytes"]
        for z in zones
    )
