"""CPU-only admission and fault identity for the reviewed timeout recorder."""

import datetime
import json

SOURCE = "r9v-timeout-trace"


class TraceGate:
    def __init__(self, identity):
        self.identity = identity
        self.host_at = None
        self.trace_at = None
        self.highwater = {"host": 0, "trace": 0}
        self.fault = None

    def observe(self, envelope, now):
        try:
            if (
                envelope.get("channel") != "userspace"
                or envelope["source"][0] != self.identity["sender"]
            ):
                return None
            received = datetime.datetime.fromisoformat(envelope["received_utc"])
            if received.tzinfo is None or not 0 <= now - received.timestamp() <= 10:
                return None
            record = json.loads(envelope["message"])
            if record.get("boot_id") != self.identity["boot_id"]:
                return None
            source, kind = record.get("source"), record.get("kind")
            if source == SOURCE and kind in ("ready", "heartbeat", "first_timeout"):
                channel = "trace"
            elif (
                source is None
                and kind == "heartbeat"
                and type(record.get("sequence")) is int
                and isinstance(record.get("memory"), dict)
            ):
                channel = "host"
            else:
                return None
            stamp = record.get("monotonic_ns")
            expected = self.identity["monotonic_ns"] + int(
                (now - self.identity["wall_time"]) * 1e9
            )
            if type(stamp) is not int or not -2e9 <= expected - stamp <= 45e9:
                return None
            if (
                stamp <= self.highwater[channel]
                or stamp < self.identity["recorder_start_ns"]
            ):
                return None
            self.highwater[channel] = stamp
            if channel == "host":
                self.host_at = received.timestamp()
                return "host"
            if kind == "first_timeout" or (
                kind == "heartbeat" and record.get("captured") is True
            ):
                self.fault = self.fault or record
                return "fault"
            # Readiness is retained, but only an explicit uncaptured heartbeat admits.
            if kind == "heartbeat" and record.get("captured") is False:
                self.trace_at = received.timestamp()
                return "trace"
        except (KeyError, TypeError, ValueError, AttributeError, IndexError):
            return None
        return None

    def ready(self, now):
        return (
            self.fault is None
            and self.host_at is not None
            and self.trace_at is not None
            and 0 <= now - self.host_at <= 15
            and 0 <= now - self.trace_at <= 45
        )

    def failure(self, now):
        if self.fault is not None:
            return "timeout recorder captured a fault"
        if self.host_at is None or now - self.host_at > 25:
            return "host heartbeat absent"
        if self.trace_at is None or now - self.trace_at > 45:
            return "timeout recorder heartbeat absent"
        return None


def validate_identity(current, reviewed):
    """Fail closed on recorder replacement, source drift or insufficient admission evidence."""
    for key in (
        "boot_id",
        "kernel",
        "hashes",
        "source_revision",
        "source_hashes",
        "image",
        "manifest",
        "recorder_start_ns",
        "recorder_cmdline",
    ):
        if current.get(key) != reviewed.get(key):
            raise ValueError("preflight identity mismatch: " + key)
    service = current["service"]
    if any(
        service.get(k) != v
        for k, v in {
            "ActiveState": "active",
            "SubState": "running",
            "Result": "success",
        }.items()
    ):
        raise ValueError("recorder service is not active/running/success")
    if service.get("MainPID") != reviewed["service"]["MainPID"]:
        raise ValueError("recorder process changed; inspect before rearming")
    if (
        current["symlinks"]
        or current["installed_uid"] != 0
        or current["unit_uid"] != 0
        or current["installed_mode"] != 0o755
        or current["unit_mode"] != 0o644
    ):
        raise ValueError("recorder ownership or mode differs")
    if current["kfd_users"] or current["kfd_status"] != 1:
        raise ValueError("GPU compute use present or unavailable")
    if any(name.startswith("r9v-") for name in current["running_containers"]):
        raise ValueError("conflicting R9V container")
    pressure = current["pressure"]
    if pressure["meminfo"]["MemAvailable_bytes"] < 64 * 2**30:
        raise ValueError("less than 64 GiB host memory available")
    zones = pressure["normal_zones"]
    if not zones or any(z["free_bytes"] <= z["high_bytes"] for z in zones):
        raise ValueError("Normal-zone free memory is not above high watermark")
