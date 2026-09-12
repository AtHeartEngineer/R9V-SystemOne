import copy
import datetime
import json
from pathlib import Path

import pytest

from tools.remote_watch import RunTracker
from tools.trace_gate import TraceGate, validate_identity


@pytest.fixture
def identity():
    return {
        "sender": "192.168.1.231",
        "boot_id": "boot-a",
        "monotonic_ns": 1000_000_000_000,
        "wall_time": 100,
        "recorder_start_ns": 900_000_000_000,
    }


def envelope(
    identity,
    kind="heartbeat",
    source="r9v-timeout-trace",
    stamp=1000_000_000_000,
    **fields,
):
    row = {
        "kind": kind,
        "boot_id": identity["boot_id"],
        "monotonic_ns": stamp,
        **fields,
    }
    if source is not None:
        row["source"] = source
    else:
        row.update(sequence=10, memory={})
    return {
        "channel": "userspace",
        "source": [identity["sender"], 1234],
        "received_utc": datetime.datetime.fromtimestamp(
            100, datetime.timezone.utc
        ).isoformat(),
        "message": json.dumps(row),
    }


def test_recorder_cannot_mask_missing_host_heartbeat(identity):
    gate = TraceGate(identity)
    assert gate.observe(envelope(identity, captured=False), 100) == "trace"
    assert not gate.ready(100)
    assert gate.failure(100) == "host heartbeat absent"
    assert gate.observe(envelope(identity, source=None), 100) == "host"
    assert gate.ready(100)
    message = envelope(identity, stamp=1030_000_000_000, captured=False)
    message["received_utc"] = datetime.datetime.fromtimestamp(
        130, datetime.timezone.utc
    ).isoformat()
    gate.observe(message, 130)
    assert gate.failure(130) == "host heartbeat absent"


@pytest.mark.parametrize(
    "change",
    [
        {"boot_id": "wrong"},
        {"monotonic_ns": 800_000_000_000},
        {"monotonic_ns": 1100_000_000_000},
        {"source": "other"},
    ],
)
def test_wrong_identity_or_stale_rewrapped_event_rejected(identity, change):
    gate = TraceGate(identity)
    msg = envelope(identity, kind="first_timeout")
    msg["message"] = json.dumps({**json.loads(msg["message"]), **change})
    assert gate.observe(msg, 100) is None
    assert gate.fault is None


def test_replay_and_stale_receiver_rejected(identity):
    gate = TraceGate(identity)
    msg = envelope(identity, captured=False)
    assert gate.observe(msg, 100) == "trace"
    assert gate.observe(msg, 101) is None
    assert TraceGate(identity).observe(msg, 111) is None
    msg["source"][0] = "192.168.1.99"
    assert TraceGate(identity).observe(msg, 100) is None


@pytest.mark.parametrize(
    "kind,fields",
    [
        ("first_timeout", {"trace": "ring=comp_1.0.0 pasid=7"}),
        ("heartbeat", {"captured": True}),
    ],
)
def test_timeout_or_captured_heartbeat_latches_fault_and_captures_once(
    identity, kind, fields
):
    import uuid

    tracker = RunTracker()
    gate = TraceGate(identity)
    run = str(uuid.uuid4())
    tracker.observe({"kind": "r9v.session_start", "run_id": run, "seconds": 180}, 1)
    tracker.observe({"kind": "r9v.progress", "run_id": run, "container": "r9v-test"}, 2)
    msg = envelope(identity, kind=kind, **fields)
    assert tracker.observe_trace(gate, msg, 3, 100) == "fault"
    assert len(tracker.due(3)) == 1
    assert tracker.due(4) == []
    fresh = envelope(identity, stamp=1001_000_000_000, captured=False)
    gate.observe(fresh, 101)
    assert not gate.ready(101)
    assert gate.fault["kind"] == kind


def test_ready_is_not_uncaptured_admission(identity):
    gate = TraceGate(identity)
    gate.observe(envelope(identity, kind="ready"), 100)
    gate.observe(envelope(identity, source=None), 100)
    assert not gate.ready(100)


@pytest.mark.parametrize(
    "key,value",
    [
        ("kernel", "wrong"),
        ("hashes", {}),
        ("source_hashes", {}),
        ("recorder_start_ns", 0),
        ("installed_uid", 1000),
        ("symlinks", True),
        ("kfd_users", "123"),
    ],
)
def test_live_identity_gate_rejects_changed_recorder_source_or_gpu_use(key, value):
    # Fixture identity is the reviewed read-only preflight, not runtime initialization.
    original = json.loads(
        Path(__file__).with_name("trace-preflight-fixture.json").read_text()
    )
    validate_identity(original, original)
    changed = copy.deepcopy(original)
    changed[key] = value
    with pytest.raises(ValueError):
        validate_identity(changed, original)


def test_live_gate_rejects_service_failure_and_memory_watermark():
    original = json.loads(
        Path(__file__).with_name("trace-preflight-fixture.json").read_text()
    )
    changed = copy.deepcopy(original)
    changed["service"]["ActiveState"] = "failed"
    with pytest.raises(ValueError):
        validate_identity(changed, original)
    changed = copy.deepcopy(original)
    changed["pressure"]["normal_zones"][0]["free_bytes"] = 0
    with pytest.raises(ValueError):
        validate_identity(changed, original)
