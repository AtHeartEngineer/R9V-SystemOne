import json

import pytest

from tools import capture_routes


def test_route_corpora_are_independent_and_cover_modalities():
    train, held = (capture_routes.corpus(split) for split in ("train", "holdout"))
    assert len(train) == len(held) == 11
    assert {json.dumps(p) for p in train}.isdisjoint({json.dumps(p) for p in held})
    assert any("tools" in p for p in train)
    assert any(isinstance(p["messages"][0]["content"], list) for p in train)


def test_route_failure_disables_collection_and_preserves_response(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(capture_routes, "call", lambda *a: {"error": "worker stopped"})
    with pytest.raises(ValueError, match="request 0"):
        capture_routes.run("fixture", "model", tmp_path, "train")
    assert not (tmp_path / "enable").exists()
    assert (tmp_path / "response-0.json").exists()
    assert not (tmp_path / "result.json").exists()


def test_replay_records_request_before_transport_without_enabling_collection(
    tmp_path, monkeypatch
):
    bodies = []

    def respond(url, endpoint, body):
        index = len(bodies)
        assert not (tmp_path / "enable").exists()
        assert json.loads((tmp_path / f"request-{index}.json").read_text()) == body
        events = [
            json.loads(line)
            for line in (tmp_path / "events.jsonl").read_text().splitlines()
        ]
        assert events[-1]["event"] == "request_start"
        assert events[-1]["index"] == index
        bodies.append(body)
        return {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}

    monkeypatch.setattr(capture_routes, "call", respond)
    capture_routes.run("fixture", "model", tmp_path, "train", collect=False, limit=4)
    assert len(bodies) == 4
    assert [body["messages"] for body in bodies] == [
        p["messages"] for p in capture_routes.corpus("train")[:4]
    ]
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["collection_enabled"] is False
    assert result["corpus_complete"] is False
    assert not (tmp_path / "dump").exists()


def test_transport_failure_keeps_pending_request_and_error(tmp_path, monkeypatch):
    def fail(*args):
        raise TimeoutError("fixture timeout")

    monkeypatch.setattr(capture_routes, "call", fail)
    with pytest.raises(TimeoutError):
        capture_routes.run("fixture", "model", tmp_path, "train")
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [entry["event"] for entry in events] == [
        "session_start",
        "histogram_enabled",
        "request_start",
        "session_error",
    ]
    assert (tmp_path / "request-0.json").exists()
    assert not (tmp_path / "response-0.json").exists()
    assert not (tmp_path / "enable").exists()


def test_failed_evidence_sync_prevents_request_dispatch(tmp_path, monkeypatch):
    def fail_sync(*args):
        raise OSError("fixture disk error")

    calls = []
    monkeypatch.setattr(capture_routes.os, "fsync", fail_sync)
    monkeypatch.setattr(capture_routes, "call", lambda *args: calls.append(args))
    with pytest.raises(OSError, match="disk error"):
        capture_routes.run("fixture", "model", tmp_path, "train")
    assert calls == []
