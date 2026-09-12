import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import qualify_runtime as qualification
from tools.qualify_runtime import validate


def protocol():
    return {
        "config": {"R9V_IMAGE": "test"},
        "prompt": "reference",
        "arms": [{"name": "a1"}, {"name": "b"}, {"name": "a2"}],
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("seconds", 2701),
        ("seconds", 179),
        ("warmups", 0),
        ("trials", 2),
        ("max_tokens", True),
        ("startup_seconds", 1801),
        ("startup_seconds", 0),
    ],
)
def test_qualification_rejects_unbounded_or_unmatched_protocol(key, value):
    data = protocol()
    data[key] = value
    with pytest.raises(ValueError):
        validate(data)


@pytest.mark.parametrize("context", [True, 0, 131073])
def test_qualification_rejects_invalid_partial_context(context):
    data = protocol()
    data["workload_context"] = context
    with pytest.raises(ValueError, match="workload_context"):
        validate(data)


def test_qualification_rejects_duplicate_arms_without_overwriting_evidence():
    data = protocol()
    data["arms"].append({"name": "a1"})
    with pytest.raises(ValueError, match="unique"):
        validate(data)


def test_qualification_defaults_reserve_bounded_repeated_measurement():
    data = validate(protocol())
    assert (data["seconds"], data["warmups"], data["trials"]) == (2700, 2, 3)


def test_failed_launch_claims_only_new_container_for_cleanup(monkeypatch):
    absent = SimpleNamespace(returncode=1, stdout="")
    now = 1789210000.0
    monkeypatch.setattr(qualification.time, "time", lambda: now + 5)
    created = datetime.datetime.fromtimestamp(now + 1, datetime.timezone.utc).isoformat()
    present = SimpleNamespace(returncode=0, stdout=json.dumps([{
        "Name": "/new", "Image": "sha256:image", "Created": created
    }]))
    monkeypatch.setattr(qualification.subprocess, "run", lambda *a, **k: absent)
    assert not qualification.container_created_after_launch("new", "sha256:image", now, False)
    monkeypatch.setattr(qualification.subprocess, "run", lambda *a, **k: present)
    assert qualification.container_created_after_launch("new", "sha256:image", now, True)
    assert not qualification.container_created_after_launch("new", "sha256:other", now, True)
    assert not qualification.container_created_after_launch("existing", "sha256:image", now, True)
    recent_unowned = SimpleNamespace(returncode=0, stdout=json.dumps([{
        "Name": "/new", "Image": "sha256:image",
        "Created": datetime.datetime.fromtimestamp(now - 0.1, datetime.timezone.utc).isoformat()
    }]))
    monkeypatch.setattr(qualification.subprocess, "run", lambda *a, **k: recent_unowned)
    assert not qualification.container_created_after_launch("new", "sha256:image", now, True)
    old = SimpleNamespace(returncode=0, stdout=json.dumps([{
        "Name": "/new", "Image": "sha256:image", "Created": "2020-01-01T00:00:00Z"
    }]))
    monkeypatch.setattr(qualification.subprocess, "run", lambda *a, **k: old)
    assert not qualification.container_created_after_launch("new", "sha256:image", now, True)
    malformed = SimpleNamespace(returncode=0, stdout="{}")
    monkeypatch.setattr(qualification.subprocess, "run", lambda *a, **k: malformed)
    assert not qualification.container_created_after_launch("new", "sha256:image", now, True)


@pytest.mark.parametrize("collect", [None, False, True])
@pytest.mark.parametrize("route_failure", [False, True])
def test_session_freezes_inputs_and_stops_its_container(tmp_path, monkeypatch, collect, route_failure):
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "r9v").write_text("original wrapper")
    (root / "scripts/launch.sh").write_text("original launcher")
    profile = root / "profile.env"
    profile.write_text("original defaults")
    manifest = root / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(qualification, "ROOT", root)
    data = protocol()
    data["arms"] = [{"name": "a"}]
    if route_failure:
        if collect is None:
            pytest.skip("route failure requires route workload")
        data["arms"].append({"name": "must-not-start"})
        data["continue_on_workload_failure"] = False
    if collect is not None:
        data["arms"][0].update(route_split="train", route_collect=collect, route_limit=4)
        data["config"]["R9V_ROUTE_PROFILE_DIR"] = str(tmp_path / "routes")
    data["config"].update(
        R9V_EXPECTED_GPU_BDFS="a,b",
        R9V_PROFILE=str(profile),
        R9V_EXPERT_MANIFEST_PATH=str(manifest),
        R9V_SERVED_MODEL_NAME="test",
    )
    monkeypatch.setattr(
        qualification,
        "memory_snapshot",
        lambda _: {"host_available_bytes": 100 * 2**30},
    )
    monkeypatch.setattr(
        qualification,
        "gpu_free_snapshot",
        lambda bdfs: [{"bdf": bdf, "free_bytes": 4 * 2**30} for bdf in bdfs],
    )
    monkeypatch.setattr(qualification, "cgroup_snapshot", lambda _: {})
    monkeypatch.setattr(
        qualification.subprocess,
        "check_output",
        lambda args, **kw: "sha256:fixed" if args[1] == "image" else "123",
    )
    monkeypatch.setattr(
        qualification.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1)
    )

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b"metrics"

    monkeypatch.setattr(
        qualification.urllib.request, "urlopen", lambda *a, **kw: Response()
    )
    session = qualification.Session(data, tmp_path / "evidence")
    commands = []
    lifecycle = []
    monkeypatch.setattr(session, "collect_early", lambda reason: lifecycle.append("capture"))

    def command(args, path, **kwargs):
        commands.append(list(map(str, args)))
        if list(map(str, args))[:2] == ["docker", "stop"]:
            lifecycle.append("stop")
        if route_failure and path.name == "routes.log":
            raise RuntimeError("injected route failure")
        if Path(args[0]).name == "launch.sh":
            (root / "scripts/launch.sh").write_text("changed checkout")
            assert Path(args[0]).read_text() == "original launcher"
            assert (session.root / "r9v").read_text() == "original wrapper"
            assert kwargs["env"]["R9V_IMAGE"] == "sha256:fixed"
            assert Path(kwargs["env"]["R9V_EXPERT_MANIFEST_PATH"]).read_text() == "{}"
        if path.name.startswith("request-"):
            path.write_text(
                json.dumps({"finish_reason": "length", "tg_tokens_per_second": 70})
            )
        elif path.name == "workers.json":
            path.write_text("[]")
        else:
            path.write_text("")

    monkeypatch.setattr(session, "command", command)
    assert session.run() == (1 if route_failure else 0)
    if route_failure:
        assert lifecycle == ["capture", "stop"]
        assert not (session.output / "must-not-start").exists()
    assert any(args[:2] == ["docker", "stop"] for args in commands)
    routes = [args for args in commands if any(arg.endswith("capture_routes.py") for arg in args)]
    if collect is not None:
        assert len(routes) == 1
        assert ("--no-collect" in routes[0]) is (not collect)
        assert routes[0][routes[0].index("--limit") + 1] == "4"
        assert routes[0][routes[0].index("--directory") + 1] == str(tmp_path / "routes")
    else:
        assert routes == []
    result = json.loads((session.output / "results.json").read_text())[0]
    if route_failure:
        assert not result["passed"]
        assert result["error"] == "injected route failure"
    else:
        assert result["passed"] and result["tg_samples"] == [70, 70, 70]


def test_route_arm_can_skip_semantic_workload_without_changing_other_arms():
    data = protocol()
    data["workload"] = True
    data["arms"][1]["workload"] = False
    result = validate(data)
    assert result["workload"] is True
    assert result["arms"][1]["workload"] is False
    data["arms"][1]["workload"] = "false"
    with pytest.raises(ValueError, match="boolean"):
        validate(data)


def test_route_identity_uses_verified_package_runtime_and_effective_config(tmp_path, monkeypatch):
    monkeypatch.setattr(qualification, "ROOT", tmp_path)
    runtime = tmp_path / "runtime.json"
    runtime.write_text('{"id":"runtime"}')
    profile_dir = tmp_path / "profiles" / "p"
    profile_dir.mkdir(parents=True)
    profile_env = profile_dir / "profile.env"
    profile_env.write_text("")
    package = tmp_path / "package.json"
    package.write_text('{"id":"qwen-package"}')
    (profile_dir / "profile.json").write_text(json.dumps({
        "model_package": "qwen-package", "descriptors": {"model_package": "package.json"}}))
    package_hash = __import__("hashlib").sha256(package.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    data = protocol()
    data["config"].update(
        R9V_RUNTIME_DESCRIPTOR=str(runtime),
        R9V_MODEL_PACKAGE="qwen-package",
        R9V_MODEL_PACKAGE_SHA256=package_hash,
        R9V_EXPECTED_GPU_BDFS="a,b",
        R9V_PROFILE=str(profile_env),
        R9V_EXPERT_MANIFEST_PATH=str(manifest),
        R9V_IMAGE="sha256:image",
    )
    session = qualification.Session(data, tmp_path / "evidence")
    identity = session.route_identity(data["config"])
    assert identity["model_package"] == "qwen-package"
    assert identity["model_hash"] == package_hash
    assert len(identity["runtime_hash"]) == 64
    assert len(identity["config_hash"]) == 64
    holdout = dict(data["config"], R9V_CONTAINER_NAME="other", R9V_ROUTE_PROFILE_DIR="/other/routes")
    assert session.route_identity(holdout) == identity
    changed = dict(data["config"], R9V_IMAGE="sha256:changed")
    assert session.route_identity(changed)["runtime_hash"] != identity["runtime_hash"]
    with pytest.raises(ValueError, match="model package identity"):
        session.route_identity(dict(data["config"], R9V_MODEL_PACKAGE_SHA256="b" * 64))


@pytest.mark.parametrize("key,value", [
    ("route_collect", "false"), ("route_collect", 0),
    ("route_limit", True), ("route_limit", 0), ("route_limit", 12),
])
def test_invalid_route_controls_are_rejected_before_launch(key, value):
    data = protocol()
    data["arms"][0].update(route_split="train", **{key: value})
    with pytest.raises(ValueError, match=key):
        validate(data)


@pytest.mark.parametrize("key,value", [("route_collect", False), ("route_limit", 4)])
def test_route_controls_require_replay_arm(key, value):
    data = protocol()
    data["arms"][0][key] = value
    with pytest.raises(ValueError, match="require route_split"):
        validate(data)


@pytest.fixture(autouse=True)
def no_real_driver_probes(monkeypatch):
    from types import SimpleNamespace

    from tools import qualify_runtime, watch_runtime

    def factory(*a, **kw):
        return SimpleNamespace(poll=lambda: {"status": "ok"}, close=lambda: None)

    monkeypatch.setattr(watch_runtime, "DriverSampler", factory)
    monkeypatch.setattr(qualify_runtime, "DriverSampler", factory)


def test_reclaim_gate_records_samples_and_requires_same_gpu_baseline(tmp_path, monkeypatch):
    data = protocol()
    data["config"]["R9V_EXPECTED_GPU_BDFS"] = "a,b"
    session = qualification.Session(data, tmp_path / "evidence")
    session.output.mkdir()
    session.deadline = qualification.time.monotonic() + 300
    readings = iter([
        [{"bdf": "a", "free_bytes": 3 * 2**30}, {"bdf": "b", "free_bytes": 4 * 2**30}],
        [{"bdf": "a", "free_bytes": 4 * 2**30}, {"bdf": "b", "free_bytes": 4 * 2**30}],
    ])
    monkeypatch.setattr(qualification, "gpu_free_snapshot", lambda _: next(readings))
    monkeypatch.setattr(qualification.time, "sleep", lambda _: None)
    recovered, evidence = session.reclaim_gpu_memory(
        session.output,
        [{"bdf": "a", "free_bytes": 4 * 2**30}, {"bdf": "b", "free_bytes": 4 * 2**30}],
    )
    assert recovered
    assert len(evidence["samples"]) == 2
    assert json.loads((session.output / "cleanup-reclamation.json").read_text())["recovered"]
