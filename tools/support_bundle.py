#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Save crash evidence before a container is removed, including after a reboot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tools import capture_runtime as capture
    from tools.profile_state import default_state_dir, validate_state_profile
except ModuleNotFoundError:
    import capture_runtime as capture
    from profile_state import default_state_dir, validate_state_profile


def save_json(path: Path, value: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as output:
        capture.write_record(output, value, 8 * 1024 * 1024)


_PRIVATE_CONFIG_WORDS = ("PASSWORD", "SECRET", "API_KEY", "CREDENTIAL")
_PATH_CONFIG_WORDS = ("_PATH", "_DIR", "_FILE", "_ROOT")


def _file_identity(value: str) -> dict[str, Any]:
    """Describe a configured file without exporting its absolute path/content."""
    path = Path(value).expanduser()
    result: dict[str, Any] = {"basename": path.name or str(path), "path_type": "file"}
    try:
        if path.is_file():
            digest = hashlib.sha256()
            size = path.stat().st_size
            complete = True
            if size <= 64 * 1024 * 1024:
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            else:
                complete = False
            if complete:
                result["sha256"] = digest.hexdigest()
            result["bytes"] = size
            result["hash_status"] = "complete" if complete else "omitted-over-64MiB"
        elif path.is_dir():
            result["path_type"] = "directory"
            result["exists"] = True
        else:
            result["exists"] = False
    except OSError as error:
        result["error"] = type(error).__name__
    return result


def _private_key(key: str) -> bool:
    upper = key.upper()
    return any(word in upper for word in _PRIVATE_CONFIG_WORDS) or upper.endswith("_TOKEN")


def _safe_value(key: str, value: Any) -> Any:
    if _private_key(key):
        return "<redacted>"
    upper = key.upper()
    if isinstance(value, str) and (Path(value).is_absolute() or upper in {'PATH', 'SOURCE'} or any(word in upper for word in _PATH_CONFIG_WORDS)):
        return _file_identity(value)
    if isinstance(value, dict):
        return {str(k): _safe_value(str(k), v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_safe_value(key, item) for item in value]
    return value


def _safe_config(config: dict[str, Any]) -> dict[str, Any]:
    """Keep release-relevant settings while avoiding model/user path leakage."""
    safe: dict[str, Any] = {}
    for key, value in sorted(config.items()):
        safe[str(key)] = _safe_value(str(key), value)
    return safe


def _placement_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Read small placement/catalog records to make headroom failures actionable."""
    records: dict[str, Any] = {}
    keys = (
        "R9V_EXPERT_MANIFEST_PATH",
        "R9V_PLACEMENT_PLAN",
        "R9V_EXPERT_CATALOG_PATH",
        "R9V_CALIBRATION_PATH",
        "R9V_MEMORY_SEED_PATH",
    )
    for key in keys:
        raw = config.get(key)
        if not isinstance(raw, str) or not raw:
            continue
        path = Path(raw).expanduser()
        entry: dict[str, Any] = {"identity": _file_identity(raw)}
        try:
            if path.is_file() and path.stat().st_size <= 2 * 1024 * 1024:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    entry["record"] = _safe_value("record", value)
                else:
                    entry["error"] = "record is not an object"
            elif path.is_file():
                entry["error"] = "record exceeds 2 MiB"
        except (OSError, ValueError, TypeError) as error:
            entry["error"] = type(error).__name__
        records[key] = entry
    return records


def source_identity(root: Path) -> dict[str, Any]:
    """Identify distributed diagnostic code even without Git metadata."""
    names = (
        "r9v", "tools/setup_profile.py", "tools/profile_state.py",
        "tools/profile_doctor.py", "tools/host_preflight.py",
        "tools/support_bundle.py", "tools/capture_runtime.py",
        "tools/watch_runtime.py", "tools/prepare_placement.py",
        "tools/plan_experts.py", "tools/memory_seed.py", "scripts/launch.sh",
    )
    files = {}
    for name in names:
        path = root / name
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                files[name] = {"error": "outside source tree or symlink"}
            elif path.stat().st_size > 2 * 1024 * 1024:
                files[name] = {"error": "source file exceeds 2 MiB"}
            else:
                payload = path.read_bytes()
                files[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        except OSError as error:
            files[name] = {"error": type(error).__name__}
    return {"schema": "r9v.support-source.v1", "files": files,
            "scope": "Selected host launch and diagnostic files; runtime image identity is recorded separately."}


def collect(output: Path, container: str, port: int, setup_state: dict[str, Any] | None = None) -> dict:
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    save_json(output / "source-identity.json", source_identity(Path(__file__).resolve().parents[1]))
    # No full environment, model data, or generated completions are collected.
    # Raw server/kernel logs can still contain user data and need review.
    probes = {
        "container": [
            "docker",
            "inspect",
            "--format",
            capture.INSPECT_FORMAT,
            container,
        ],
        "server-log": ["docker", "logs", "--timestamps", "--tail", "2000", container],
        "kernel-current": [
            "journalctl",
            "-k",
            "-b",
            "0",
            "--no-pager",
            "-n",
            "1000",
            "-o",
            "short-iso-precise",
        ],
        "kernel-previous": [
            "journalctl",
            "-k",
            "-b",
            "-1",
            "--no-pager",
            "-n",
            "1000",
            "-o",
            "short-iso-precise",
        ],
        "gpu-inventory": ["amd-smi", "list"],
        "kernel-version": ["uname", "-a"],
        "revision": [
            "git",
            "-C",
            str(Path(__file__).resolve().parents[1]),
            "rev-parse",
            "HEAD",
        ],
        "runtime-versions": [
            "docker",
            "exec",
            container,
            "python3",
            "-c",
            "import json, torch, importlib.metadata as m; "
            "print(json.dumps({'torch':torch.__version__,'hip':torch.version.hip,"
            "'vllm':m.version('vllm')}))",
        ],
    }
    # docker cp also works after exit, when docker exec can no longer inspect
    # the serving processes. The runtime keeps one small record per TP rank.
    for record_name in ('worker-0', 'worker-1', 'scheduler'):
        probes[f'{record_name}-copy'] = ['docker', 'cp',
            f'{container}:/tmp/r9v-{record_name}.json', str(output / f'{record_name}-record.json')]
    results = {}
    for name, command in probes.items():
        result = capture.command_output(command, timeout=10)
        save_json(output / (name + ".json"), result)
        results[name] = (
            ("partial" if result.get("text") else "unavailable")
            if "error" in result
            else ("collected" if result.get("text", "").strip() else "empty")
        )
    for record_name in ('worker-0', 'worker-1', 'scheduler'):
        path = output / f'{record_name}-record.json'
        if path.is_symlink():
            path.unlink()
            results[f'{record_name}-copy'] = 'unexpected symlink omitted'
            continue
        if path.is_file():
            path.chmod(0o600)
            if path.stat().st_size > 64 * 1024:
                path.write_text('{"error":"worker record exceeded 64 KiB"}\n')
                results[f'{record_name}-copy'] = 'oversized record omitted'
                continue
            try:
                record = json.loads(path.read_text())
                valid = isinstance(record, dict) and bool(record)
                if record_name.startswith('worker-'):
                    valid = valid and record.get('schema') == 'r9v.worker.v1' and record.get('rank') == int(record_name[-1])
                results[f'{record_name}-copy'] = 'collected' if valid else 'invalid record'
            except (OSError, ValueError):
                results[f'{record_name}-copy'] = 'invalid record'
    state = capture.command_json(probes["container"])
    save_json(
        output / "snapshot.json",
        {
            "host": capture.host_snapshot(),
            "gpus": capture.gpu_snapshot(),
            "pcie": capture.pcie_snapshot(),
            "metrics": capture.fetch_metrics(port),
            "cgroup": capture.cgroup_snapshot(int(state.get("pid", 0))),
        },
    )
    config = setup_state.get("config", {}) if isinstance(setup_state, dict) else {}
    if isinstance(config, dict):
        save_json(
            output / "setup-diagnostics.json",
            {
                "schema": "r9v.support-setup.v1",
                "ready": setup_state.get("ready") if isinstance(setup_state, dict) else None,
                "config": _safe_config(config),
                "placement_records": _placement_summary(config),
            },
        )
    if isinstance(setup_state, dict) and isinstance(setup_state.get('latest_capture'), str):
        capture_path = Path(setup_state['latest_capture'])
        copied = []
        for name in ('timeline.jsonl', 'timeline.1.jsonl', 'timeline.2.jsonl', 'collector-events.json', 'support.collector.log', 'support-final.collector.log', 'early.collector.log', 'early-final.collector.log'):
            source = capture_path / name
            if source.is_file() and source.stat().st_size <= 8 * 2**20:
                target = output / name
                descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with source.open('rb') as src, os.fdopen(descriptor, 'wb') as dst:
                    dst.write(src.read(8 * 2**20))
                copied.append(name)
        save_json(output / 'rolling-capture.json', {'source_basename': capture_path.name, 'files': copied})
    output_files = []
    for path in sorted(output.iterdir()):
        if path.is_file() and not path.is_symlink() and path.name != "manifest.json":
            output_files.append({"path": path.name, "bytes": path.stat().st_size,
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {
        "schema": "r9v.support.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "container": container,
        "runtime_state": state,
        "probes": results,
        "required_probe_summary": {
            name: results.get(name, "unavailable")
            for name in ("container", "server-log", "gpu-inventory", "worker-0-copy", "worker-1-copy")
        },
        "limits": "Each command is limited to 256 KiB and 10 seconds; unavailable evidence is explicit.",
        "review": "Review raw server and kernel logs for private data before sharing. Nothing is uploaded.",
        "diagnostics": {
            "setup": "collected" if config else "unavailable",
            "placement_records": "expert maps, headroom targets, quantization/calibration identities included when configured and readable",
        },
        "output_files": output_files,
        "archive": "Use --archive to create a bounded local tar.gz; it is never uploaded.",
    }
    manifest["required_probe_summary"]["complete"] = all(
        value == "collected" for name, value in manifest["required_probe_summary"].items()
        if name != "complete"
    )
    save_json(output / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--container",
        default=os.environ.get("R9V_CONTAINER_NAME"),
    )
    parser.add_argument(
        "--port", type=int, default=os.environ.get("R9V_HOST_PORT")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="new private directory; never overwritten",
    )
    parser.add_argument('--state-dir', type=Path, default=None)
    parser.add_argument('--archive', type=Path, default=None,
                        help='optional local tar.gz of this bounded evidence directory')
    args = parser.parse_args(argv)
    profile_id = os.environ.get('R9V_PROFILE_ID', 'qwen38-flash-next/ud-iq4-xs/dual-r9700-128k')
    args.state_dir = args.state_dir or default_state_dir(profile_id)
    args.output = args.output or args.state_dir / ('support-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f'))
    if args.archive and (args.archive.exists() or args.archive.resolve().is_relative_to(args.output.resolve())):
        parser.error('--archive must be a new file outside the evidence directory')
    state = {}
    state_error = None
    try:
        candidate = json.loads((args.state_dir / 'setup.json').read_text())
        if not isinstance(candidate, dict) or not isinstance(candidate.get('config', {}), dict):
            raise ValueError('saved setup state is malformed')
        validate_state_profile(candidate, profile_id)
        state = candidate
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError) as error:
        state_error = str(error)
    config = state.get('config', {})
    if state_error:
        state = {}
        config = {}
    args.container = args.container or config.get('R9V_CONTAINER_NAME', 'r9v-qwen38-flash-next')
    try:
        args.port = int(args.port if args.port is not None else config.get('R9V_HOST_PORT', '8004'))
    except (ValueError, TypeError):
        parser.error('invalid saved container port; supply --port')
    if (
        not isinstance(args.container, str) or not args.container
        or args.container.startswith("-")
        or not 1 <= args.port <= 65535
    ):
        parser.error("invalid container name or port")
    try:
        # Keep the small public collect(output, container, port) seam usable by
        # callers that inject a collector, while the built-in collector receives
        # setup metadata for release diagnostics.
        if collect.__module__ == __name__:
            result = collect(args.output, args.container, args.port, state)
        else:
            result = collect(args.output, args.container, args.port)
        if state_error:
            save_json(args.output / 'setup-state-error.json', {'error': state_error})
        if state and not (args.output / 'rolling-capture.json').exists():
            if isinstance(state.get('latest_capture'), str) and state['latest_capture']:
                capture_path = Path(state['latest_capture'])
                copied = []
                for name in ('timeline.jsonl', 'timeline.1.jsonl', 'timeline.2.jsonl', 'collector-events.json', 'support.collector.log', 'support-final.collector.log', 'early.collector.log', 'early-final.collector.log'):
                    source = capture_path / name
                    if source.is_file() and source.stat().st_size <= 8 * 2**20:
                        descriptor = os.open(args.output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                        with source.open('rb') as src, os.fdopen(descriptor, 'wb') as dst:
                            dst.write(src.read(8 * 2**20))
                        copied.append(name)
                save_json(args.output / 'rolling-capture.json', {'source_basename': capture_path.name, 'files': copied})
        manifest_path = args.output / 'manifest.json'
        if manifest_path.is_file():
            final_manifest = json.loads(manifest_path.read_text())
            final_manifest['output_files'] = [{'path': p.name, 'bytes': p.stat().st_size,
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(args.output.iterdir())
                if p.is_file() and not p.is_symlink() and p.name != 'manifest.json']
            manifest_path.write_text(json.dumps(final_manifest, indent=2) + '\n')
        if args.archive:
            args.archive.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(args.archive, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'wb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as archive:
                for path in sorted(args.output.iterdir()):
                    if path.is_file() and not path.is_symlink():
                        archive.add(path, arcname=path.name, recursive=False)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        print(f"Support collection failed: {error}")
        return 1
    print(f"Support evidence: {args.output}")
    print(result["review"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
