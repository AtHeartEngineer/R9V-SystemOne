#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Resumable installation and bounded startup for the Qwen Flash Next profile."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request

try:
    from tools.profile_doctor import discover_kfd_gpus
    from tools.verify_package import _sha256
except ModuleNotFoundError:
    from profile_doctor import discover_kfd_gpus
    from verify_package import _sha256

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/qwen38-flash-next/dual-r9700/profile.json"
PLE_BYTES = 28_800_138_240


def save(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def identity(path):
    info = path.stat()
    return [str(path.resolve()), info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns]


def ensure_artifact(artifact, model_dir, receipt, persist, download, full_hash=False):
    path = (model_dir / artifact["path"]).resolve()
    if not path.is_relative_to(model_dir.resolve()):
        raise ValueError("artifact escapes model directory")
    key = artifact["path"]
    old = receipt.get(key, {})
    if (not full_hash and path.is_file() and old.get("identity") == identity(path)
            and old.get("sha256") == artifact["sha256"]):
        print(f"Reusing previously hash-verified file: {key}", flush=True)
        return
    if not path.is_file() or path.stat().st_size != artifact["bytes"]:
        download(key)
    if not path.resolve().is_relative_to(model_dir.resolve()):
        raise ValueError("downloaded artifact escapes model directory")
    before = identity(path)
    print(f"Verifying {key}", flush=True)
    if before[3] != artifact["bytes"] or _sha256(path) != artifact["sha256"]:
        raise ValueError(f"Integrity check failed: {path}; repair this file and retry")
    if identity(path) != before:
        raise ValueError(f"File changed during verification: {path}")
    receipt[key] = {"identity": before, "sha256": artifact["sha256"]}
    persist()


def run(command, *, env=None, capture=False, timeout=None):
    print("Running: " + " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), env=env, check=True,
                          text=True, stdout=subprocess.PIPE if capture else None,
                          timeout=timeout)


def select_devices(bdfs=None):
    gpus = discover_kfd_gpus()
    eligible = []
    for index, gpu in enumerate(gpus):
        if gpu.gfx_target != 120001:
            continue
        try:
            vram = int((Path('/sys/bus/pci/devices') / gpu.bdf /
                        'mem_info_vram_total').read_text())
        except (OSError, ValueError):
            continue
        if vram >= 31 * 2**30:
            eligible.append((index, gpu))
    if bdfs:
        wanted = [b.strip().lower() for b in bdfs.split(',')]
        by_bdf = {g.bdf: (i, g) for i, g in eligible}
        if len(wanted) != 2 or len(set(wanted)) != 2 or any(b not in by_bdf for b in wanted):
            raise ValueError("--gpu-bdfs must select two distinct 32 GiB gfx1201 GPUs")
        eligible = [by_bdf[b] for b in wanted]
    if len(eligible) != 2:
        raise ValueError("Need exactly two 32 GiB gfx1201 GPUs; use --gpu-bdfs to select a pair")
    nodes = [Path('/dev/kfd')]
    for _, gpu in eligible:
        if gpu.render_minor is None:
            raise ValueError(f"Cannot resolve render device for {gpu.bdf}")
        nodes.append(Path('/dev/dri') / f'renderD{gpu.render_minor}')
    for node in nodes:
        if not os.access(node, os.R_OK | os.W_OK):
            raise ValueError(f"Device access required: {node}")
    return {"R9V_VISIBLE_DEVICES": ','.join(str(i) for i, _ in eligible),
            "R9V_EXPECTED_GPU_BDFS": ','.join(g.bdf for _, g in eligible)}


def profile_settings():
    """Freeze the effective profile defaults and explicit R9V overrides together."""
    profile_env = PROFILE.parent / 'profile.env'
    result = run(['bash', '-c', 'set -a; source "$1"; env -0', 'r9v-setup', profile_env],
                 capture=True, timeout=10)
    excluded = {'R9V_CONFIG_FILE', 'R9V_PROFILE', 'R9V_PROFILE_ID', 'R9V_PROFILE_ROOT',
                'R9V_REPO_ROOT', 'R9V_SYS_ROOT', 'R9V_PROC_ROOT', 'R9V_BASE_IMAGE',
                'R9V_RUNTIME_ONLY', 'R9V_MAX_JOBS', 'R9V_VLLM_VERSION'}
    return {key: value for entry in result.stdout.split('\0') if '=' in entry
            for key, value in [entry.split('=', 1)]
            if key.startswith('R9V_') and key not in excluded}


def container_user_args():
    """Root in rootless Docker maps to the daemon owner, not host root."""
    options = json.loads(run(['docker', 'info', '--format', '{{json .SecurityOptions}}'],
                             capture=True, timeout=30).stdout)
    if not isinstance(options, list) or not all(isinstance(item, str) for item in options):
        raise ValueError('Docker did not report valid security options')
    if 'name=rootless' in options:
        return ['--user', '0:0']
    result = ['--user', f'{os.getuid()}:{os.getgid()}']
    if 'name=userns' in options:
        # Rootful daemon remapping would otherwise turn the host UID into a
        # subordinate UID with no write access to the bind-mounted data directory.
        result += ['--userns', 'host']
    return result


def setup(args, state, state_path):
    profile = json.loads(PROFILE.read_text())
    descriptor = ROOT / profile['descriptors']['model_package']
    package = json.loads(descriptor.read_text())
    runtime = json.loads((ROOT / profile['descriptors']['runtime']).read_text())
    image = args.image or runtime.get('distribution', {}).get('image')
    if not image and not args.build:
        raise ValueError("No published image is configured yet. Supply --image REGISTRY/IMAGE@sha256:DIGEST, "
                         "--image LOCAL_IMAGE --local-image, or explicitly opt into --build.")
    if image and image.startswith('-'):
        raise ValueError('Image must not start with a dash')
    if image and not args.local_image and not re.fullmatch(r'[^\s]+@sha256:[0-9a-f]{64}', image):
        raise ValueError("Remote images must be pinned with @sha256:<64 lowercase hex digits>")
    if not args.accept_model_license:
        raise ValueError("Read the model license and supply --accept-model-license")
    if os.environ.get('R9V_CONFIG_FILE'):
        raise ValueError("Unset R9V_CONFIG_FILE for setup; this flow saves its own machine configuration")
    run(['docker', 'info', '--format', '{{.DockerRootDir}}'], capture=True, timeout=30)
    config = profile_settings()
    config.update(select_devices(args.gpu_bdfs))
    run([ROOT / 'scripts/profile-doctor.sh', '--host-only'],
        env={**os.environ, **config, 'R9V_RUNTIME_PREBUILT': '1'})
    model = Path(args.model_dir or os.environ.get('R9V_MODEL_DIR', '')).expanduser().resolve()
    if not args.model_dir and not os.environ.get('R9V_MODEL_DIR'):
        raise ValueError("Supply --model-dir to choose the model storage destination")
    data = Path(args.data_dir).expanduser().resolve() if args.data_dir else model / 'r9v-data'
    model.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    artifacts = [a for a in package['artifacts'] if a.get('required', True)]
    missing = sum(a['bytes'] for a in artifacts if not (model / a['path']).is_file()
                  or (model / a['path']).stat().st_size != a['bytes'])
    ple = (Path(args.ple_path).expanduser().resolve() if args.ple_path
           else data / 'per_layer_token_embd.iq4_nl.bin')
    ple.parent.mkdir(parents=True, exist_ok=True)
    if ple.exists() and (not ple.is_file() or ple.stat().st_size != PLE_BYTES):
        raise ValueError(f'Unexpected PLE payload: {ple}; choose a new path or repair the file')
    derived = 0 if ple.is_file() else PLE_BYTES
    print(f"Model destination: {model}; missing/unfinished files up to {missing / 2**30:.2f} GiB")
    print(f"PLE destination: {ple}; new payload {derived / 2**30:.2f} GiB")
    print("Runtime image/build storage is additional in Docker's data root; allow tens of GiB.")
    required = {model.stat().st_dev: [model, missing]}
    entry = required.setdefault(ple.parent.stat().st_dev, [ple.parent, 0])
    entry[1] += derived
    for location, amount in required.values():
        if shutil.disk_usage(location).free < amount + 2**30:
            raise ValueError(f"Insufficient space at {location}: need {amount / 2**30:.2f} GiB plus 1 GiB reserve")
    if args.build:
        run([ROOT / 'scripts/build-image.sh'])
        image = os.environ.get('R9V_IMAGE', 'r9v-qwen38-flash-next:latest')
    elif not args.local_image:
        run(['docker', 'pull', image])
    user_args = container_user_args()
    image_id = run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'],
                   capture=True, timeout=30).stdout.strip()
    # Save the immutable local image ID, including when the input was a mutable local tag.
    state['ready'] = False
    save(state_path, state)
    distribution = package['distribution']
    receipt = state.setdefault('artifacts', {})
    def download(relative):
        if not shutil.which('hf'):
            raise ValueError("Install the Hugging Face hf CLI to download missing model files")
        run(['hf', 'download', distribution['repository'], relative, '--revision',
             distribution['revision'], '--local-dir', model])
    for artifact in artifacts:
        ensure_artifact(artifact, model, receipt, lambda: save(state_path, state),
                        download, args.hash)
    shards = ['/models/' + a['path'] for a in artifacts
              if a['path'].startswith('target/') and a['path'].endswith('.gguf')]
    run(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'python3',
         *user_args, '--security-opt', 'label=disable',
         '--volume', f'{ROOT}:/r9v:ro', '--volume', f'{model}:/models:ro',
         '--volume', f'{ple.parent}:/r9v-data', image_id, '/r9v/tools/prepare_ple.py',
         *shards, '--output', '/r9v-data/' + ple.name])
    config.update(R9V_IMAGE=image_id, R9V_MODEL_DIR=str(model), R9V_PLE_PATH=str(ple),
                  R9V_CACHE_DIR=str(data / 'cache'), R9V_RUNTIME_PREBUILT='1')
    state['config'] = config
    state['ready'] = False
    save(state_path, state)
    env = {**os.environ, **config}
    run([ROOT / 'scripts/profile-doctor.sh'], env=env)
    state['ready'] = True
    save(state_path, state)
    print(f"Setup complete. Configuration: {state_path}. Run ./r9v start qwen38")


def start(args, state, state_path):
    if not state.get('ready'):
        raise ValueError("Run setup successfully before start")
    env = {**os.environ, **state['config']}
    # Saved paths/image/GPU identity must not be silently replaced by a shell config.
    env.pop('R9V_CONFIG_FILE', None)
    container = env.get('R9V_CONTAINER_NAME', 'r9v-qwen38-flash-next')
    port = env.get('R9V_HOST_PORT', '8004')
    run([ROOT / 'scripts/launch.sh'], env=env)
    deadline = time.monotonic() + args.timeout
    print("Waiting for model loading, compilation and graph capture; logs remain in Docker.", flush=True)
    while time.monotonic() < deadline:
        status = run(['docker', 'inspect', container, '--format', '{{.State.Status}}'],
                     capture=True, timeout=10).stdout.strip()
        if status != 'running':
            raise ValueError(f"Container stopped during startup: {status}")
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3) as response:
                if response.status == 200:
                    run([ROOT / 'scripts/profile-doctor.sh', '--runtime'], env=env)
                    print(f"Ready: http://127.0.0.1:{port}/v1")
                    return
        except OSError:
            pass
        time.sleep(min(5, max(0, deadline - time.monotonic())))
    raise ValueError(f"Readiness timed out after {args.timeout}s; container retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'start'])
    parser.add_argument('--model-dir')
    parser.add_argument('--data-dir')
    parser.add_argument('--ple-path', help='reuse or prepare a PLE file at this exact path')
    default_state = Path(os.environ.get('XDG_STATE_HOME', str(Path.home() / '.local/state'))) / 'r9v/qwen38'
    parser.add_argument('--state-dir', type=Path, default=default_state)
    parser.add_argument('--image')
    parser.add_argument('--local-image', action='store_true')
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--gpu-bdfs')
    parser.add_argument('--accept-model-license', action='store_true')
    parser.add_argument('--hash', action='store_true', help='rehash all artifacts, ignoring verification receipts')
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 86400:
        parser.error('--timeout must be 1..86400 seconds')
    if args.build and (args.image or args.local_image):
        parser.error('--build cannot be combined with --image/--local-image')
    args.state_dir = args.state_dir.expanduser().resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_path = args.state_dir / 'setup.json'
    with (args.state_dir / 'setup.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another setup/start is already using this state directory')
        state = {}
        try:
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if not isinstance(state, dict):
                raise ValueError('Invalid setup state: expected a JSON object')
            if args.action == 'setup':
                setup(args, state, state_path)
            else:
                start(args, state, state_path)
        except (ValueError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as error:
            print(f'Failed: {error}', file=sys.stderr)
            if args.action == 'start' and state.get('config'):
                output = args.state_dir / ('failure-' + time.strftime('%Y%m%d-%H%M%S'))
                try:
                    run([ROOT / 'scripts/profile-diagnostics.sh', 'support', '--output', output],
                        env={**os.environ, **state['config'], 'R9V_CONFIG_FILE': ''}, timeout=180)
                    print(f'Diagnostics: {output}; review logs before sharing')
                except (OSError, subprocess.SubprocessError) as capture_error:
                    print(f'Diagnostic collection failed: {capture_error}', file=sys.stderr)
            return 130 if isinstance(error, KeyboardInterrupt) else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
