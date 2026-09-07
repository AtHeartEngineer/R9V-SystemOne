# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
import subprocess


def test_build_uses_pinned_version_without_tags(tmp_path):
    import shutil

    source_root = Path(__file__).resolve().parents[1]
    root = tmp_path / 'repo'
    (root / 'scripts').mkdir(parents=True)
    shutil.copy2(source_root / 'scripts/build-image.sh', root / 'scripts/build-image.sh')
    for relative in ('vendor/vllm/docker/Dockerfile.r9v_rocm714',
                     'vendor/vllm-gguf-plugin/setup.py', 'kernels/r9v-gfx1201/README.md'):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    for repo in (root, root / 'vendor/vllm'):
        subprocess.run(['git', 'init', '-q', str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test',
                        '-c', 'user.email=test@example.invalid', 'commit',
                        '--allow-empty', '-qm', 'fixture'], check=True)
    revision = subprocess.check_output(['git', '-C', str(root / 'vendor/vllm'),
                                       'rev-parse', 'HEAD'], text=True).strip()
    runtime = root / 'runtimes/qwen38-flash-next-gfx1201-v1/runtime.json'
    runtime.parent.mkdir(parents=True)
    runtime.write_text(json.dumps({'source': {'vllm_revision': revision,
        'vllm_package_version': '0.26.1rc0+r9v.g' + revision[:12]}}))
    # Run the real shell orchestration; Docker calls are recorded, never built.
    fake = tmp_path / 'docker'
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$TEST_DOCKER_CALLS"\n')
    fake.chmod(0o755)
    env = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH'],
           'TEST_DOCKER_CALLS': str(tmp_path / 'calls'), 'R9V_RUNTIME_ONLY': '0'}
    env.pop('R9V_VLLM_VERSION', None)
    result = subprocess.run([str(root / 'scripts/build-image.sh')], env=env,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / 'calls').read_text()
    source = json.loads((root / 'runtimes/qwen38-flash-next-gfx1201-v1/runtime.json').read_text())['source']
    assert 'R9V_VLLM_VERSION=' + source['vllm_package_version'] in calls
    assert 'R9V_REVISION=' in calls
