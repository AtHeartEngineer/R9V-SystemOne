# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
import subprocess


def test_build_uses_pinned_version_without_tags(tmp_path):
    root = Path(__file__).resolve().parents[1]
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
