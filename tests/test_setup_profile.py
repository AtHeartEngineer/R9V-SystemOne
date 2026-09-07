# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import setup_profile as setup


def artifact(data=b'weights'):
    return {'path': 'target/model.gguf', 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def test_resume_then_mutation_forces_verification(tmp_path, monkeypatch):
    item = artifact()
    path = tmp_path / item['path']
    path.parent.mkdir()
    path.write_bytes(b'weights')
    receipt = {}
    saved = []
    setup.ensure_artifact(item, tmp_path, receipt, lambda: saved.append(True), None)
    assert saved == [True]
    real = setup._sha256
    monkeypatch.setattr(setup, '_sha256', lambda p: pytest.fail('unnecessary rehash'))
    setup.ensure_artifact(item, tmp_path, receipt, lambda: None, None)
    monkeypatch.setattr(setup, '_sha256', real)
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Integrity'):
        setup.ensure_artifact(item, tmp_path, receipt, lambda: None, None)


def test_interrupted_download_resumes_and_checkpoints(tmp_path):
    item = artifact()
    receipt = {}
    def interrupted(name):
        path = tmp_path / name
        path.parent.mkdir()
        path.write_bytes(b'w')
        raise OSError('interrupted')
    with pytest.raises(OSError):
        setup.ensure_artifact(item, tmp_path, receipt, lambda: None, interrupted)
    assert receipt == {}
    def finish(name):
        (tmp_path / name).write_bytes(b'weights')
    state = tmp_path / 'state.json'
    setup.ensure_artifact(item, tmp_path, receipt, lambda: setup.save(state, receipt), finish)
    assert json.loads(state.read_text()) == receipt
    assert state.stat().st_mode & 0o777 == 0o600


def test_changed_during_hash_is_not_receipted(tmp_path, monkeypatch):
    item = artifact()
    path = tmp_path / item['path']
    path.parent.mkdir()
    path.write_bytes(b'weights')
    def changing(p):
        p.write_bytes(b'changed')
        return item['sha256']
    monkeypatch.setattr(setup, '_sha256', changing)
    receipt = {}
    with pytest.raises(ValueError, match='changed during'):
        setup.ensure_artifact(item, tmp_path, receipt, lambda: None, None)
    assert not receipt


def test_rejects_path_escape(tmp_path):
    item = artifact()
    item['path'] = '../outside'
    with pytest.raises(ValueError, match='escapes'):
        setup.ensure_artifact(item, tmp_path, {}, lambda: None, None)


def test_explicit_hash_bypasses_receipt(tmp_path, monkeypatch):
    item = artifact()
    path = tmp_path / item['path']
    path.parent.mkdir()
    path.write_bytes(b'weights')
    receipt = {item['path']: {'identity': setup.identity(path), 'sha256': item['sha256']}}
    calls = []
    monkeypatch.setattr(setup, '_sha256', lambda p: calls.append(p) or item['sha256'])
    setup.ensure_artifact(item, tmp_path, receipt, lambda: None, None, True)
    assert calls == [path]


def test_start_stopped_container_fails_without_waiting(tmp_path, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='exited')
    monkeypatch.setattr(setup, 'run', run)
    with pytest.raises(ValueError, match='stopped'):
        setup.start(SimpleNamespace(timeout=10), {'ready': True, 'config': {}}, tmp_path)
    assert len(calls) == 2


def test_setup_requires_explicit_image_before_side_effects(tmp_path):
    args = SimpleNamespace(image=None, build=False)
    with pytest.raises(ValueError, match='No published image'):
        setup.setup(args, {}, tmp_path / 'setup.json')


def test_cli_dispatch():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    for action in ('setup', 'start'):
        result = subprocess.run([str(root / 'r9v'), action, 'qwen38', '--dry-run'],
                                capture_output=True, text=True, check=True)
        assert json.loads(result.stdout)['command'][-1] == action


def test_setup_reuses_assets_pins_image_and_requires_final_doctor(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    model = tmp_path / 'models'
    model.mkdir()
    item = artifact()
    (model / 'target').mkdir()
    (model / item['path']).write_bytes(b'weights')
    (root / 'package.json').write_text(json.dumps({
        'artifacts': [item], 'distribution': {'repository': 'test/repo', 'revision': 'abc'}}))
    (root / 'runtime.json').write_text('{}')
    profile = root / 'profile.json'
    profile.write_text(json.dumps({'descriptors': {
        'model_package': 'package.json', 'runtime': 'runtime.json'}}))
    monkeypatch.setattr(setup, 'ROOT', root)
    monkeypatch.setattr(setup, 'PROFILE', profile)
    monkeypatch.setattr(setup, 'PLE_BYTES', 1)
    monkeypatch.setattr(setup.os, 'access', lambda *a: True)
    monkeypatch.delenv('R9V_CONFIG_FILE', raising=False)
    monkeypatch.setattr(setup, 'profile_settings', lambda: {})
    monkeypatch.setattr(setup, 'select_devices', lambda b: {
        'R9V_VISIBLE_DEVICES': '1,2', 'R9V_EXPECTED_GPU_BDFS': 'a,b'})
    calls = []
    fail_doctor = [True]
    def run(command, **kwargs):
        calls.append([str(c) for c in command])
        if str(command[0]).endswith('profile-doctor.sh') and len(command) == 1 and fail_doctor[0]:
            raise ValueError('doctor failed')
        return SimpleNamespace(stdout='sha256:resolved-image\n')
    monkeypatch.setattr(setup, 'run', run)
    args = SimpleNamespace(image='local:test', local_image=True, build=False,
                           accept_model_license=True, gpu_bdfs=None,
                           model_dir=str(model), data_dir=None, ple_path=None, hash=False)
    state = {}
    state_path = tmp_path / 'setup.json'
    with pytest.raises(ValueError, match='doctor failed'):
        setup.setup(args, state, state_path)
    assert json.loads(state_path.read_text())['ready'] is False
    assert state['config']['R9V_IMAGE'] == 'sha256:resolved-image'
    monkeypatch.setattr(setup, '_sha256', lambda p: pytest.fail('rehash on retry'))
    fail_doctor[0] = False
    setup.setup(args, state, state_path)
    assert json.loads(state_path.read_text())['ready'] is True
    assert not any(c[:2] == ['docker', 'pull'] or c[0] == 'hf' for c in calls)
    extraction = next(c for c in calls if c[:2] == ['docker', 'run'])
    assert 'sha256:resolved-image' in extraction
    assert '/models/target/model.gguf' in extraction


def test_cli_interrupted_download_resumes_with_persistent_receipts(tmp_path):
    """Exercise separate setup processes with tiny assets and fake external services."""
    import os
    import subprocess
    import sys
    import textwrap

    root = tmp_path / 'fixture'
    (root / 'scripts').mkdir(parents=True)
    (root / 'tools').mkdir()
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    artifacts = [dict(artifact(payload), path=name) for name, payload in
                 [('first.txt', b'first'), ('target/model.gguf', b'weights')]]
    (root / 'package.json').write_text(json.dumps({'artifacts': artifacts,
        'distribution': {'repository': 'fixture/model', 'revision': 'immutable'}}))
    (root / 'runtime.json').write_text('{}')
    (root / 'profile.json').write_text(json.dumps({'descriptors': {
        'model_package': 'package.json', 'runtime': 'runtime.json'}}))
    doctor = root / 'scripts/profile-doctor.sh'
    doctor.write_text('#!/bin/sh\nexit 0\n')
    doctor.chmod(0o755)
    hf = binaries / 'hf'
    hf.write_text('#!' + sys.executable + '\n' + textwrap.dedent('''
        import os, sys
        from pathlib import Path
        name = sys.argv[3]
        root = Path(sys.argv[sys.argv.index('--local-dir') + 1])
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(os.environ['TEST_CALLS'], 'a') as stream: stream.write(name + '\\n')
        if name == 'first.txt': path.write_bytes(b'first')
        elif not (root / 'interrupted').exists():
            path.write_bytes(b'w')
            (root / 'interrupted').touch()
            raise SystemExit(17)
        else: path.write_bytes(b'weights')
    '''))
    hf.chmod(0o755)
    docker = binaries / 'docker'
    docker.write_text('#!' + sys.executable + '\n' + textwrap.dedent('''
        import sys
        from pathlib import Path
        if sys.argv[1] == 'info': print('/fixture/docker')
        elif sys.argv[1:3] == ['image', 'inspect']: print('sha256:fixture')
        elif sys.argv[1] == 'run':
            volume = next(x for x in sys.argv if x.endswith(':/r9v-data'))
            (Path(volume[:-10]) / 'per_layer_token_embd.iq4_nl.bin').write_bytes(b'p')
        else: raise SystemExit('unexpected Docker call')
    '''))
    docker.chmod(0o755)
    bootstrap = tmp_path / 'bootstrap.py'
    bootstrap.write_text(textwrap.dedent('''
        import os, sys
        from pathlib import Path
        sys.path.insert(0, os.environ['TEST_SOURCE'])
        from tools import setup_profile as s
        s.ROOT = Path(os.environ['TEST_ROOT'])
        s.PROFILE = s.ROOT / 'profile.json'
        s.PLE_BYTES = 1
        s.profile_settings = lambda: {'R9V_MAX_MODEL_LEN': '4096'}
        s.select_devices = lambda b: {'R9V_VISIBLE_DEVICES': '0,1'}
        raise SystemExit(s.main())
    '''))
    state = tmp_path / 'state'
    model = tmp_path / 'model'
    env = {**os.environ, 'PATH': str(binaries) + os.pathsep + os.environ['PATH'],
           'TEST_ROOT': str(root), 'TEST_SOURCE': str(Path(__file__).resolve().parents[1]),
           'TEST_CALLS': str(tmp_path / 'calls')}
    env.pop('R9V_CONFIG_FILE', None)
    command = [sys.executable, str(bootstrap), 'setup', '--model-dir', str(model),
               '--state-dir', str(state), '--image', 'fixture:local', '--local-image',
               '--accept-model-license']
    failed = subprocess.run(command, env=env, capture_output=True, text=True)
    assert failed.returncode == 1, failed.stdout + failed.stderr
    receipt = json.loads((state / 'setup.json').read_text())
    assert not receipt['ready']
    assert set(receipt['artifacts']) == {'first.txt'}
    success = subprocess.run(command, env=env, capture_output=True, text=True)
    assert success.returncode == 0, success.stdout + success.stderr
    assert 'Reusing previously hash-verified file: first.txt' in success.stdout
    receipt = json.loads((state / 'setup.json').read_text())
    assert receipt['ready']
    assert receipt['config']['R9V_MAX_MODEL_LEN'] == '4096'
    assert (tmp_path / 'calls').read_text().splitlines() == [
        'first.txt', 'target/model.gguf', 'target/model.gguf']
    # Invalid retry arguments must preserve the successful installation.
    invalid = subprocess.run(command + ['--image=-bad'],
                             env=env, capture_output=True, text=True)
    assert invalid.returncode == 1
    assert json.loads((state / 'setup.json').read_text())['ready']
