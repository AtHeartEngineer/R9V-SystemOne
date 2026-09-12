# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest

from tools.profile_state import default_state_dir, validate_state_profile
from tools import setup_profile

IQ4 = 'qwen38-flash-next/ud-iq4-xs/dual-r9700-128k'
Q4 = 'qwen38-flash-next/ud-q4-k-xl/dual-r9700-128k'


def test_quants_have_distinct_state_and_legacy_iq4_path(tmp_path):
    env = {'XDG_STATE_HOME': str(tmp_path)}
    assert default_state_dir(IQ4, env) == tmp_path / 'r9v/qwen38'
    assert default_state_dir(Q4, env) != default_state_dir(IQ4, env)
    assert default_state_dir('a/b', env) != default_state_dir('a-b', env)


def test_wrong_quant_state_is_rejected_including_legacy():
    for state in ({'ready': True}, {'profile_id': IQ4}, {'config': {'R9V_PROFILE_ID': IQ4}}):
        with pytest.raises(ValueError, match='belongs to'):
            validate_state_profile(state, Q4)
        validate_state_profile(state, IQ4)
    validate_state_profile({}, Q4)


def test_setup_uses_selected_descriptor_not_iq4_default(tmp_path, monkeypatch):
    root = tmp_path / 'profiles/q4'
    root.mkdir(parents=True)
    (root / 'profile.json').write_text(json.dumps({'id': Q4}))
    monkeypatch.setattr(setup_profile, 'ROOT', tmp_path)
    monkeypatch.setenv('R9V_PROFILE_ROOT', str(root))
    monkeypatch.setenv('R9V_PROFILE_ID', Q4)
    assert setup_profile.selected_profile()[1]['id'] == Q4
    monkeypatch.setenv('R9V_PROFILE_ID', IQ4)
    with pytest.raises(ValueError, match='does not match'):
        setup_profile.selected_profile()


def test_saved_headroom_cannot_silently_bypass_planning(tmp_path, monkeypatch):
    from tools import prepare_placement
    monkeypatch.setattr(prepare_placement, 'apply', lambda *a: False)
    monkeypatch.setattr(setup_profile, 'run', lambda *a, **k: pytest.fail('must not launch'))
    state = {'ready': True, 'config': {'R9V_HEADROOM_SELECTION': '1',
             'R9V_MIN_FREE_VRAM_GIB_BY_RANK': '6,4'}}
    with pytest.raises(ValueError, match='requires a qualified'):
        setup_profile.start(SimpleNamespace(state_dir=tmp_path), state, tmp_path / 'setup.json')
