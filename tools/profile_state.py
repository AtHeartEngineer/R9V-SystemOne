# SPDX-License-Identifier: Apache-2.0
"""Resolve installation state without sharing configurations across model quants."""

import os
import re
from pathlib import Path


def state_key(profile_id):
    if profile_id == 'qwen38-flash-next/ud-iq4-xs/dual-r9700-128k':
        return 'qwen38'  # Preserve existing installations.
    if not isinstance(profile_id, str) or not re.fullmatch(r'[a-z0-9][a-z0-9/_-]*', profile_id):
        raise ValueError('Invalid profile identity for installation state')
    # Keep separators as directory boundaries; flattening would alias distinct IDs.
    if any(part in {'', '.', '..'} for part in profile_id.split('/')):
        raise ValueError('Invalid profile identity for installation state')
    return 'profiles/' + profile_id


def default_state_dir(profile_id, env=None):
    env = os.environ if env is None else env
    if env.get('R9V_STATE_DIR'):
        return Path(env['R9V_STATE_DIR']).expanduser()
    base = Path(env.get('XDG_STATE_HOME', str(Path.home() / '.local/state')))
    return base / 'r9v' / state_key(profile_id)


def validate_state_profile(state, profile_id):
    saved = state.get('profile_id') or state.get('config', {}).get('R9V_PROFILE_ID')
    if saved is None and state:
        # Only the original profile had legacy state with no identity.
        saved = 'qwen38-flash-next/ud-iq4-xs/dual-r9700-128k'
    if saved is not None and saved != profile_id:
        raise ValueError(f'Saved setup belongs to {saved}, requested {profile_id}; select its state directory or set up this profile in a new directory')
