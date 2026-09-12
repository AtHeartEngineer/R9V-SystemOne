# SPDX-License-Identifier: Apache-2.0
import pytest
from tools.package_sources import artifact_source


def test_auxiliary_artifacts_keep_their_own_immutable_upstream():
    package = {'distribution': {'repository': 'unsloth/model', 'revision': 'a' * 40}}
    target = {'path': 'UD-Q4_K_XL/shard.gguf'}
    auxiliary = {'path': 'mtp/model.safetensors', 'distribution': {'repository': 'Dyluhn/package', 'revision': 'b' * 40}}
    assert artifact_source(package, target) == ('unsloth/model', 'a' * 40, target['path'])
    assert artifact_source(package, auxiliary) == ('Dyluhn/package', 'b' * 40, auxiliary['path'])


@pytest.mark.parametrize('relative,revision', [('../escape', 'a' * 40), ('/escape', 'a' * 40), ('ok', 'main')])
def test_source_rejects_escaped_paths_and_mutable_revisions(relative, revision):
    with pytest.raises(ValueError):
        artifact_source({'distribution': {'repository': 'owner/model', 'revision': revision}}, {'path': relative})
