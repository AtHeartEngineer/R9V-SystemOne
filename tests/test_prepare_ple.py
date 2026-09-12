from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).parents[1] / "tools" / "prepare_ple.py"
SPEC = importlib.util.spec_from_file_location("prepare_ple", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare_ple = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare_ple
SPEC.loader.exec_module(prepare_ple)


def test_sample_offsets_cover_start_middle_and_end() -> None:
    size = 64 * 1024
    offsets = prepare_ple.sample_offsets(size)
    assert offsets == (0, size // 2 - 2048, size - 4096)


def test_validate_samples_uses_tensor_subrange(tmp_path: Path) -> None:
    prefix = b"prefix" * 1000
    payload = bytes(index % 251 for index in range(64 * 1024))
    suffix = b"suffix" * 1000
    source = tmp_path / "source.gguf"
    target = tmp_path / "ple.bin"
    source.write_bytes(prefix + payload + suffix)
    target.write_bytes(payload)

    span = prepare_ple.TensorSpan(
        source=str(source),
        tensor_name="per_layer_token_embd.weight",
        data_offset=len(prefix),
        packed_bytes=len(payload),
        tensor_type="IQ4_NL",
        gguf_shape=(160, 728),
    )
    assert prepare_ple.validate_samples(span, target)

    damaged = bytearray(payload)
    damaged[-1] ^= 0xFF
    target.write_bytes(damaged)
    assert not prepare_ple.validate_samples(span, target)


def test_parse_shape() -> None:
    assert prepare_ple.parse_shape("160,320001536") == (160, 320001536)


def test_prepare_reuses_payload_without_destroying_hash_provenance(tmp_path, monkeypatch):
    import hashlib
    import json

    payload = bytes(index % 251 for index in range(64 * 1024))
    source = tmp_path / 'source.gguf'
    source.write_bytes(b'header' + payload)
    output = tmp_path / 'ple.bin'
    span = prepare_ple.TensorSpan(str(source), 'per_layer_token_embd.weight', 6,
                                 len(payload), 'IQ4_NL', (160, 728))
    monkeypatch.setattr(prepare_ple, 'locate_tensor', lambda *args: span)
    monkeypatch.setattr(sys, 'argv', ['prepare_ple.py', str(source), '--output', str(output),
        '--expected-bytes', str(len(payload)), '--expected-shape', '160,728'])
    assert prepare_ple.main() == 0
    manifest = output.with_name(output.name + '.manifest.json')
    original = manifest.read_bytes()
    assert json.loads(original)['payload_sha256'] == hashlib.sha256(payload).hexdigest()
    before = output.stat().st_mtime_ns
    assert prepare_ple.main() == 0
    assert output.stat().st_mtime_ns == before
    assert manifest.read_bytes() == original
