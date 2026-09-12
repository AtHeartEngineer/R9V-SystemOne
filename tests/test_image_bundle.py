import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
import image_bundle


def _manifest(tmp_path, data=b"abcdef", **overrides):
    digest = hashlib.sha256(data).hexdigest()
    value = {
        "schema": image_bundle.SCHEMA,
        "format": image_bundle.FORMAT,
        "image_ids": ["sha256:" + "a" * 64],
        "parts": [
            {
                "name": "part-000.gz",
                "bytes": len(data),
                "sha256": digest,
                "url": "https://github.com/acme/r9v/releases/download/r1/part-000.gz",
            }
        ],
        "bytes": len(data),
        "sha256": digest,
    }
    value.update(overrides)
    return value, data


def test_manifest_rejects_path_escape_duplicate_and_bool_size(tmp_path):
    value, _ = _manifest(tmp_path)
    for bad in ("../x", "part/a", "part-000.gz"):
        candidate = json.loads(json.dumps(value))
        candidate["parts"][0]["name"] = bad
        if bad == "part-000.gz":
            candidate["parts"].append(dict(candidate["parts"][0]))
        with pytest.raises(image_bundle.ImageBundleError):
            image_bundle.validate_manifest(candidate)
    candidate = json.loads(json.dumps(value))
    candidate["parts"][0]["bytes"] = True
    with pytest.raises(image_bundle.ImageBundleError):
        image_bundle.validate_manifest(candidate)


def test_manifest_rejects_unpinned_or_authenticated_url(tmp_path):
    value, _ = _manifest(tmp_path)
    for url in (
        "http://github.com/a/b/releases/download/r/p",
        "https://u:p@github.com/a/b/releases/download/r/p",
        "https://github.com/a/b/archive/r/p",
    ):
        value["parts"][0]["url"] = url
        with pytest.raises(image_bundle.ImageBundleError):
            image_bundle.validate_manifest(value)


def test_download_resume_and_hash_before_rename(tmp_path, monkeypatch):
    value, data = _manifest(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "part-000.gz.part").write_bytes(data[:2])

    class Response:
        status = 206
        headers = {"Content-Range": "bytes 2-5/6"}

        def read(self, n):
            chunk, self.body = self.body, b""
            return chunk

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    response = Response()
    response.body = data[2:]
    monkeypatch.setattr(
        image_bundle.urllib.request, "urlopen", lambda request, timeout=60: response
    )
    image_bundle.ensure_parts(value, cache)
    assert (cache / "part-000.gz").read_bytes() == data
    assert not (cache / "part-000.gz.part").exists()

    (cache / "part-000.gz").write_bytes(b"bad")
    with pytest.raises(image_bundle.ImageBundleError):
        image_bundle.verify_parts(value, cache)


def test_complete_partial_is_finalized_or_redownloaded(tmp_path, monkeypatch):
    value, data = _manifest(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    partial = cache / "part-000.gz.part"
    partial.write_bytes(data)

    def unexpected(*args, **kwargs):
        raise AssertionError("complete valid partial must not request HTTP")

    monkeypatch.setattr(image_bundle.urllib.request, "urlopen", unexpected)
    image_bundle.ensure_parts(value, cache)
    assert (cache / "part-000.gz").read_bytes() == data
    partial.write_bytes(b"x" * len(data))

    class Response:
        status = 200
        headers = {}

        def read(self, n):
            chunk, self.body = self.body, b""
            return chunk

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    response = Response()
    response.body = data
    monkeypatch.setattr(
        image_bundle.urllib.request, "urlopen", lambda *a, **k: response
    )
    image_bundle.ensure_parts(value, cache, force_download=True)
    assert (cache / "part-000.gz").read_bytes() == data


def test_download_rejects_truncated_or_oversize_response(tmp_path, monkeypatch):
    value, _ = _manifest(tmp_path, data=b"abcdef")
    cache = tmp_path / "cache"

    class Response:
        status = 200
        headers = {}

        def __init__(self, body):
            self.body = body

        def read(self, n):
            chunk, self.body = self.body, b""
            return chunk

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        image_bundle.urllib.request,
        "urlopen",
        lambda *args, **kwargs: Response(b"short"),
    )
    with pytest.raises(image_bundle.ImageBundleError, match="before"):
        image_bundle.ensure_parts(value, cache)
    assert not (cache / "part-000.gz").exists()
    monkeypatch.setattr(
        image_bundle.urllib.request,
        "urlopen",
        lambda *args, **kwargs: Response(b"toolong!"),
    )
    with pytest.raises(image_bundle.ImageBundleError, match="exceeded"):
        image_bundle.ensure_parts(value, cache)
    assert (cache / "part-000.gz.part").stat().st_size <= value["parts"][0]["bytes"]


def test_download_rejects_wrong_resume_range_and_cache_symlink(tmp_path, monkeypatch):
    value, data = _manifest(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "part-000.gz.part").write_bytes(data[:2])

    class Response:
        status = 206
        headers = {"Content-Range": "bytes 0-3/6"}

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        image_bundle.urllib.request, "urlopen", lambda *args, **kwargs: Response()
    )
    with pytest.raises(image_bundle.ImageBundleError, match="Content-Range"):
        image_bundle.ensure_parts(value, cache)
    (cache / "part-000.gz.part").unlink()
    (cache / "part-000.gz").symlink_to(tmp_path / "outside")
    with pytest.raises(image_bundle.ImageBundleError, match="regular file"):
        image_bundle.ensure_parts(value, cache)


def test_load_streams_verified_parts_and_returns_exact_id(tmp_path):
    value, data = _manifest(tmp_path, data=b"docker-save-payload")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "part-000.gz").write_bytes(data)
    script = tmp_path / "docker"
    script.write_text("""#!/bin/sh
if [ \"$1 $2\" = \"image load\" ]; then cat > \"$FAKE_CAPTURE\"; exit 0; fi
if [ \"$1 $2\" = \"image inspect\" ]; then echo \"$FAKE_ID\"; exit 0; fi
exit 1
""")
    script.chmod(0o755)
    expected = value["image_ids"][0]
    import os

    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "FAKE_CAPTURE": str(tmp_path / "capture"),
        "FAKE_ID": expected,
    }
    result = subprocess.run(
        [str(script), "image", "load"], env=env, input=data, capture_output=True
    )
    assert result.returncode == 0
    # Exercise the real loader with the fake executable and inspect response.
    old = os.environ.copy()
    os.environ.update(env)
    try:
        assert (
            image_bundle.load_bundle(value, cache, docker=(str(script),), timeout=5)
            == expected
        )
    finally:
        os.environ.clear()
        os.environ.update(old)
    assert (tmp_path / "capture").read_bytes() == data


def test_load_rejects_wrong_image_and_timeout(tmp_path):
    value, data = _manifest(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "part-000.gz").write_bytes(data)
    script = tmp_path / "docker"
    script.write_text(
        '#!/bin/sh\nif [ "$2" = load ]; then sleep 2; else echo sha256-'
        + "b" * 64
        + "; fi\n"
    )
    script.chmod(0o755)
    with pytest.raises(image_bundle.ImageBundleError, match="timed out"):
        image_bundle.load_bundle(value, cache, docker=(str(script),), timeout=0.01)


def test_load_rejects_wrong_image_and_docker_failure(tmp_path):
    value, data = _manifest(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "part-000.gz").write_bytes(data)
    script = tmp_path / "docker"
    script.write_text(
        '#!/bin/sh\nif [ "$2" = load ]; then cat >/dev/null; exit ${FAIL_LOAD:-0}; fi\necho sha256-'
        + "b" * 64
        + "\n"
    )
    script.chmod(0o755)
    with pytest.raises(image_bundle.ImageBundleError, match="mismatch"):
        image_bundle.load_bundle(value, cache, docker=(str(script),), timeout=5)
    import os

    old = os.environ.get("FAIL_LOAD")
    os.environ["FAIL_LOAD"] = "7"
    try:
        with pytest.raises(image_bundle.ImageBundleError, match="exit code 7"):
            image_bundle.load_bundle(value, cache, docker=(str(script),), timeout=5)
    finally:
        if old is None:
            os.environ.pop("FAIL_LOAD", None)
        else:
            os.environ["FAIL_LOAD"] = old


def test_load_timeout_bounds_child_that_never_reads(tmp_path):
    value, data = _manifest(tmp_path, data=b"x" * 2_000_000)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "part-000.gz").write_bytes(data)
    script = tmp_path / "docker"
    script.write_text(
        '#!/bin/sh\nif [ "$2" = load ]; then sleep 60; else echo sha256:'
        + "a" * 64
        + "; fi\n"
    )
    script.chmod(0o755)
    with pytest.raises(image_bundle.ImageBundleError, match="timed out"):
        image_bundle.load_bundle(value, cache, docker=(str(script),), timeout=0.05)
