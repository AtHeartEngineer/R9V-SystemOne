from tools.host_pressure import exhausted, normal_zones


def test_memavailable_cannot_hide_exhausted_normal_zone():
    text = """Node 0, zone      DMA
  pages free     10000
        min      5
Node 0, zone    Normal
  pages free     8899
        min      16614
        low      414239
        high     811864
"""
    zones = normal_zones(text, 4096)
    assert len(zones) == 1
    assert zones[0]["free_bytes"] == 35596 * 1024
    assert exhausted(
        {"meminfo": {"MemAvailable_bytes": 65 * 2**30}, "normal_zones": zones}
    )


def test_missing_or_usable_normal_zone_is_not_exhaustion():
    assert not exhausted({})
    assert not exhausted({"normal_zones": [{"free_bytes": 100, "min_bytes": 50}]})
    assert not exhausted({"normal_zones": [{"node": 0}]})


def test_monitor_aborts_on_persistent_zone_exhaustion_with_high_available_ram(
    tmp_path, monkeypatch
):
    import json
    from types import SimpleNamespace

    from tools import qualify_runtime as qualification

    readings = iter([30, 100, 30, 30])
    states = []
    session = qualification.Session(
        {
            "config": {"R9V_IMAGE": "unused", "R9V_EXPECTED_GPU_BDFS": "a,b"},
            "prompt": "unused",
            "arms": [{"name": "a"}],
        },
        tmp_path,
    )
    monkeypatch.setattr(
        qualification,
        "memory_snapshot",
        lambda _: {
            "host_available_bytes": 65 * 2**30,
            "host_pressure": {
                "normal_zones": [{"free_bytes": next(readings), "min_bytes": 50}]
            },
        },
    )

    def wait(_):
        states.append(session.failure)

    session.stop = SimpleNamespace(is_set=lambda: len(states) == 4, wait=wait)
    session.reporter = SimpleNamespace(send=lambda *args, **kwargs: None)
    session.monitor()
    assert states[:3] == [None, None, None]
    assert "minimum watermark" in states[3]
    evidence = [json.loads(line) for line in (tmp_path / "memory.jsonl").read_text().splitlines()]
    assert len(evidence) == 4
    assert evidence[-1]["host_pressure"]["normal_zones"][0]["free_bytes"] == 30
