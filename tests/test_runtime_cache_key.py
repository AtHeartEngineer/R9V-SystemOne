from tools.runtime_cache_key import cache_key


def test_compile_cache_separates_image_placement_driver_and_workload():
    key = cache_key(
        "image-a", b"placement-a", {"R9V_MAX_MODEL_LEN": "4096"}, "driver-a"
    )
    assert key != cache_key(
        "image-b", b"placement-a", {"R9V_MAX_MODEL_LEN": "4096"}, "driver-a"
    )
    assert key != cache_key(
        "image-a", b"placement-b", {"R9V_MAX_MODEL_LEN": "4096"}, "driver-a"
    )
    assert key != cache_key(
        "image-a", b"placement-a", {"R9V_MAX_MODEL_LEN": "8192"}, "driver-a"
    )
    assert key != cache_key(
        "image-a", b"placement-a", {"R9V_MAX_MODEL_LEN": "4096"}, "driver-b"
    )
    assert key == cache_key(
        "image-a",
        b"placement-a",
        {"R9V_MAX_MODEL_LEN": "4096", "R9V_CONTAINER_NAME": "new-name"},
        "driver-a",
    )
