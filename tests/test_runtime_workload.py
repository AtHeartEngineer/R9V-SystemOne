import json

import pytest

from tools import runtime_workload as workload


@pytest.mark.parametrize("bad_vision", [False, True])
def test_workload_keeps_answer_observations_separate_from_runtime_health(
    tmp_path, monkeypatch, bad_vision
):
    monkeypatch.setattr(workload.time, "sleep", lambda _: None)

    def call(url, endpoint, body, timeout=1200):
        content = body["messages"][0]["content"]
        count = (
            content.count("The following passage") * 10 + 37
            if isinstance(content, str)
            else 100
        )
        if endpoint == "/tokenize":
            return {"count": count}
        message = {"content": "42"}
        if isinstance(content, list):
            message["content"] = "blue" if bad_vision else "red"
        elif content.startswith("Remember"):
            message["content"] = "R9V-731"
        elif "tools" in body:
            message = {
                "tool_calls": [
                    {
                        "function": {
                            "name": "record_code",
                            "arguments": '{"code":"R9V-731"}',
                        }
                    }
                ]
            }
        return {
            "choices": [{"finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": count},
        }

    monkeypatch.setattr(workload, "call", call)
    output = tmp_path / "run"
    if bad_vision:
        workload.run("fixture", "model", output, 4096)
        result = json.loads((output / "result.json").read_text())
        assert result["passed"]
        assert "vision color mismatch" in result["semantic_observations"]
        assert result["runtime_failures"] == []
        assert "minimum_physical_free_bytes" in result
        assert "context" in result["checks"]
        assert "idle_resume" in result["checks"]
        assert len(result["semantic_observations"]) == 3
    else:
        workload.run("fixture", "model", output, 4096)
        assert json.loads((output / "result.json").read_text())["passed"]
        assert (
            json.loads((output / "context.json").read_text())["response"]["usage"][
                "prompt_tokens"
            ]
            == 3967
        )


def test_workload_rejects_missing_context_envelope(tmp_path, monkeypatch):
    monkeypatch.setattr(workload.time, "sleep", lambda _: None)

    def call(url, endpoint, body, timeout=1200):
        content = body["messages"][0]["content"]
        if endpoint == "/tokenize":
            return {"count": content.count("The following passage") * 10 + 37}
        if "tools" in body:
            message = {"tool_calls": [{"function": {
                "name": "record_code", "arguments": '{"code":"R9V-731"}'
            }}]}
        elif isinstance(content, list):
            message = {"content": "red"}
        elif content.startswith("Remember"):
            message = {"content": "wrong"}
        else:
            message = {"content": "42"}
        return {"choices": [{"finish_reason": "stop", "message": message}],
                "usage": {"prompt_tokens": 1}}

    monkeypatch.setattr(workload, "call", call)
    with pytest.raises(ValueError, match="advertised envelope"):
        workload.run("fixture", "model", tmp_path / "run", 4096)
    result = json.loads((tmp_path / "run" / "result.json").read_text())
    assert not result["passed"]
    assert any("advertised envelope" in item for item in result["runtime_failures"])
