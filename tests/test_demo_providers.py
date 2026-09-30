"""`--demo` end to end through the real CLI and LiteLLM for each provider preset. HTTP is answered by a scripted
fake provider in that provider's own wire format, so every layer except the network runs: presets, key lookup,
tool-mode probe, request/response translation, retries, the pipeline, the report. Faults are injected mid-demo.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

from harness import cli
from harness import llm as llm_mod
from harness.config import DEFAULT_CONFIG_PATH

PY = sys.executable
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
KEYS = {"gemini": "AIza_fake_demo_key", "openai": "sk-proj-fake-demo-key", "qwen": "sk-fake-demo-key",
        "groq": "gsk_fake_demo_key"}
REPRO = ("import sys\nfrom toolkit.text import slugify\nout = slugify('Hello  World!')\nprint(repr(out))\n"
         "sys.exit(0 if out == 'hello-world' else 1)\n")
REGRESSION = ("from toolkit.text import slugify\n\n\ndef test_collapses_and_strips():\n"
              "    assert slugify('Hello  World!') == 'hello-world'\n"
              "    assert slugify('  Release notes: v2.0  ') == 'release-notes-v2-0'\n")


def demo_script() -> list[tuple[str, list[tuple[str, dict]]]]:
    """(text, [(tool, arguments)]) for each model turn that solves fixtures/issues/01_slugify.md."""
    repro_cmd = f'"{PY}" @scratch/repro.py'
    return [
        (json.dumps({"title": "slugify leaves repeated and trailing hyphens", "kind": "bug",
                     "summary": "Runs of separators are not collapsed and ends are not stripped.",
                     "expected": "hello-world", "actual": "hello--world-", "error_messages": [],
                     "mentioned_paths": [], "mentioned_symbols": ["slugify"], "repro_hints": "slugify('Hello  World!')"}),
         []),
        ("", [("view_file", {"path": "toolkit/text.py"})]),
        ("", [("finish", {"files": ["toolkit/text.py"], "symbols": ["slugify"], "confidence": "high",
                          "root_cause": "slugify replaces each non-alphanumeric character separately."})]),
        ("", [("create_file", {"path": "@scratch/repro.py", "content": REPRO})]),
        ("", [("run_command", {"command": repro_cmd})]),
        ("", [("finish", {"reproduced": True, "command": repro_cmd, "observed": "'hello--world-'"})]),
        ("", [("str_replace", {"path": "toolkit/text.py",
                               "old_str": '    return re.sub(r"[^a-z0-9]", "-", s)',
                               "new_str": '    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")'})]),
        ("", [("create_file", {"path": "tests/test_slugify_regression.py", "content": REGRESSION})]),
        ("", [("finish", {"summary": "Collapse separator runs and strip hyphens at the ends.",
                          "files_changed": ["toolkit/text.py"], "tests_added": ["tests/test_slugify_regression.py"]})]),
    ]


class FakeProvider:
    """Answers LiteLLM's HTTP requests from the script, in the wire format of the endpoint that was called.

    faults maps a request number (0-based, the probe included) to a (status, body) reply or an exception to raise.
    Native tool calls are sent when the request carried tools; otherwise calls are written as ```tool blocks.
    """

    def __init__(self, script: list, faults: dict | None = None, down_from: int | None = None) -> None:
        self.script = list(script)
        self.faults = dict(faults or {})
        self.down_from = down_from  # every request from this number on fails to connect
        self.requests: list[tuple[str, dict]] = []

    def send(self, client, request: httpx.Request, **kwargs) -> httpx.Response:
        n = len(self.requests)
        body = json.loads(request.content or b"{}")
        self.requests.append((str(request.url), body))
        if self.down_from is not None and n >= self.down_from:
            raise httpx.ConnectError("[Errno 61] Connection refused", request=request)
        fault = self.faults.pop(n, None)
        if isinstance(fault, BaseException):
            raise fault
        if fault is not None:
            return httpx.Response(fault[0], json=fault[1], request=request)
        tools = json.dumps(body.get("tools") or [])
        if tools.count('"name"') == 1 and '"ping"' in tools:
            text, calls = "", [("ping", {"message": "ok"})]
        else:
            text, calls = self.script.pop(0)
        if calls and not body.get("tools"):
            text, calls = "".join(f'```tool\n{json.dumps({"name": n, "arguments": a})}\n```' for n, a in calls), []
        return httpx.Response(200, json=self._encode(request, text, calls, n), request=request)

    @staticmethod
    def _encode(request: httpx.Request, text: str, calls: list, n: int) -> dict:
        if "generativelanguage.googleapis.com" in str(request.url):
            parts = [{"functionCall": {"name": name, "args": args}, "thoughtSignature": f"sig{n}"}
                     for name, args in calls] or [{"text": text}]
            return {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
                    "usageMetadata": {"promptTokenCount": 900, "candidatesTokenCount": 60, "totalTokenCount": 960}}
        if request.url.path.endswith("/responses"):
            output = [{"type": "function_call", "id": f"fc_{n}_{i}", "call_id": f"call_{n}_{i}", "name": name,
                       "arguments": json.dumps(args), "status": "completed"} for i, (name, args) in enumerate(calls)]
            output = output or [{"type": "message", "id": f"msg_{n}", "role": "assistant", "status": "completed",
                                 "content": [{"type": "output_text", "text": text, "annotations": []}]}]
            return {"id": f"resp_{n}", "object": "response", "created_at": 1, "status": "completed", "model": "m",
                    "output": output, "usage": {"input_tokens": 900, "output_tokens": 60, "total_tokens": 960}}
        message: dict = {"role": "assistant", "content": text or None}
        if calls:
            message["tool_calls"] = [{"id": f"call_{n}_{i}", "type": "function", "function": {
                "name": name, "arguments": json.dumps(args)}} for i, (name, args) in enumerate(calls)]
        return {"id": f"c{n}", "object": "chat.completion", "created": 1, "model": "m", "service_tier": "on_demand",
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 60, "total_tokens": 960}}


@pytest.fixture()
def demo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run `--demo --model <preset>` against a FakeProvider; returns (exit code, output, run dir, provider)."""
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["output"] = {"runs_dir": str(tmp_path / "runs"), "workspaces_dir": str(tmp_path / "ws")}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setenv("COLUMNS", "200")
    for cache in (llm_mod._TOOL_MODE_CACHE, llm_mod._DROPPED_PARAMS, llm_mod._MAX_TOKENS_CAP,
                  llm_mod._PROMPT_TOKEN_CAP):
        cache.clear()
    real_make_llm = cli.make_llm

    def make_llm(cfg, ui):
        client = real_make_llm(cfg, ui)
        client._sleep = lambda s: None  # provider-hinted waits are asserted, not slept
        return client

    monkeypatch.setattr(cli, "make_llm", make_llm)

    def run(preset: str, provider: FakeProvider, capsys) -> tuple[int, str, Path | None, FakeProvider]:
        monkeypatch.setenv("AI_API_KEY", KEYS[preset])
        monkeypatch.setattr(httpx.Client, "send", lambda client, request, **kw: provider.send(client, request))
        code = cli.main(["--config", str(config), "--demo", "--model", preset, "--strict-exit"])
        out = capsys.readouterr().out
        runs = sorted((tmp_path / "runs").glob("*/")) if (tmp_path / "runs").exists() else []
        return code, out, (runs[-1] if runs else None), provider

    run.tmp = tmp_path
    return run


def demo_workspace(tmp_path: Path) -> Path:
    return sorted((tmp_path / "ws").glob("demo-*"))[-1]


def hidden_test_passes(repo: Path) -> bool:
    shutil.copy(FIXTURES / "hidden_tests" / "test_issue_01.py", repo / "tests" / "test_issue_01.py")
    res = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_issue_01.py"],
                         cwd=repo, capture_output=True, timeout=120)
    return res.returncode == 0


def trajectory(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "trajectory.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("preset, first_url", [
    ("gemini", "https://generativelanguage.googleapis.com/"),
    ("openai", "https://api.openai.com/v1/"),
    ("qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"),
    ("groq", "https://api.groq.com/openai/v1/chat/completions"),
])
def test_demo_verified_for_every_provider(demo, capsys, preset: str, first_url: str) -> None:
    started = time.monotonic()
    code, out, run_dir, provider = demo(preset, FakeProvider(demo_script()), capsys)
    assert code == 0, out[-3000:]
    assert "VERIFIED FIX" in out and "Traceback" not in out
    assert provider.script == []  # every scripted turn was used: no extra or missing model calls
    assert all(url.startswith(first_url) for url, _ in provider.requests)

    state = json.loads((run_dir / "state.json").read_text())
    assert state["status"] == "verified"
    assert state["review"]["verdict"] == "skipped"  # unambiguous verification costs no review call
    metrics = json.loads((run_dir / "metrics.json").read_text())
    probe = 0 if preset == "groq" else 1  # groq forces text mode: no probe (the probe precedes the run's metrics)
    assert metrics["total"]["llm_calls"] == len(demo_script()) == len(provider.requests) - probe
    assert (run_dir / "patch.diff").read_text().count('.strip("-")') == 1
    assert KEYS[preset] not in (run_dir / "trajectory.jsonl").read_text()
    assert hidden_test_passes(demo_workspace(demo.tmp))
    assert time.monotonic() - started < 90  # offline: no real waiting anywhere


def test_demo_tool_calls_round_trip_in_each_format(demo, capsys) -> None:
    """Native calls come back with the provider's ids (Gemini: with the thought signature)."""
    _, _, _, provider = demo("gemini", FakeProvider(demo_script()), capsys)
    replayed = [part for _, body in provider.requests for turn in body.get("contents", [])
                for part in turn.get("parts", []) if "function_call" in part]
    assert replayed and all(part.get("thoughtSignature", "").startswith("sig") for part in replayed)
    _, _, _, provider = demo("groq", FakeProvider(demo_script()), capsys)
    assert all("tools" not in body for _, body in provider.requests)  # text protocol only
    assert any("[tool_result" in m["content"] for m in provider.requests[-1][1]["messages"] if m["role"] == "user")


GEMINI_PER_MINUTE = (429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message":
                     "You exceeded your current quota. Please retry in 4.2s.", "details": [
                         {"violations": [{"quotaId": "GenerateContentInputTokensPerModelPerMinute-FreeTier"}]}]}})
OVERLOADED = (503, {"error": {"message": "The server is overloaded", "type": "server_error", "code": None}})


def test_demo_survives_rate_limits_and_overload(demo, capsys) -> None:
    faults = {3: GEMINI_PER_MINUTE, 7: GEMINI_PER_MINUTE}
    code, out, run_dir, provider = demo("gemini", FakeProvider(demo_script(), faults), capsys)
    assert code == 0 and "VERIFIED FIX" in out and "Rate limited; retrying in" in out
    retries = [e for e in trajectory(run_dir) if e["event"] == "llm_retry"]
    assert len(retries) == 2 and all(4.2 < e["delay_s"] < 5.5 and e["hinted"] for e in retries)

    code, out, _, _ = demo("openai", FakeProvider(demo_script(), {2: OVERLOADED, 3: OVERLOADED}), capsys)
    assert code == 0 and "VERIFIED FIX" in out, out[-4000:]


def test_demo_endpoint_lost_after_the_fix_still_verifies_it(demo, capsys) -> None:
    """The provider goes down just before the FIX phase's finish call: the change made is verified anyway."""
    last_turn = len(demo_script())  # probe + turns 0..n-2 answered, the final finish request fails
    code, out, run_dir, provider = demo("qwen", FakeProvider(demo_script(), down_from=last_turn), capsys)
    state = json.loads((run_dir / "state.json").read_text())
    assert state["status"] == "verified" and code == 0, state["notes"]
    assert any("model endpoint failed" in n for n in state["notes"])
    assert any("failed after the patch passed verification" in n for n in state["notes"])
    assert len(provider.requests) == last_turn + 3  # 2 retries for a dead connection, not 5
    assert hidden_test_passes(demo_workspace(demo.tmp))


def test_demo_endpoint_lost_before_any_change_reports_error(demo, capsys) -> None:
    code, out, run_dir, _ = demo("qwen", FakeProvider(demo_script(), down_from=2), capsys)
    assert code == 1 and "Traceback" not in out
    state = json.loads((run_dir / "state.json").read_text())
    assert state["status"] == "error" and "Could not reach https://dashscope-intl" in " ".join(state["notes"])
    assert not (run_dir / "patch.diff").read_text().strip()


@pytest.mark.parametrize("status, body, expected", [
    (401, {"error": {"message": "Incorrect API key provided", "type": "invalid_request_error",
                     "code": "invalid_api_key"}}, "API key rejected by openai (Incorrect API key provided)"),
    (404, {"error": {"message": "The model does not exist", "type": "invalid_request_error",
                     "code": "model_not_found"}}, "Model not found: openai/gpt-5.4-mini"),
    (403, {"error": {"message": "Country, region, or territory not supported", "type": "request_forbidden",
                     "code": "unsupported_country_region_territory"}}, "Access denied by openai"),
])
def test_demo_stops_at_once_on_access_errors(demo, capsys, status, body, expected) -> None:
    code, out, run_dir, provider = demo("openai", FakeProvider(demo_script(), {0: (status, body)}), capsys)
    assert code == 1 and expected in " ".join(out.split()) and "Traceback" not in out
    assert len(provider.requests) == 1  # no retries: waiting cannot fix a key, a model name or a region


def test_demo_with_another_providers_key_sends_nothing(demo, capsys, monkeypatch) -> None:
    provider = FakeProvider(demo_script())
    monkeypatch.setitem(KEYS, "gemini", KEYS["groq"])
    code, out, _, _ = demo("gemini", provider, capsys)
    assert code == 1 and "looks like a Groq key" in out and "MODEL=groq" in out
    assert provider.requests == []
