"""Tests for LLMClient (litellm.completion is monkeypatched; no network) and FakeLLM."""
import json
from types import SimpleNamespace

import litellm
import pytest

from harness import llm as llm_mod
from harness.config import load_config
from harness.events import Trajectory
from harness.llm import LLMClient
from harness.llm_fake import FakeLLM
from harness.types import (
    BudgetExceeded,
    ContextOverflow,
    FatalLLMError,
    LLMResponse,
    Metrics,
    ToolCall,
    Usage,
)

FAKE_KEY = "fake-key-for-tests-only"


def fake_response(content=None, tool_calls=None, finish_reason="stop", prompt=5, completion=2):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
                           usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion))


def native_call(name, arguments, call_id="call_1"):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


class Recorder:
    """Stand-in for litellm.completion: returns or raises scripted items, records kwargs."""

    def __init__(self, *items):
        self.items = list(items)
        self.kwargs = []

    def __call__(self, **kwargs):
        self.kwargs.append(kwargs)
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    for name in ("HARNESS_MODEL", "HARNESS_API_BASE", "HARNESS_TOOL_MODE"):
        monkeypatch.delenv(name, raising=False)
    llm_mod._TOOL_MODE_CACHE.clear()


def make_client(monkeypatch, recorder, tool_mode="native", tmp_path=None):
    monkeypatch.setattr(litellm, "completion", recorder)
    cfg = load_config()
    cfg.model.tool_mode = tool_mode
    traj = Trajectory(tmp_path / "t.jsonl" if tmp_path else None)
    client = LLMClient(cfg, Metrics(), traj)
    client._sleep = lambda s: None
    return client


def test_native_tool_call_parsed(monkeypatch, tmp_path):
    rec = Recorder(fake_response(tool_calls=[native_call("view_file", '{"path": "a.py", "start_line": 2}')]))
    client = make_client(monkeypatch, rec, tmp_path=tmp_path)
    tools = [llm_mod.PING_TOOL]
    resp = client.complete([{"role": "user", "content": "hi"}], tools, "localize")
    assert resp.tool_calls == [ToolCall("call_1", "view_file", {"path": "a.py", "start_line": 2})]
    assert resp.text == "" and resp.finish_reason == "stop"
    kw = rec.kwargs[0]
    assert kw["tools"] == tools and kw["tool_choice"] == "auto"
    assert kw["api_key"] == FAKE_KEY and kw["temperature"] == 0.0 and kw["seed"] == 42
    assert client.metrics.total_tokens == 7 and client.metrics.per_phase["localize"].llm_calls == 1
    log = (tmp_path / "t.jsonl").read_text()
    assert FAKE_KEY not in log
    assert json.loads(log.splitlines()[-1])["prompt_tokens"] == 5


def test_bad_json_arguments_flagged(monkeypatch):
    rec = Recorder(fake_response(tool_calls=[native_call("view_file", '{"path": "a.py",')]))
    resp = make_client(monkeypatch, rec).complete([], [llm_mod.PING_TOOL], "fix")
    call = resp.tool_calls[0]
    assert call.name == "view_file" and call.arguments == {} and call.parse_error


def test_retry_then_success(monkeypatch):
    err = litellm.RateLimitError(message="slow down", llm_provider="openai", model="m")
    rec = Recorder(err, fake_response(content="hello"))
    client = make_client(monkeypatch, rec)
    delays = []
    client._sleep = delays.append
    resp = client.complete([{"role": "user", "content": "x"}], None, "fix")
    assert resp.text == "hello" and len(rec.kwargs) == 2
    assert len(delays) == 1 and 1 <= delays[0] <= 2
    assert client.metrics.llm_calls == 1


def test_retries_exhausted_is_fatal(monkeypatch):
    errs = [litellm.APIConnectionError(message="down", llm_provider="openai", model="m") for _ in range(10)]
    client = make_client(monkeypatch, Recorder(*errs))
    client.cfg.model.max_retries = 2
    with pytest.raises(FatalLLMError, match="after 2 retries"):
        client.complete([], None, "fix")


def test_budget_exceeded(monkeypatch):
    rec = Recorder()
    client = make_client(monkeypatch, rec)
    client.metrics.add_llm("fix", Usage(client.cfg.budgets.max_total_tokens, 0))
    with pytest.raises(BudgetExceeded):
        client.complete([], None, "fix")
    assert rec.kwargs == []  # checked before calling the model

    client2 = make_client(monkeypatch, Recorder())
    client2.cfg.budgets.max_llm_calls = 1
    client2.metrics.add_llm("fix", Usage(1, 1))
    with pytest.raises(BudgetExceeded, match="call budget"):
        client2.complete([], None, "fix")


def test_error_mapping(monkeypatch):
    ctx = litellm.ContextWindowExceededError(message="too long", model="m", llm_provider="openai")
    with pytest.raises(ContextOverflow):
        make_client(monkeypatch, Recorder(ctx)).complete([], None, "fix")
    auth = litellm.AuthenticationError(message="bad key", llm_provider="openai", model="m")
    with pytest.raises(FatalLLMError, match="check AI_API_KEY"):
        make_client(monkeypatch, Recorder(auth)).complete([], None, "fix")
    bad = litellm.BadRequestError(message="weird param", model="m", llm_provider="openai")
    with pytest.raises(FatalLLMError, match="weird param"):
        make_client(monkeypatch, Recorder(bad)).complete([], None, "fix")


def test_text_mode_parses_tool_block(monkeypatch):
    text = 'Looking.\n```tool\n{"name": "view_file", "arguments": {"path": "b.py"}}\n```'
    rec = Recorder(fake_response(content=text))
    resp = make_client(monkeypatch, rec, tool_mode="text").complete([], [llm_mod.PING_TOOL], "fix")
    assert "tools" not in rec.kwargs[0]  # tools are never sent natively in text mode
    assert resp.tool_calls[0].name == "view_file" and resp.tool_calls[0].arguments == {"path": "b.py"}


def test_probe_native_and_cached(monkeypatch):
    rec = Recorder(fake_response(tool_calls=[native_call("ping", '{"message": "ok"}')]),
                   fake_response(content="done"))
    client = make_client(monkeypatch, rec, tool_mode="auto")
    assert client.tool_mode == "native"
    assert rec.kwargs[0]["tools"][0]["function"]["name"] == "ping"
    client.complete([], None, "fix")
    other = LLMClient(client.cfg, Metrics(), Trajectory(None))
    assert other.tool_mode == "native" and len(rec.kwargs) == 2  # no second probe


def test_probe_text_when_no_tool_call(monkeypatch):
    client = make_client(monkeypatch, Recorder(fake_response(content="ok")), tool_mode="auto")
    assert client.probe_tool_mode() == "text"


def test_probe_text_on_bad_request(monkeypatch):
    bad = litellm.BadRequestError(message="tools not supported", model="m", llm_provider="openai")
    client = make_client(monkeypatch, Recorder(bad), tool_mode="auto")
    assert client.tool_mode == "text"


def test_probe_auth_error_is_fatal(monkeypatch):
    auth = litellm.AuthenticationError(message="bad key", llm_provider="openai", model="m")
    client = make_client(monkeypatch, Recorder(auth), tool_mode="auto")
    with pytest.raises(FatalLLMError):
        client.probe_tool_mode()


def test_fake_llm_records_and_updates_metrics():
    metrics = Metrics()
    first = LLMResponse("a", [], Usage(3, 1))
    fake = FakeLLM([first, lambda msgs: LLMResponse(f"saw {len(msgs)}", [], Usage(2, 2))], metrics=metrics)
    assert fake.probe_tool_mode() == "native"
    msgs = [{"role": "user", "content": "x"}]
    assert fake.complete(msgs, None, "p").text == "a"
    msgs.append({"role": "assistant", "content": "a"})
    assert fake.complete(msgs, None, "p").text == "saw 2"
    assert len(fake.calls) == 2 and len(fake.calls[0]) == 1  # recorded copies, not live references
    assert metrics.total_tokens == 8 and metrics.llm_calls == 2
    with pytest.raises(AssertionError, match="script exhausted"):
        fake.complete(msgs, None, "p")


def test_fake_llm_budget():
    cfg = load_config()
    cfg.budgets.max_llm_calls = 1
    fake = FakeLLM([LLMResponse("a", [], Usage(1, 1)), LLMResponse("b", [], Usage(1, 1))], cfg=cfg)
    fake.complete([], None, "p")
    with pytest.raises(BudgetExceeded):
        fake.complete([], None, "p")


# ---------------------------------------------------------------- Phase M1: authoritative probe
@pytest.mark.parametrize("reply, reason", [
    (fake_response(content="Sure, ok!"), "no tool call"),
    (fake_response(tool_calls=[native_call("ping", '{"message": "ok"')]), "not a JSON object"),
    (fake_response(tool_calls=[native_call("ping", '"ok"')]), "not a JSON object"),
    (fake_response(tool_calls=[native_call("pong", '{"message": "ok"}')]), "unknown tool name"),
])
def test_probe_text_triggers(monkeypatch, reply, reason):
    client = make_client(monkeypatch, Recorder(reply), tool_mode="auto")
    assert client.tool_mode == "text"
    assert reason in client.probe_info["reason"]


def test_probe_native_records_raw_reply(monkeypatch):
    rec = Recorder(fake_response(tool_calls=[native_call("ping", '{"message": "ok"}')]))
    client = make_client(monkeypatch, rec, tool_mode="auto")
    assert client.probe_tool_mode() == "native"
    assert "ping" in client.probe_info["raw"] and client.metrics.per_phase["probe"].llm_calls == 1


def test_force_text_mode_skips_probe(monkeypatch):
    rec = Recorder()  # any model call would fail: the script is empty
    client = make_client(monkeypatch, rec, tool_mode="auto")
    client.cfg.model.name = "dashscope/Qwen-Max"
    client.cfg.model.force_text_mode_for = ["qwen", "deepseek-reasoner"]
    assert client.tool_mode == "text" and rec.kwargs == []
    assert "qwen" in client.probe_info["reason"]


def test_force_text_mode_no_match_still_probes(monkeypatch):
    rec = Recorder(fake_response(tool_calls=[native_call("ping", '{"message": "ok"}')]))
    client = make_client(monkeypatch, rec, tool_mode="auto")
    client.cfg.model.name = "deepseek/deepseek-chat"
    client.cfg.model.force_text_mode_for = ["qwen", "deepseek-reasoner"]
    assert client.tool_mode == "native" and len(rec.kwargs) == 1


def test_ping_verbose_output(monkeypatch, capsys):
    from harness.cli import main
    rec = Recorder(fake_response(content="I cannot call tools, but: ping ok"), fake_response(content="PONG"))
    monkeypatch.setattr(litellm, "completion", rec)
    assert main(["--ping", "--verbose"]) == 0
    out = capsys.readouterr().out
    assert "Endpoint:  (provider default)" in out
    assert "Probe:     text (no tool call in reply)" in out
    assert "I cannot call tools" in out and "Reply:     'PONG'" in out
    assert FAKE_KEY not in out
