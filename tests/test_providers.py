"""Provider support: model presets, key lookup, and the real LiteLLM request path per provider (HTTP is faked)."""
import json
from pathlib import Path

import httpx
import pytest

from harness import llm as llm_mod
from harness import providers
from harness.config import load_config
from harness.events import Trajectory, redact
from harness.llm import LLMClient
from harness.shell import sanitised_env
from harness.types import HarnessError, Metrics

GROQ_KEY = "gsk_fake_key_for_tests"
GEMINI_KEY = "AIza_fake_key_for_tests"
PLAIN_KEY = "sk-fake-key-for-tests"


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AI_API_KEY", "HARNESS_API_BASE", "HARNESS_TOOL_MODE"):
        monkeypatch.delenv(name, raising=False)
    llm_mod._TOOL_MODE_CACHE.clear()
    llm_mod._DROPPED_PARAMS.clear()


# ---------------------------------------------------------------- model selection
@pytest.mark.parametrize("preset, name", [
    ("groq", "groq/qwen/qwen3.8-27b"),
    ("gemini", "gemini/gemini-3.5-flash"),
    ("openai", "openai/gpt-5.4-mini"),
    ("qwen", "dashscope/qwen3-coder-plus"),
    ("ollama", "ollama_chat/qwen3:8b"),
])
def test_presets_expand(preset: str, name: str) -> None:
    cfg = load_config(model=preset)
    assert cfg.model.name == name and cfg.model.preset == preset and cfg.model.source == "--model"


def test_groq_limits_stay_with_groq() -> None:
    groq = load_config(model="groq")
    assert groq.model.tokens_per_minute == 8000 and groq.model.force_text_mode_for == ["qwen3"]
    assert groq.context.working_budget_tokens == 5600
    gemini = load_config(model="gemini")
    assert gemini.model.tokens_per_minute is None and gemini.model.force_text_mode_for == []
    assert gemini.model.context_window == 1048576 and gemini.context.working_budget_tokens == 48000
    assert gemini.model.temperature is None and gemini.model.reasoning_effort == "low"


def test_qwen_presets_pick_the_region() -> None:
    assert "dashscope-intl" in load_config(model="qwen").model.api_base
    assert "dashscope-intl" not in load_config(model="qwen-cn").model.api_base


def test_full_litellm_name_and_context_lookup() -> None:
    assert load_config(model="openai/gpt-5.4-mini").model.context_window == 272000
    unknown = load_config(model="openai/my-own-served-model")
    assert unknown.model.context_window == 32768 and unknown.model.preset is None


def test_model_specific_settings_do_not_carry_over(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  name: groq/x\n  tokens_per_minute: 6000\n  api_base: http://example.invalid/v1\n")
    assert load_config(str(path)).model.tokens_per_minute == 6000
    other = load_config(str(path), model="openai/gpt-5.4-mini")
    assert other.model.tokens_per_minute is None and other.model.api_base is None


def test_precedence_flag_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL", "qwen")
    assert load_config().model.preset == "qwen"
    assert load_config(model="gemini").model.preset == "gemini"


@pytest.mark.parametrize("key, preset", [(GROQ_KEY, "groq"), (GEMINI_KEY, "gemini"),
                                         ("sk-proj-fake", "openai"), ("sk-or-fake", "openrouter")])
def test_auto_detects_the_provider_from_the_key(monkeypatch: pytest.MonkeyPatch, key: str, preset: str) -> None:
    monkeypatch.setenv("AI_API_KEY", key)
    cfg = load_config()
    assert cfg.model.preset == preset and cfg.model.source == "auto: AI_API_KEY format"


def test_auto_from_provider_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", PLAIN_KEY)
    cfg = load_config()
    assert cfg.model.preset == "qwen" and cfg.model.source == "auto: DASHSCOPE_API_KEY is set"
    assert cfg.key_and_source() == (PLAIN_KEY, "DASHSCOPE_API_KEY")


def test_auto_with_an_ambiguous_key_asks_for_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_API_KEY", PLAIN_KEY)  # OpenAI, DashScope and DeepSeek all issue sk-... keys
    with pytest.raises(HarnessError, match="Could not tell the provider.*MODEL=<groq\\|gemini"):
        load_config().api_key()
    assert load_config(model="qwen").api_key() == PLAIN_KEY


def test_no_key_at_all() -> None:
    with pytest.raises(HarnessError, match="No API key found"):
        load_config().api_key()
    with pytest.raises(HarnessError, match="set AI_API_KEY or GEMINI_API_KEY or GOOGLE_API_KEY"):
        load_config(model="gemini").api_key()


# ---------------------------------------------------------------- keys
def test_provider_variable_is_used_without_ai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    assert load_config(model="gemini").key_and_source() == (GEMINI_KEY, "GEMINI_API_KEY")


def test_mismatched_ai_api_key_yields_to_the_provider_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_API_KEY", GROQ_KEY)
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    assert load_config(model="gemini").key_and_source() == (GEMINI_KEY, "GEMINI_API_KEY")
    assert load_config(model="groq").key_and_source() == (GROQ_KEY, "AI_API_KEY")


def test_local_models_need_no_key() -> None:
    assert load_config(model="ollama").key_and_source() == (None, "none")
    cfg = load_config(model="openai/local-model")
    cfg.model.api_base = "http://127.0.0.1:8000/v1"
    assert cfg.api_key() is None


def test_key_mismatch_hint() -> None:
    assert "MODEL=groq" in providers.key_mismatch(GROQ_KEY, "gemini")
    assert providers.key_mismatch(GROQ_KEY, "groq") is None
    assert providers.key_mismatch(PLAIN_KEY, "openai") is None


def test_every_provider_key_is_redacted_and_kept_from_child_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key-123")
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    assert redact(f"a fake-openai-key-123 b {GEMINI_KEY}") == "a *** b ***"
    env = sanitised_env()
    assert "OPENAI_API_KEY" not in env and "GEMINI_API_KEY" not in env


# ---------------------------------------------------------------- the real request path, HTTP faked
TOOL = {"type": "function", "function": {"name": "view_file", "description": "Show a file.", "parameters": {
    "type": "object", "properties": {"path": {"type": "string", "description": "file"}}, "required": ["path"]}}}
USAGE = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}


def provider_reply(request: httpx.Request) -> dict:
    """A tool call in the wire format of the provider the request went to."""
    url = str(request.url)
    if "generativelanguage.googleapis.com" in url:
        return {"candidates": [{"content": {"role": "model", "parts": [
                    {"functionCall": {"name": "view_file", "args": {"path": "a.py"}}, "thoughtSignature": "SIG"}]},
                "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3, "totalTokenCount": 13}}
    if request.url.path.endswith("/responses"):
        return {"id": "resp_1", "object": "response", "created_at": 1, "status": "completed", "model": "m",
                "output": [{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "view_file",
                            "arguments": '{"path": "a.py"}', "status": "completed"}],
                "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}}
    return {"id": "x", "object": "chat.completion", "created": 1, "model": "m", "service_tier": "on_demand",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {
                    "name": "view_file", "arguments": '{"path": "a.py"}'}}]}}],
            "usage": USAGE}


@pytest.fixture()
def wire(monkeypatch: pytest.MonkeyPatch) -> list:
    """Record every HTTP request LiteLLM sends and answer it like the provider would."""
    sent: list[tuple[httpx.Request, dict]] = []

    def send(self, request, **kwargs):
        sent.append((request, json.loads(request.content or b"{}")))
        return httpx.Response(200, json=provider_reply(request), request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    return sent


def two_turns(preset: str, monkeypatch: pytest.MonkeyPatch) -> LLMClient:
    """A native tool call, then the follow-up request that carries its result."""
    monkeypatch.setenv("AI_API_KEY", "fake-wire-key")
    cfg = load_config(model=preset)
    cfg.model.tool_mode, cfg.model.force_text_mode_for = "native", []
    client = LLMClient(cfg, Metrics(), Trajectory(None))
    client._sleep = lambda s: None
    first = client.complete([{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}], [TOOL], "fix")
    call = first.tool_calls[0]
    assert (call.name, call.arguments) == ("view_file", {"path": "a.py"}) and client.metrics.total_tokens == 13
    history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
               {"role": "assistant", "content": None, "tool_calls": [{"id": call.id, "type": "function", "function": {
                   "name": call.name, "arguments": json.dumps(call.arguments)}}]},
               {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": "file text"}]
    client.complete(history, [TOOL], "fix")
    return client


def test_gemini_wire(wire, monkeypatch: pytest.MonkeyPatch) -> None:
    two_turns("gemini", monkeypatch)
    request, body = wire[-1]
    assert "gemini-3.5-flash:generateContent" in str(request.url)
    assert body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "low"
    model_turn = body["contents"][1]["parts"][0]
    assert model_turn["thoughtSignature"] == "SIG"  # Gemini 3 rejects a replayed call without it
    assert body["contents"][2]["parts"][0]["function_response"]["response"] == {"content": "file text"}


def test_openai_wire(wire, monkeypatch: pytest.MonkeyPatch) -> None:
    two_turns("openai", monkeypatch)
    request, body = wire[-1]
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert body["reasoning"] == {"effort": "low"} and "temperature" not in body
    kinds = [item.get("type") for item in body["input"]]
    assert kinds[-2:] == ["function_call", "function_call_output"]
    assert body["input"][-1]["call_id"] == body["input"][-2]["call_id"]


@pytest.mark.parametrize("preset, url", [
    ("qwen", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"),
    ("groq", "https://api.groq.com/openai/v1/chat/completions"),
])
def test_chat_completions_wire(wire, monkeypatch: pytest.MonkeyPatch, preset: str, url: str) -> None:
    two_turns(preset, monkeypatch)
    request, body = wire[-1]
    assert str(request.url) == url and request.headers["authorization"] == "Bearer fake-wire-key"
    assert body["temperature"] == 0.0 and body["messages"][-1] == {
        "role": "tool", "tool_call_id": "call_1", "name": "view_file", "content": "file text"}


def test_local_model_sends_no_key(wire, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(model="openai/local-model")
    cfg.model.api_base, cfg.model.tool_mode = "http://localhost:8000/v1", "native"
    LLMClient(cfg, Metrics(), Trajectory(None)).complete([{"role": "user", "content": "hi"}], [TOOL], "fix")
    request, _ = wire[-1]
    assert str(request.url) == "http://localhost:8000/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer no-key-needed"
