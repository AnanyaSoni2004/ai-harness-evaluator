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
    for name in ("HARNESS_MODEL", "HARNESS_API_BASE", "HARNESS_TOOL_MODE", "HARNESS_SPECTRUM"):
        monkeypatch.delenv(name, raising=False)
    llm_mod._TOOL_MODE_CACHE.clear()
    llm_mod._DROPPED_PARAMS.clear()
    llm_mod._MAX_TOKENS_CAP.clear()
    llm_mod._PROMPT_TOKEN_CAP.clear()


def make_client(monkeypatch, recorder, tool_mode="native", tmp_path=None):
    monkeypatch.setattr(litellm, "completion", recorder)
    cfg = load_config()
    cfg.model.name = "openai/test-model"  # independent of the model configured in config.yaml
    cfg.model.force_text_mode_for = []
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
    with pytest.raises(FatalLLMError, match="check the API key.*Model: openai/test-model"):
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
    monkeypatch.setenv("HARNESS_MODEL", "openai/test-model")  # not matched by force_text_mode_for
    assert main(["--ping", "--verbose"]) == 0
    out = capsys.readouterr().out
    assert "Endpoint:  (provider default)" in out
    assert "Probe:     text (no tool call in reply)" in out
    assert "I cannot call tools" in out and "Reply:     'PONG'" in out
    assert FAKE_KEY not in out


# ---------------------------------------------------------------- Phase M3: portability
def test_context_budgets_derived_from_32k_window(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  context_window: 32768\n")
    cfg = load_config(str(path))
    assert cfg.context.working_budget_tokens == int(32768 * 0.55) == 18022
    assert cfg.context.max_tool_output_chars == int(32768 * 0.06 * 3.5) == 6881


def test_context_budgets_unchanged_for_large_window(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  context_window: 131072\n")  # configured budgets are already below the caps
    cfg = load_config(str(path))
    assert cfg.context.working_budget_tokens == 48000
    assert cfg.context.max_tool_output_chars == 8000


def test_context_budgets_capped_by_tokens_per_minute(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  context_window: 131072\n  tokens_per_minute: 8000\n")
    cfg = load_config(str(path))
    assert cfg.context.working_budget_tokens == int(8000 * 0.7) == 5600
    assert cfg.context.max_tool_output_chars == int(8000 * 0.2 * 3.5) == 5600
    assert cfg.spectrum.max_evidence_chars == int(8000 * 0.1 * 3.5) == 2800


def test_unsupported_param_dropped_and_retried(monkeypatch, tmp_path):
    bad = litellm.BadRequestError(message="Unsupported parameter: 'seed' is not supported with this model.",
                                  model="m", llm_provider="openai")
    rec = Recorder(bad, fake_response(content="ok"), fake_response(content="again"))
    client = make_client(monkeypatch, rec, tmp_path=tmp_path)
    assert client.complete([], None, "fix").text == "ok"
    assert "seed" in rec.kwargs[0] and "seed" not in rec.kwargs[1]
    client.complete([], None, "fix")
    assert "seed" not in rec.kwargs[2]  # stays dropped for the rest of the process
    assert rec.kwargs[1]["temperature"] == 0.0
    assert "llm_param_dropped" in (tmp_path / "t.jsonl").read_text()


def test_param_dropped_only_once(monkeypatch):
    bad = [litellm.BadRequestError(message="temperature must be 1 for this model", model="m",
                                   llm_provider="openai") for _ in range(2)]
    rec = Recorder(*bad)
    with pytest.raises(FatalLLMError, match="temperature"):
        make_client(monkeypatch, rec).complete([], None, "fix")
    assert len(rec.kwargs) == 2 and "temperature" not in rec.kwargs[1]


def test_tool_choice_dropped_in_native_mode(monkeypatch):
    bad = litellm.BadRequestError(message='"tool_choice" is not supported', model="m", llm_provider="openai")
    rec = Recorder(bad, fake_response(tool_calls=[native_call("view_file", '{"path": "a.py"}')]))
    resp = make_client(monkeypatch, rec).complete([], [llm_mod.PING_TOOL], "fix")
    assert "tool_choice" not in rec.kwargs[1] and rec.kwargs[1]["tools"]
    assert resp.tool_calls[0].name == "view_file"


def test_unrelated_bad_request_not_retried(monkeypatch):
    bad = litellm.BadRequestError(message="messages must not be empty", model="m", llm_provider="openai")
    rec = Recorder(bad)
    with pytest.raises(FatalLLMError):
        make_client(monkeypatch, rec).complete([], None, "fix")
    assert len(rec.kwargs) == 1


# ---------------------------------------------------------------- provider rate limits
def rate_limit(message: str):
    return litellm.RateLimitError(message=message, llm_provider="groq", model="m")


TPD = ('GroqException - {"error":{"message":"Rate limit reached for model `qwen/qwen3.8-27b` in organization '
       '`org_x` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 198061, Requested 2015. '
       'Please try again in 32.832s.","type":"tokens","code":"rate_limit_exceeded"}}')
OTPM = ('GroqException - {"error":{"message":"Request too large for model `qwen/qwen3.8-27b` in organization `org_x` '
        'service tier `on_demand` on output tokens per minute (OTPM): Limit 1000, Requested 1847. The request\'s '
        'expected output tokens exceed the enforced limit; reduce max_tokens","type":"tokens"}}')
TPM = 'Request too large for model `m` on tokens per minute (TPM): Limit 6000, Requested 9000.'
PER_MINUTE = 'Rate limit reached for model `m` on tokens per minute (TPM): Limit 6000, Used 5900. Please try again in 2m30s.'


def test_parse_wait_hint():
    from harness.llm import parse_wait_hint
    assert parse_wait_hint("Please try again in 32.832s.") == pytest.approx(32.832)
    assert parse_wait_hint("try again in 1m26.4s") == pytest.approx(86.4)
    assert parse_wait_hint("try again in 420ms") == pytest.approx(0.42)
    assert parse_wait_hint("no hint here") is None


def test_daily_limit_stops_immediately(monkeypatch):
    rec = Recorder(rate_limit(TPD), fake_response(content="never reached"))
    client = make_client(monkeypatch, rec)
    delays = []
    client._sleep = delays.append
    with pytest.raises(BudgetExceeded, match="daily/quota limit.*tokens per day"):
        client.complete([], None, "fix")
    assert len(rec.kwargs) == 1 and delays == []  # no pointless waiting


# Gemini says "exceeded your current quota" for per-minute limits too, with a retry delay.
GEMINI_PER_MINUTE = ('litellm.RateLimitError: GeminiException - {"error": {"code": 429, "message": "You exceeded your '
                     'current quota, please check your plan and billing details.\\n* Quota exceeded for metric: '
                     'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 250000'
                     '\\nPlease retry in 7.5s.", "status": "RESOURCE_EXHAUSTED", "details": [{"violations": [{"quotaId": '
                     '"GenerateContentInputTokensPerModelPerMinute-FreeTier"}]}, {"retryDelay": "7s"}]}}')
GEMINI_PER_DAY = GEMINI_PER_MINUTE.replace("PerMinute", "PerDay").replace("Please retry in 7.5s.", "")
OPENAI_BILLING = ('OpenAIException - {"error": {"message": "You exceeded your current quota, please check your plan '
                  'and billing details.", "type": "insufficient_quota", "code": "insufficient_quota"}}')
OPENAI_TPM = ('OpenAIException - Request too large for gpt-5.4-mini in organization org-x on tokens per min (TPM): '
              'Limit 30000, Requested 41000.')


def test_parse_wait_hint_gemini():
    from harness.llm import parse_wait_hint
    assert parse_wait_hint("Please retry in 36.919834337s.") == pytest.approx(36.919834337)
    assert parse_wait_hint('"retryDelay": "36s"') == pytest.approx(36)


def test_gemini_per_minute_quota_waits_and_retries(monkeypatch):
    rec = Recorder(rate_limit(GEMINI_PER_MINUTE), fake_response(content="ok"))
    client = make_client(monkeypatch, rec)
    delays = []
    client._sleep = delays.append
    assert client.complete([], None, "fix").text == "ok"
    assert len(delays) == 1 and 7.5 < delays[0] < 9


@pytest.mark.parametrize("message", [GEMINI_PER_DAY, OPENAI_BILLING])
def test_daily_and_billing_quotas_stop(monkeypatch, message):
    rec = Recorder(rate_limit(message))
    client = make_client(monkeypatch, rec)
    with pytest.raises(BudgetExceeded, match="daily/quota limit"):
        client.complete([], None, "fix")
    assert len(rec.kwargs) == 1


def test_openai_tpm_wording_is_learned(monkeypatch):
    rec = Recorder(rate_limit(OPENAI_TPM))
    client = make_client(monkeypatch, rec)
    with pytest.raises(ContextOverflow):
        client.complete([], None, "fix")
    assert client.prompt_token_cap == 22500


def test_reasoning_effort_and_default_temperature(monkeypatch):
    rec = Recorder(fake_response(content="a"),
                   litellm.BadRequestError(message="Unsupported parameter: 'reasoning_effort'", model="m",
                                           llm_provider="openai"),
                   fake_response(content="b"))
    client = make_client(monkeypatch, rec)
    client.cfg.model.temperature = None
    client.cfg.model.reasoning_effort = "low"
    client.complete([], None, "fix")
    assert rec.kwargs[0]["reasoning_effort"] == "low" and "temperature" not in rec.kwargs[0]
    client.complete([], None, "fix")
    assert "reasoning_effort" not in rec.kwargs[2]


def test_auth_error_names_the_key_mismatch(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "gsk_fake_key_for_tests")
    auth = litellm.AuthenticationError(message="API key not valid", llm_provider="gemini", model="m")
    client = make_client(monkeypatch, Recorder(auth))
    client.cfg.model.name = "gemini/gemini-3.5-flash"
    with pytest.raises(FatalLLMError, match="looks like a Groq key.*MODEL=groq"):
        client.complete([], None, "fix")


def test_output_too_large_shrinks_max_tokens(monkeypatch):
    rec = Recorder(rate_limit(OTPM), fake_response(content="ok"), fake_response(content="again"))
    client = make_client(monkeypatch, rec)
    delays = []
    client._sleep = delays.append
    assert client.complete([], None, "fix").text == "ok"
    assert rec.kwargs[0]["max_tokens"] == 4096 and rec.kwargs[1]["max_tokens"] == 800 and delays == []
    client.complete([], None, "fix")
    assert rec.kwargs[2]["max_tokens"] == 800  # remembered for the rest of the run


def test_input_too_large_is_context_overflow(monkeypatch):
    with pytest.raises(ContextOverflow):
        make_client(monkeypatch, Recorder(rate_limit(TPM))).complete([], None, "fix")


def test_input_too_large_learns_prompt_cap(monkeypatch):
    client = make_client(monkeypatch, Recorder(rate_limit(TPM), rate_limit(TPM.replace("6000", "9000"))))
    client.cfg.model.tokens_per_minute = None
    assert client.prompt_token_cap is None
    with pytest.raises(ContextOverflow):
        client.complete([], None, "fix")
    assert client.prompt_token_cap == int(6000 * 0.75)
    with pytest.raises(ContextOverflow):
        client.complete([], None, "fix")
    assert client.prompt_token_cap == int(6000 * 0.75)  # a larger limit never loosens a learned cap


def test_prompt_cap_from_config(monkeypatch):
    client = make_client(monkeypatch, Recorder())
    client.cfg.model.tokens_per_minute = 8000
    assert client.prompt_token_cap == 6000


def test_per_minute_limit_uses_provider_hint(monkeypatch):
    rec = Recorder(rate_limit("Please try again in 7.5s"), fake_response(content="ok"))
    client = make_client(monkeypatch, rec)
    delays = []
    client._sleep = delays.append
    assert client.complete([], None, "fix").text == "ok"
    assert len(delays) == 1 and 7.7 <= delays[0] <= 8.5


def test_long_hint_or_exhausted_retries_become_budget_exceeded(monkeypatch):
    with pytest.raises(BudgetExceeded, match="asks to wait 150s"):
        make_client(monkeypatch, Recorder(rate_limit(PER_MINUTE))).complete([], None, "fix")
    client = make_client(monkeypatch, Recorder(*[rate_limit("slow down") for _ in range(4)]))
    client.cfg.model.max_retries = 2
    with pytest.raises(BudgetExceeded, match="still limited after 2 retries"):
        client.complete([], None, "fix")


# ---------------------------------------------------------------- provider-rejected tool calls (Groq tool_use_failed)
GENERATION = '{"name": "str_replace", "arguments": {"path": "toolkit/inventory.py", "old_str": "a", "new_str": "b"}}'
TOOL_USE_FAILED = ('GroqException - ' + json.dumps({"error": {
    "message": "Tool call validation failed: attempted to call tool 'str_replace' which was not in request.tools",
    "type": "invalid_request_error", "code": "tool_use_failed", "failed_generation": GENERATION}}))


def test_rejected_tool_call_becomes_a_response(monkeypatch):
    bad = litellm.BadRequestError(message=TOOL_USE_FAILED, model="m", llm_provider="groq")
    client = make_client(monkeypatch, Recorder(bad))
    resp = client.complete([], [llm_mod.PING_TOOL], "localize")
    assert resp.finish_reason == "tool_use_failed" and resp.text == GENERATION
    assert resp.tool_calls[0].name == "str_replace" and resp.tool_calls[0].arguments["new_str"] == "b"
    assert client.metrics.llm_calls == 1


def test_rejected_tool_call_without_generation(monkeypatch):
    body = 'GroqException - {"error": {"message": "Failed to call a function.", "code": "tool_use_failed"}}'
    bad = litellm.BadRequestError(message=body, model="m", llm_provider="groq")
    resp = make_client(monkeypatch, Recorder(bad)).complete([], [llm_mod.PING_TOOL], "fix")
    assert resp.tool_calls[0].name == "__invalid__" and "could not use your reply" in resp.tool_calls[0].parse_error


def test_probe_treats_tool_use_failed_as_text(monkeypatch):
    bad = litellm.BadRequestError(message=TOOL_USE_FAILED, model="m", llm_provider="groq")
    client = make_client(monkeypatch, Recorder(bad), tool_mode="auto")
    assert client.tool_mode == "text" and "tool_use_failed" in client.probe_info["reason"]


def test_output_parse_failed_is_also_feedback(monkeypatch):
    body = ('GroqException - ' + json.dumps({"error": {
        "message": "Parsing failed. The model generated output that could not be parsed.",
        "type": "invalid_request_error", "code": "output_parse_failed",
        "failed_generation": "Search tests for remove error."}}))
    bad = litellm.BadRequestError(message=body, model="m", llm_provider="groq")
    resp = make_client(monkeypatch, Recorder(bad)).complete([], [llm_mod.PING_TOOL], "localize")
    assert resp.finish_reason == "tool_use_failed" and resp.text == "Search tests for remove error."
    assert resp.tool_calls[0].name == "__invalid__" and "could not use your reply" in resp.tool_calls[0].parse_error
