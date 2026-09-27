"""LLMClient built on LiteLLM: retries, tool-mode probe, budgets."""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Callable

import litellm

from harness import textproto
from harness.events import NullUI, Trajectory
from harness.types import (
    BudgetExceeded,
    ContextOverflow,
    FatalLLMError,
    LLMResponse,
    Metrics,
    ToolCall,
    Usage,
)

litellm.drop_params = True
litellm.suppress_debug_info = True


def _exc_tuple(names: list[str]) -> tuple[type, ...]:
    """Collect the named litellm exception classes that exist in this litellm version."""
    found = (getattr(litellm, n, None) for n in names)
    return tuple(c for c in found if isinstance(c, type))


RETRYABLE = _exc_tuple(["RateLimitError", "APIConnectionError", "Timeout", "InternalServerError",
                        "ServiceUnavailableError", "APIError"])
CONTEXT_ERRORS = _exc_tuple(["ContextWindowExceededError"])
AUTH_ERRORS = _exc_tuple(["AuthenticationError", "PermissionDeniedError", "NotFoundError"])
BAD_REQUEST = _exc_tuple(["BadRequestError"])
RATE_LIMIT = _exc_tuple(["RateLimitError"])

# Provider limits that waiting a minute cannot fix (Groq: "tokens per day (TPD)"; OpenAI: quota).
_DAILY_LIMIT = re.compile(r"per day|\((?:TPD|RPD)\)|insufficient_quota|exceeded your current quota", re.I)
# One request asks for more output than the per-minute output limit allows: shrink max_tokens instead.
_OUTPUT_TOO_LARGE = re.compile(r"Request too large.*?output tokens per minute.*?Limit (\d+)", re.I | re.S)
_TOO_LARGE = re.compile(r"Request too large", re.I)
# The per-minute input limit (Groq: "tokens per minute (TPM): Limit 8000"). One prompt above it can never succeed.
_TPM_LIMIT = re.compile(r"tokens per minute \(TPM\):\s*Limit (\d+)", re.I)
_TRY_AGAIN = re.compile(r"try again in\s+([0-9hms.]+)", re.I)
_PROVIDER_MESSAGE = re.compile(r'"message"\s*:\s*"([^"]+)"')
MAX_HINTED_WAIT_S = 90.0
MIN_MAX_TOKENS = 256
# Share of a per-minute token limit one prompt may use: estimate_tokens is approximate, so keep headroom.
PROMPT_SHARE_OF_TPM = 0.75
_MAX_TOKENS_CAP: dict[str, int] = {}  # model name -> max_tokens learned from "Request too large" errors
_PROMPT_TOKEN_CAP: dict[str, int] = {}  # model name -> prompt size learned from input "Request too large" errors


# Provider rejected what the model generated (Groq: tool_use_failed, output_parse_failed, ... + failed_generation).
_REJECTED_GENERATION = re.compile(r"tool_use_failed|output_parse_failed|failed_generation")


class ToolUseFailed(Exception):
    """The provider rejected the model's own output or tool call (Groq HTTP 400 with `failed_generation`)."""

    def __init__(self, generation: str, message: str) -> None:
        super().__init__(message)
        self.generation = generation


def _failed_generation(error: Exception) -> str:
    """The rejected tool call text the provider echoes back (`failed_generation`), or ''."""
    text = str(error)
    start = text.find("{")
    while start != -1:
        try:
            body = json.loads(text[start:])
            return str((body.get("error") or {}).get("failed_generation") or "")
        except ValueError:
            start = text.find("{", start + 1)
    return ""


def parse_wait_hint(text: str) -> float | None:
    """Seconds from 'try again in 1m26.4s' / '32.832s' / '420ms', or None."""
    m = _TRY_AGAIN.search(text or "")
    if not m:
        return None
    units = {"ms": 0.001, "h": 3600.0, "m": 60.0, "s": 1.0}
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", m.group(1))
    return sum(float(n) * units[u] for n, u in parts) if parts else None


def _provider_message(error: Exception) -> str:
    """The provider's own error message (short), for notes and warnings."""
    text = str(error)
    m = _PROVIDER_MESSAGE.search(text)
    msg = m.group(1) if m else text
    msg = re.sub(r" in organization `[^`]*`", "", msg)  # account IDs do not belong on screen or in reports
    msg = re.sub(r" service tier `[^`]*`", "", msg)
    msg = re.split(r"\s*Need more tokens\?", msg)[0]  # provider upsell text
    return msg[:300]

AUTH_MESSAGE = "Authentication/model error: check AI_API_KEY and model.name in config.yaml"

PING_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Reply to a ping.",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string", "description": "The message to send."}},
            "required": ["message"],
        },
    },
}

# Optional request parameters some endpoints reject; dropped per model after the first rejection.
DROPPABLE_PARAMS = ("seed", "temperature", "tool_choice")
_DROPPED_PARAMS: dict[str, set[str]] = {}


def _rejected_param(error_text: str, kwargs: dict) -> str | None:
    """The droppable parameter named in a bad-request error, if it is still being sent."""
    lowered = error_text.lower()
    for param in DROPPABLE_PARAMS:
        if param in kwargs and re.search(rf"\b{param}\b", lowered):
            return param
    return None


# Probe results ({"mode", "reason", "raw"}) are cached per (model, endpoint) for the process lifetime.
_TOOL_MODE_CACHE: dict[tuple[str, str | None], dict[str, str]] = {}


class LLMClient:
    """Single entry point to the model: budgets, retries, error mapping, and tool-call parsing."""

    def __init__(self, cfg: Any, metrics: Metrics, trajectory: Trajectory, ui: Any = None) -> None:
        """Create a client; tool mode is resolved lazily (probed once when set to 'auto')."""
        self.cfg = cfg
        self.metrics = metrics
        self.trajectory = trajectory
        self.ui = ui if ui is not None else NullUI()
        self._tool_mode: str | None = None if cfg.model.tool_mode == "auto" else cfg.model.tool_mode
        self._sleep: Callable[[float], None] = time.sleep
        self._n_calls = 0
        # Diagnostics from the last tool-mode decision: {"mode", "reason", "raw"}.
        self.probe_info: dict[str, str] = {}

    # ------------------------------------------------------------------ tool mode
    @property
    def tool_mode(self) -> str:
        """'native' or 'text'; probes the endpoint on first access when configured as 'auto'."""
        if self._tool_mode is None:
            self._tool_mode = self.probe_tool_mode()
        return self._tool_mode

    def _forced_text_match(self) -> str | None:
        """The model.force_text_mode_for entry matching model.name (case-insensitive), if any."""
        name = self.cfg.model.name.lower()
        for pattern in self.cfg.model.force_text_mode_for or []:
            if str(pattern).strip() and str(pattern).lower() in name:
                return str(pattern)
        return None

    @staticmethod
    def _judge_probe(message: Any) -> tuple[str, str]:
        """Decide (mode, reason) from the probe reply: native only for a well-formed ping call."""
        calls = getattr(message, "tool_calls", None) or []
        if not calls:
            return "text", "no tool call in reply"
        fn = getattr(calls[0], "function", None)
        name = str(getattr(fn, "name", "") or "")
        if name != PING_TOOL["function"]["name"]:
            return "text", f"returned unknown tool name {name!r}"
        raw_args = getattr(fn, "arguments", None)
        if isinstance(raw_args, dict):
            return "native", "well-formed ping call"
        try:
            if not isinstance(json.loads(raw_args or ""), dict):
                raise ValueError("not an object")
        except (ValueError, TypeError):
            return "text", "tool arguments are not a JSON object"
        return "native", "well-formed ping call"

    @staticmethod
    def _raw_reply(message: Any) -> str:
        """Content plus any tool calls of a reply, as text for diagnostics."""
        parts = [str(getattr(message, "content", "") or "")]
        for tc in getattr(message, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            parts.append(f"[tool_call {getattr(fn, 'name', '')}({getattr(fn, 'arguments', '')})]")
        return " ".join(p for p in parts if p)

    def probe_tool_mode(self) -> str:
        """Decide native vs text tool calling once per (model, endpoint); see _judge_probe."""
        key = (self.cfg.model.name, self.cfg.model.api_base)
        if key in _TOOL_MODE_CACHE:
            self.probe_info = dict(_TOOL_MODE_CACHE[key], reason="cached: " + _TOOL_MODE_CACHE[key]["reason"])
            return self.probe_info["mode"]
        forced = self._forced_text_match()
        if forced is not None:
            info = {"mode": "text", "reason": f"forced by force_text_mode_for entry {forced!r}", "raw": ""}
        else:
            messages = [{"role": "user", "content": "Call the ping tool with message 'ok'."}]
            try:
                raw = self._call_with_retries(messages, [PING_TOOL], "probe")
                message = raw.choices[0].message
                self._record_usage(raw, "probe")
                mode, reason = self._judge_probe(message)
                info = {"mode": mode, "reason": reason, "raw": self._raw_reply(message)}
            except ToolUseFailed as e:
                info = {"mode": "text", "reason": "provider rejected the model's tool call (tool_use_failed)",
                        "raw": e.generation or str(e)}
            except FatalLLMError as e:
                if isinstance(e.__cause__, BAD_REQUEST) and not isinstance(e.__cause__, AUTH_ERRORS):
                    info = {"mode": "text", "reason": "endpoint rejected tools (bad request)", "raw": str(e)}
                else:
                    raise
        self.trajectory.log("tool_mode_probe", model=self.cfg.model.name, mode=info["mode"],
                            reason=info["reason"], raw=info["raw"][:200])
        _TOOL_MODE_CACHE[key] = info
        self.probe_info = dict(info)
        return info["mode"]

    # ------------------------------------------------------------------ calls
    @property
    def prompt_token_cap(self) -> int | None:
        """Largest prompt the provider accepts in one request (model.tokens_per_minute or learned), or None."""
        caps = [_PROMPT_TOKEN_CAP.get(self.cfg.model.name)]
        tpm = getattr(self.cfg.model, "tokens_per_minute", None)
        if tpm:
            caps.append(int(int(tpm) * PROMPT_SHARE_OF_TPM))
        known = [c for c in caps if c]
        return min(known) if known else None

    def _check_budget(self) -> None:
        """Raise BudgetExceeded if the run's token or call budget is used up."""
        budgets = self.cfg.budgets
        if self.metrics.total_tokens >= budgets.max_total_tokens:
            raise BudgetExceeded(f"token budget exhausted ({self.metrics.total_tokens} tokens)")
        if self.metrics.llm_calls >= budgets.max_llm_calls:
            raise BudgetExceeded(f"LLM call budget exhausted ({self.metrics.llm_calls} calls)")

    def _call_with_retries(self, messages: list[dict], tools: list[dict] | None, phase: str) -> Any:
        """Call litellm.completion, retrying transient errors and mapping the rest to harness errors."""
        m = self.cfg.model
        kwargs: dict[str, Any] = {
            "model": m.name,
            "messages": messages,
            "api_key": self.cfg.api_key(),
            "api_base": m.api_base,
            "temperature": m.temperature,
            "max_tokens": min(m.max_output_tokens, _MAX_TOKENS_CAP.get(m.name, m.max_output_tokens)),
            "timeout": m.request_timeout_s,
        }
        if m.seed is not None:
            kwargs["seed"] = m.seed
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        dropped = _DROPPED_PARAMS.setdefault(m.name, set())
        for param in dropped:
            kwargs.pop(param, None)
        attempt = 0
        while True:
            try:
                with self.ui.thinking(phase):
                    return litellm.completion(**kwargs)
            except CONTEXT_ERRORS as e:
                raise ContextOverflow(str(e)) from e
            except AUTH_ERRORS as e:
                raise FatalLLMError(AUTH_MESSAGE) from e
            except BAD_REQUEST as e:
                if _REJECTED_GENERATION.search(str(e)):
                    raise ToolUseFailed(_failed_generation(e), _provider_message(e)) from e
                param = _rejected_param(str(e), kwargs)
                if param is None:
                    raise FatalLLMError(str(e)) from e
                # The endpoint rejects this parameter: drop it for the rest of the process, retry once.
                dropped.add(param)
                kwargs.pop(param)
                self.trajectory.log("llm_param_dropped", phase=phase, model=m.name, param=param,
                                    error=str(e)[:300])
                self.ui.warn(f"Endpoint rejected '{param}'; dropping it for this run")
            except RATE_LIMIT as e:
                text = str(e)
                if _DAILY_LIMIT.search(text):
                    raise BudgetExceeded(f"provider daily/quota limit reached: {_provider_message(e)}") from e
                too_large = _OUTPUT_TOO_LARGE.search(text)
                if too_large:
                    cap = max(MIN_MAX_TOKENS, int(int(too_large.group(1)) * 0.8))
                    if kwargs["max_tokens"] > cap:
                        _MAX_TOKENS_CAP[m.name] = cap
                        kwargs["max_tokens"] = cap
                        self.trajectory.log("llm_max_tokens_reduced", phase=phase, model=m.name, max_tokens=cap)
                        self.ui.warn(f"Provider output limit: max_tokens reduced to {cap} for this run")
                        continue
                elif _TOO_LARGE.search(text):
                    # Over the per-minute input limit, not the context window: remember the limit so the
                    # agent compacts below it (retrying the same prompt can never succeed).
                    tpm = _TPM_LIMIT.search(text)
                    if tpm:
                        cap = int(int(tpm.group(1)) * PROMPT_SHARE_OF_TPM)
                        _PROMPT_TOKEN_CAP[m.name] = min(cap, _PROMPT_TOKEN_CAP.get(m.name, cap))
                        self.trajectory.log("llm_prompt_cap_learned", phase=phase, model=m.name,
                                            prompt_tokens=_PROMPT_TOKEN_CAP[m.name])
                        self.ui.warn(f"Provider per-minute token limit: keeping prompts under "
                                     f"~{_PROMPT_TOKEN_CAP[m.name]} tokens for this run")
                    raise ContextOverflow(_provider_message(e)) from e
                hint = parse_wait_hint(text) or self._retry_after(e)
                if attempt >= m.max_retries:
                    raise BudgetExceeded(f"rate limit: still limited after {attempt} retries: "
                                         f"{_provider_message(e)}") from e
                if hint is not None and hint > MAX_HINTED_WAIT_S:
                    raise BudgetExceeded(f"rate limit: provider asks to wait {hint:.0f}s: {_provider_message(e)}") from e
                delay = (hint + random.uniform(0.2, 1.0)) if hint is not None else min(2 ** attempt, 30) + random.uniform(0, 1)
                self.trajectory.log("llm_retry", phase=phase, attempt=attempt + 1, error="RateLimitError",
                                    delay_s=round(delay, 2), hinted=hint is not None)
                self.ui.warn(f"Rate limited; retrying in {delay:.1f}s" + (" (provider hint)" if hint is not None else ""))
                self._sleep(delay)
                attempt += 1
            except RETRYABLE as e:
                if attempt >= m.max_retries:
                    raise FatalLLMError(f"Model endpoint failed after {attempt} retries: {e}") from e
                delay = min(2 ** attempt, 30) + random.uniform(0, 1)
                self.trajectory.log("llm_retry", phase=phase, attempt=attempt + 1,
                                    error=type(e).__name__, delay_s=round(delay, 2))
                self.ui.warn(f"Model call failed ({type(e).__name__}); retrying in {delay:.1f}s")
                self._sleep(delay)
                attempt += 1

    def _rejected_tool_call(self, error: ToolUseFailed, phase: str) -> LLMResponse:
        """Turn a provider-rejected tool call into a response the agent answers with feedback."""
        calls = textproto.parse_tool_calls(error.generation, f"call{self._n_calls}")
        if not calls:
            calls = [ToolCall(f"call{self._n_calls}_0", textproto.INVALID_TOOL, {},
                              parse_error=f"the API could not use your reply as a tool call ({error})")]
        usage = Usage()  # the provider reports no usage for rejected calls
        self.metrics.add_llm(phase, usage)
        self.trajectory.log("llm_tool_use_failed", phase=phase, message=str(error)[:300],
                            generation=error.generation[:1000])
        self.ui.warn("The API rejected the model's tool call; asking it to correct itself")
        return LLMResponse(text=error.generation, tool_calls=calls, usage=usage, finish_reason="tool_use_failed")

    @staticmethod
    def _retry_after(error: Exception) -> float | None:
        """Seconds from an HTTP Retry-After header on the provider response, if present."""
        try:
            value = getattr(getattr(error, "response", None), "headers", {}).get("retry-after")
            return float(value) if value is not None else None
        except (TypeError, ValueError, AttributeError):
            return None

    def _record_usage(self, raw: Any, phase: str) -> Usage:
        """Add the response's token usage to the metrics and return it."""
        u = getattr(raw, "usage", None)
        usage = Usage(prompt_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
                      completion_tokens=int(getattr(u, "completion_tokens", 0) or 0))
        self.metrics.add_llm(phase, usage)
        return usage

    def _parse_native_calls(self, raw_calls: Any) -> list[ToolCall]:
        """Convert provider tool_calls into ToolCall objects, flagging undecodable arguments."""
        calls: list[ToolCall] = []
        for i, tc in enumerate(raw_calls or []):
            fn = getattr(tc, "function", None)
            name = str(getattr(fn, "name", "") or "")
            call_id = str(getattr(tc, "id", "") or f"call{self._n_calls}_{i}")
            raw_args = getattr(fn, "arguments", None)
            if isinstance(raw_args, dict):
                calls.append(ToolCall(call_id, name, raw_args))
                continue
            try:
                args = json.loads(raw_args) if raw_args and str(raw_args).strip() else {}
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
                calls.append(ToolCall(call_id, name, args))
            except (ValueError, TypeError) as e:
                calls.append(ToolCall(call_id, name, {}, parse_error=str(e)))
        return calls

    def complete(self, messages: list[dict], tools: list[dict] | None, phase: str) -> LLMResponse:
        """Send one chat request and return the parsed response. Enforces budgets first."""
        self._check_budget()
        native = self.tool_mode == "native"
        self._n_calls += 1
        try:
            raw = self._call_with_retries(messages, tools if (tools and native) else None, phase)
        except ToolUseFailed as e:
            return self._rejected_tool_call(e, phase)

        choice = raw.choices[0]
        message = choice.message
        # Chain-of-thought is kept out of `text` so it is never replayed into the history.
        text, inline_reasoning = textproto.split_reasoning(str(getattr(message, "content", None) or ""))
        separate_reasoning = str(getattr(message, "reasoning_content", None) or "").strip()
        reasoning = "\n\n".join(r for r in (separate_reasoning, inline_reasoning) if r)
        if native:
            tool_calls = self._parse_native_calls(getattr(message, "tool_calls", None))
        else:
            tool_calls = textproto.parse_tool_calls(text, f"call{self._n_calls}")
        finish_reason = getattr(choice, "finish_reason", None)

        usage = self._record_usage(raw, phase)
        self.trajectory.log("llm_call", phase=phase, prompt_tokens=usage.prompt_tokens,
                            completion_tokens=usage.completion_tokens, finish_reason=finish_reason,
                            tool_calls=[{"name": c.name, "arguments": c.arguments} for c in tool_calls],
                            text=text[:2000], reasoning=reasoning[:4000])
        if finish_reason == "length":
            msg = f"{phase}: model output was cut off (finish_reason=length), often from overthinking"
            self.trajectory.log("llm_truncated", phase=phase, completion_tokens=usage.completion_tokens)
            self.ui.warn(msg)
        self.ui.llm_call(phase, usage)
        return LLMResponse(text=text, tool_calls=tool_calls, usage=usage, finish_reason=finish_reason,
                           reasoning=reasoning)
