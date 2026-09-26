"""LLMClient built on LiteLLM: retries, tool-mode probe, budgets."""
from __future__ import annotations

import json
import random
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

# Probe results are cached per (model, endpoint) for the lifetime of the process.
_TOOL_MODE_CACHE: dict[tuple[str, str | None], str] = {}


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
        self.last_probe_reply = ""

    # ------------------------------------------------------------------ tool mode
    @property
    def tool_mode(self) -> str:
        """'native' or 'text'; probes the endpoint on first access when configured as 'auto'."""
        if self._tool_mode is None:
            self._tool_mode = self.probe_tool_mode()
        return self._tool_mode

    def probe_tool_mode(self) -> str:
        """Ask the model to call a ping tool; 'native' if it returns a tool call, else 'text'."""
        key = (self.cfg.model.name, self.cfg.model.api_base)
        if key in _TOOL_MODE_CACHE:
            return _TOOL_MODE_CACHE[key]
        messages = [{"role": "user", "content": "Call the ping tool with message 'ok'."}]
        try:
            raw = self._call_with_retries(messages, [PING_TOOL], "probe")
            message = raw.choices[0].message
            self.last_probe_reply = str(getattr(message, "content", "") or "")
            self._record_usage(raw, "probe")
            mode = "native" if getattr(message, "tool_calls", None) else "text"
        except FatalLLMError as e:
            if isinstance(e.__cause__, BAD_REQUEST) and not isinstance(e.__cause__, AUTH_ERRORS):
                mode = "text"
            else:
                raise
        self.trajectory.log("tool_mode_probe", model=self.cfg.model.name, mode=mode)
        _TOOL_MODE_CACHE[key] = mode
        return mode

    # ------------------------------------------------------------------ calls
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
            "max_tokens": m.max_output_tokens,
            "timeout": m.request_timeout_s,
        }
        if m.seed is not None:
            kwargs["seed"] = m.seed
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
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
                raise FatalLLMError(str(e)) from e
            except RETRYABLE as e:
                if attempt >= m.max_retries:
                    raise FatalLLMError(f"Model endpoint failed after {attempt} retries: {e}") from e
                delay = min(2 ** attempt, 30) + random.uniform(0, 1)
                self.trajectory.log("llm_retry", phase=phase, attempt=attempt + 1,
                                    error=type(e).__name__, delay_s=round(delay, 2))
                self.ui.warn(f"Model call failed ({type(e).__name__}); retrying in {delay:.1f}s")
                self._sleep(delay)
                attempt += 1

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
        raw = self._call_with_retries(messages, tools if (tools and native) else None, phase)

        choice = raw.choices[0]
        message = choice.message
        text = getattr(message, "content", None) or ""
        if native:
            tool_calls = self._parse_native_calls(getattr(message, "tool_calls", None))
        else:
            tool_calls = textproto.parse_tool_calls(text, f"call{self._n_calls}")
        finish_reason = getattr(choice, "finish_reason", None)

        usage = self._record_usage(raw, phase)
        self.trajectory.log("llm_call", phase=phase, prompt_tokens=usage.prompt_tokens,
                            completion_tokens=usage.completion_tokens, finish_reason=finish_reason,
                            tool_calls=[{"name": c.name, "arguments": c.arguments} for c in tool_calls],
                            text=text[:2000])
        self.ui.llm_call(phase, usage)
        return LLMResponse(text=text, tool_calls=tool_calls, usage=usage, finish_reason=finish_reason)
