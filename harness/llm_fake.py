"""FakeLLM: a scripted stand-in for LLMClient used by tests."""
from __future__ import annotations

import copy
from typing import Any, Callable, Union

from harness.types import BudgetExceeded, LLMResponse, Metrics

ScriptItem = Union[LLMResponse, Callable[[list], LLMResponse]]


class FakeLLM:
    """Replays scripted responses; records every messages list it receives in self.calls."""

    def __init__(self, script: list[ScriptItem], metrics: Metrics | None = None, cfg: Any = None,
                 tool_mode: str = "native") -> None:
        """script items are LLMResponse objects or callables (messages) -> LLMResponse."""
        self.script = list(script)
        self.metrics = metrics if metrics is not None else Metrics()
        self.cfg = cfg
        self.tool_mode = tool_mode
        self.calls: list[list[dict]] = []
        self.tools_seen: list[list[dict] | None] = []
        self.phases: list[str] = []

    def probe_tool_mode(self) -> str:
        """Return the configured tool mode (no network)."""
        return self.tool_mode

    def complete(self, messages: list[dict], tools: list[dict] | None, phase: str) -> LLMResponse:
        """Return the next scripted response and update metrics like the real client."""
        if self.cfg is not None:
            budgets = self.cfg.budgets
            if self.metrics.total_tokens >= budgets.max_total_tokens or self.metrics.llm_calls >= budgets.max_llm_calls:
                raise BudgetExceeded("budget exhausted")
        self.calls.append(copy.deepcopy(messages))
        self.tools_seen.append(copy.deepcopy(tools))
        self.phases.append(phase)
        if not self.script:
            raise AssertionError("FakeLLM script exhausted")
        item = self.script.pop(0)
        resp = item(messages) if callable(item) else item
        self.metrics.add_llm(phase, resp.usage)
        return resp
