"""Shared dataclasses and exceptions (the contracts between modules)."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict
    parse_error: str | None = None


@dataclass
class Usage:
    """Token usage reported for one LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class LLMResponse:
    """A parsed model reply: text, tool calls, and usage.

    `reasoning` holds chain-of-thought (reasoning_content or stripped <think> blocks). It is logged,
    never sent back to the model.
    """

    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    finish_reason: str | None = None
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Tools, processes, tests
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """What a tool returns to the agent loop. Tools never raise; failures set ok=False."""

    ok: bool
    output: str
    data: dict = field(default_factory=dict)


@dataclass
class ProcessResult:
    """Outcome of one subprocess run by shell.run_process."""

    exit_code: int | None
    output: str
    timed_out: bool
    duration_s: float


@dataclass
class TestRun:
    """Parsed outcome of one test-suite invocation."""

    __test__ = False  # not a pytest test class despite the name

    command: str
    exit_code: int | None
    passed: int
    failed: int
    errors: int
    failing_ids: list[str]
    duration_s: float
    timed_out: bool
    env_problem: bool
    output_tail: str


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------

@dataclass
class IssueSpec:
    """Structured view of the GitHub issue text."""

    raw_text: str
    title: str = ""
    kind: str = "bug"
    summary: str = ""
    expected: str = ""
    actual: str = ""
    error_messages: list[str] = field(default_factory=list)
    mentioned_paths: list[str] = field(default_factory=list)
    mentioned_symbols: list[str] = field(default_factory=list)
    repro_hints: str = ""


@dataclass
class PhaseResult:
    """Result of running one agent phase."""

    phase: str
    finished: bool
    result: dict
    steps: int
    reason: str = ""


@dataclass
class PhaseMetrics:
    """Counters for a single phase."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    seconds: float = 0.0


@dataclass
class Metrics:
    """Per-phase and total cost counters for one run."""

    per_phase: dict[str, PhaseMetrics] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None

    def _phase(self, phase: str) -> PhaseMetrics:
        """Return the metrics bucket for a phase, creating it on first use."""
        if phase not in self.per_phase:
            self.per_phase[phase] = PhaseMetrics()
        return self.per_phase[phase]

    def add_llm(self, phase: str, usage: Usage) -> None:
        """Record one LLM call and its token usage."""
        pm = self._phase(phase)
        pm.llm_calls += 1
        pm.prompt_tokens += usage.prompt_tokens
        pm.completion_tokens += usage.completion_tokens

    def add_tool(self, phase: str) -> None:
        """Record one tool call."""
        self._phase(phase).tool_calls += 1

    def add_time(self, phase: str, seconds: float) -> None:
        """Add wall-clock seconds spent in a phase."""
        self._phase(phase).seconds += seconds

    @property
    def prompt_tokens(self) -> int:
        """Total prompt tokens across phases."""
        return sum(p.prompt_tokens for p in self.per_phase.values())

    @property
    def completion_tokens(self) -> int:
        """Total completion tokens across phases."""
        return sum(p.completion_tokens for p in self.per_phase.values())

    @property
    def total_tokens(self) -> int:
        """Total prompt + completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def llm_calls(self) -> int:
        """Total LLM calls across phases."""
        return sum(p.llm_calls for p in self.per_phase.values())

    @property
    def tool_calls(self) -> int:
        """Total tool calls across phases."""
        return sum(p.tool_calls for p in self.per_phase.values())

    def to_dict(self) -> dict:
        """JSON-serialisable summary: per-phase counters plus totals."""
        end = self.ended_at if self.ended_at is not None else time.time()
        return {
            "per_phase": {name: asdict(pm) for name, pm in self.per_phase.items()},
            "total": {
                "llm_calls": self.llm_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "tool_calls": self.tool_calls,
                "seconds": round(end - self.started_at, 3),
            },
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


@dataclass
class SpectrumResult:
    """Outcome of execution-based fault localization (the Tracer).

    lines items: {"path", "line", "score", "ef", "ep", "function"}.
    functions items: {"path", "name", "start", "end", "score", "top_line", "ef", "ep"}.
    """

    ok: bool
    reason: str = ""
    formula: str = "ochiai"
    lines: list[dict] = field(default_factory=list)
    functions: list[dict] = field(default_factory=list)
    failing_runs: int = 0
    passing_runs: int = 0
    seconds: float = 0.0
    low_confidence: bool = False

    def to_dict(self) -> dict:
        """JSON-serialisable view."""
        return asdict(self)


def _jsonable(value: Any) -> Any:
    """Recursively convert Paths (and nested containers) into JSON-friendly values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class RunState:
    """Everything the orchestrator knows about one run; phases hand off through this."""

    run_id: str
    repo: str
    issue: IssueSpec
    candidates: list[dict] = field(default_factory=list)
    localization: dict = field(default_factory=dict)
    repro: dict = field(default_factory=dict)
    baseline_full: TestRun | None = None
    baseline_targeted: TestRun | None = None
    targeted_tests: list[str] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    review: dict = field(default_factory=dict)
    status: str = "running"
    notes: list[str] = field(default_factory=list)
    spectrum: dict = field(default_factory=dict)
    stop_reason: str = ""  # why further fix attempts were abandoned (a normal outcome, not an error)

    def to_dict(self) -> dict:
        """JSON-serialisable view of the whole state."""
        return _jsonable(asdict(self))


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class HarnessError(Exception):
    """Base class for harness errors that should be reported cleanly."""


class FatalLLMError(HarnessError):
    """The model endpoint cannot be used (auth, unknown model, bad request)."""


class ContextOverflow(HarnessError):
    """The prompt exceeded the model's context window."""


class BudgetExceeded(HarnessError):
    """A token, call, or time budget was exhausted."""
