"""AgentLoop: runs one phase of tool-using conversation with the model."""
from __future__ import annotations

import json
import time
from typing import Any

from harness import textproto
from harness.context import compact, estimate_tokens
from harness.events import NullUI, Trajectory
from harness.tools.registry import FINISH, ToolRegistry
from harness.types import ContextOverflow, LLMResponse, Metrics, PhaseResult, ToolCall, ToolResult

NUDGE = "Continue by calling exactly one tool, or call finish if the phase goal is met."
WRAP_UP = "You have 3 tool calls left in this phase. Wrap up and call finish."
FORCE_FINISH = "Step budget exhausted. Call finish now with your best result."
BLOCKED_REPEAT = ("Blocked: identical call already made twice. Use the earlier output or try a different "
                  "approach.")
MAX_NUDGES = 3
MAX_IDENTICAL = 2
OVERFLOW_KEEP_RECENT = 4
LOG_OUTPUT_CHARS = 2000


def _preview(name: str, args: dict) -> str:
    """Short human-readable description of a tool call for the UI."""
    if name == "view_file":
        start, end = args.get("start_line"), args.get("end_line")
        rng = f":{start or 1}-{end}" if end else (f":{start}" if start else "")
        return f"{name} {args.get('path', '')}{rng}"
    if name in ("str_replace", "create_file", "list_dir", "repo_map"):
        return f"{name} {args.get('path', '.')}"
    if name == "run_command":
        return f"{name} {str(args.get('command', ''))[:80]}"
    if name == "search_code":
        where = f" in {args['path']}" if args.get("path") not in (None, "", ".") else ""
        return f"{name} {args.get('query', '')!r}{where}"
    if name == "find_files":
        return f"{name} {args.get('pattern', '')}"
    if name == "run_tests":
        return f"{name} {args.get('targets') or '(all)'}"
    return name


class AgentLoop:
    """Runs one phase: model turns with tools until finish, budget exhaustion, or a protocol failure."""

    def __init__(self, llm: Any, registry: ToolRegistry, system_prompt: str, task: str, phase: str, cfg: Any,
                 metrics: Metrics, trajectory: Trajectory, ui: Any = None, max_steps: int = 20,
                 deadline: float | None = None) -> None:
        """deadline is a time.monotonic() timestamp; registry must contain the phase's finish tool."""
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.task = task
        self.phase = phase
        self.cfg = cfg
        self.metrics = metrics
        self.trajectory = trajectory
        self.ui = ui if ui is not None else NullUI()
        self.max_steps = max_steps
        self.deadline = deadline
        self.history: list[dict] = []
        self.steps = 0
        self._signatures: dict[str, int] = {}
        self._nudges = 0
        self._parse_failures = 0
        self._last_text = ""
        self._warned = False

    # ------------------------------------------------------------------ messages
    @property
    def native(self) -> bool:
        return self.llm.tool_mode == "native"

    def _system(self, registry: ToolRegistry) -> str:
        """System prompt, plus the text-protocol tool instructions in text mode."""
        if self.native:
            return self.system_prompt
        return self.system_prompt + "\n\n" + textproto.render_tool_instructions(registry.openai_schemas())

    def _messages(self, registry: ToolRegistry) -> list[dict]:
        return [{"role": "system", "content": self._system(registry)}, {"role": "user", "content": self.task},
                *self.history]

    def _budget(self) -> int:
        """History budget: the working budget, or less when the provider caps the prompt size per request."""
        budget = int(self.cfg.context.working_budget_tokens)
        cap = getattr(self.llm, "prompt_token_cap", None)
        return min(budget, int(cap)) if cap else budget

    def _compact(self, keep_recent: int) -> None:
        """Compact the stored history in place (system and task are never touched)."""
        budget = self._budget()
        compacted = compact(self._messages(self.registry), keep_recent, budget)
        if estimate_tokens(compacted) > budget and keep_recent > 1:
            compacted = compact(compacted, 1, budget)  # still too big: shorten everything but the last result
        self.history = compacted[2:]
        self.trajectory.log("context_compacted", phase=self.phase, keep_recent=keep_recent,
                            tokens=estimate_tokens(compacted))

    def _complete(self, registry: ToolRegistry) -> LLMResponse | None:
        """One model call with compaction; None means the context still overflowed after compacting."""
        messages = self._messages(registry)
        if estimate_tokens(messages) > self._budget():
            self._compact(self.cfg.context.keep_recent_messages)
            messages = self._messages(registry)
        tools = registry.openai_schemas() if self.native else None
        try:
            return self.llm.complete(messages, tools, self.phase)
        except ContextOverflow:
            # The client may just have learned a smaller prompt cap, so _budget() is re-read here.
            self._compact(OVERFLOW_KEEP_RECENT)
            try:
                return self.llm.complete(self._messages(registry), tools, self.phase)
            except ContextOverflow:
                return None

    # ------------------------------------------------------------------ tools
    def _finish_problem(self, call: ToolCall) -> str | None:
        """Why a finish call cannot end the phase (bad JSON or missing required fields), or None."""
        spec = self.registry.get(FINISH)
        if spec is None:
            return None
        example = textproto.format_example(spec.schema())
        if call.parse_error:
            return f"finish arguments were not valid JSON ({call.parse_error}). {example}"
        missing = [r for r in spec.parameters.get("required") or [] if call.arguments.get(r) is None]
        if missing:
            return f"finish is missing required field(s): {', '.join(missing)}. {example}"
        return None

    def _execute(self, call: ToolCall) -> ToolResult:
        """Dispatch one non-finish call with loop detection, metrics, UI and trajectory logging."""
        signature = call.name + json.dumps(call.arguments, sort_keys=True, default=str)
        preview = _preview(call.name, call.arguments)
        self.ui.tool_call(call.name, preview)
        if self._signatures.get(signature, 0) >= MAX_IDENTICAL:
            result = ToolResult(False, BLOCKED_REPEAT)
        else:
            result = self.registry.dispatch(call)
            self._signatures[signature] = self._signatures.get(signature, 0) + 1
            spec = self.registry.get(call.name)
            if result.ok and spec is not None and spec.kind == "write":
                # The code changed, so re-running earlier commands is progress, not a loop.
                self._signatures = {signature: self._signatures[signature]}
        self.metrics.add_tool(self.phase)
        lines = [l for l in result.output.strip().splitlines() if l.strip()]
        # Skip the echoed "$ command" line; the → line above already shows the call.
        first_line = next((l for l in lines if not l.startswith("$ ")), lines[0] if lines else "")
        self.ui.tool_result(result.ok, first_line[:120])
        self.trajectory.log("tool_call", phase=self.phase, name=call.name, arguments=call.arguments,
                            ok=result.ok, output=result.output[:LOG_OUTPUT_CHARS])
        return result

    # ------------------------------------------------------------------ turns
    def _handle_rejected(self, resp: LLMResponse) -> PhaseResult | None:
        """A tool call the provider rejected (native mode): answer it as text, never as a native tool_call."""
        self._nudges = 0
        self._parse_failures += 1
        call = resp.tool_calls[0] if resp.tool_calls else None
        self.history.append({"role": "assistant", "content": resp.text or "(tool call rejected by the API)"})
        if call is not None and call.name == FINISH and not call.parse_error and self._finish_problem(call) is None:
            self.metrics.add_tool(self.phase)
            self.trajectory.log("phase_finish", phase=self.phase, result=call.arguments, steps=self.steps)
            return PhaseResult(self.phase, True, dict(call.arguments), self.steps)
        if call is None or call.name == textproto.INVALID_TOOL:
            reason = f" ({call.parse_error})" if call is not None and call.parse_error else ""
            result = ToolResult(False, f"The API could not use your last reply{reason}. Call exactly one of the "
                                       "available tools, with valid JSON arguments.")
        elif call.name == FINISH:
            result = ToolResult(False, self._finish_problem(call) or f"Invalid finish call: {call.parse_error}")
        else:
            result = self._execute(call)
        self.history.append({"role": "user", "content": "Your tool call was rejected by the API. "
                             + textproto.format_tool_result(call.name if call else "?", result)})
        limit = int(getattr(self.cfg.model, "max_consecutive_parse_failures", 3) or 3)
        if self._parse_failures >= limit:
            self.trajectory.log("protocol_failure", phase=self.phase, failures=self._parse_failures)
            return PhaseResult(self.phase, False, {"text": self._last_text}, self.steps, "protocol_failure")
        return None

    def _handle(self, resp: LLMResponse) -> PhaseResult | None:
        """Apply one model reply to the history; a PhaseResult ends the phase."""
        if resp.finish_reason == "tool_use_failed":
            return self._handle_rejected(resp)
        calls = resp.tool_calls if self.native else resp.tool_calls[:1]
        self._last_text = resp.text or self._last_text
        if not calls:
            if resp.text:
                self.history.append({"role": "assistant", "content": resp.text})
            self._nudges += 1
            if self._nudges >= MAX_NUDGES:
                return PhaseResult(self.phase, False, {"text": self._last_text}, self.steps, "no_tool_calls")
            self.history.append({"role": "user", "content": NUDGE})
            return None
        self._nudges = 0

        if any(c.name == textproto.INVALID_TOOL or c.parse_error for c in calls):
            self._parse_failures += 1
        else:
            self._parse_failures = 0

        if self.native:
            self.history.append({"role": "assistant", "content": resp.text or None, "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": json.dumps(c.arguments, default=str)}} for c in calls]})
        else:
            self.history.append({"role": "assistant", "content": resp.text})

        finish: ToolCall | None = None
        for call in calls:
            if call.name == FINISH and finish is None:
                finish = call
                continue
            result = self._execute(call) if call.name != FINISH else ToolResult(False, "Only one finish call.")
            self._append_result(call, result)

        if finish is not None:
            problem = self._finish_problem(finish)
            self.ui.tool_call(FINISH, "finish")
            if problem is None:
                self.metrics.add_tool(self.phase)
                self.ui.tool_result(True, "phase finished")
                self.trajectory.log("phase_finish", phase=self.phase, result=finish.arguments, steps=self.steps)
                return PhaseResult(self.phase, True, dict(finish.arguments), self.steps)
            self.ui.tool_result(False, problem[:120])
            self._append_result(finish, ToolResult(False, problem))

        limit = int(getattr(self.cfg.model, "max_consecutive_parse_failures", 3) or 3)
        if self._parse_failures >= limit:
            self.trajectory.log("protocol_failure", phase=self.phase, failures=self._parse_failures)
            return PhaseResult(self.phase, False, {"text": self._last_text}, self.steps, "protocol_failure")
        return None

    def _append_result(self, call: ToolCall, result: ToolResult) -> None:
        """Add a tool result to the history in the current protocol's format."""
        if self.native:
            self.history.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                 "content": result.output})
        else:
            self.history.append({"role": "user", "content": textproto.format_tool_result(call.name, result)})

    def _out_of_time(self) -> bool:
        return self.deadline is not None and time.monotonic() >= self.deadline

    def _force_finish(self, reason: str) -> PhaseResult:
        """One last call with only the finish tool available."""
        finish_only = self.registry.subset([FINISH])
        self.history.append({"role": "user", "content": FORCE_FINISH})
        self.trajectory.log("force_finish", phase=self.phase, reason=reason, steps=self.steps)
        if FINISH in finish_only:
            resp = self._complete(finish_only)
            if resp is not None:
                self.steps += 1
                finish = next((c for c in resp.tool_calls if c.name == FINISH), None)
                if finish is not None and self._finish_problem(finish) is None:
                    self.metrics.add_tool(self.phase)
                    self.trajectory.log("phase_finish", phase=self.phase, result=finish.arguments,
                                        steps=self.steps, forced=True)
                    return PhaseResult(self.phase, True, dict(finish.arguments), self.steps, reason)
                self._last_text = resp.text or self._last_text
        return PhaseResult(self.phase, False, {"text": self._last_text}, self.steps, reason)

    def run(self) -> PhaseResult:
        """Run the phase. BudgetExceeded and FatalLLMError propagate to the orchestrator."""
        self.trajectory.log("phase_start", phase=self.phase, max_steps=self.max_steps, native=self.native)
        while self.steps < self.max_steps:
            if self._out_of_time():
                return self._force_finish("deadline")
            resp = self._complete(self.registry)
            if resp is None:
                return PhaseResult(self.phase, False, {"text": self._last_text}, self.steps, "context_overflow")
            self.steps += 1
            outcome = self._handle(resp)
            if outcome is not None:
                return outcome
            if self.max_steps - self.steps == 3 and not self._warned:
                self._warned = True
                self.history.append({"role": "user", "content": WRAP_UP})
        return self._force_finish("max_steps")
