# AI Coding Harness

An autonomous coding-agent harness for the LCC × DevClub AI Coding Harness Hackathon 2026.

## Quickstart

```bash
export AI_API_KEY="<PROVIDED_API_KEY>"
make setup
make run
```

## Determinism

All output-affecting settings live in `config.yaml` under `model:`. Runs use `temperature: 0.0` and
`seed: 42`. Some DeepSeek/Qwen endpoints do not support `seed`. When an endpoint rejects `seed`,
`temperature` or `tool_choice`, the harness drops that parameter for the rest of the run and records it
in the run's `trajectory.jsonl` (`llm_param_dropped`). **If `seed` is unsupported, `temperature: 0.0`
alone is our reproducibility claim**, and provider-side sampling may still vary slightly between runs.

## Execution-based fault localization (Tracer)

`REPRODUCE → TRACE → FIX`

After the harness has a reproduction that fails, it runs that reproduction and the related passing tests
under a stdlib line tracer (`harness/_trace_runner.py`, inside the target repo's own Python) and ranks every
executed line by Ochiai suspiciousness, `ef / sqrt(nf · (ef + ep))`: lines the failing run executes but passing
tests rarely do score highest. The FIX agent gets the top suspicious functions, with source context, as
evidence. This costs **zero tokens**, and the report shows it as its own `trace` row.

It narrows the search; it does not prove anything. The defect can be a *missing* line or a caller of the
flagged code, and the prompt says so. With too few passing tests to contrast against, it reports no ranking
instead of a list of ties. It is purely additive: in a non-Python repo, with no reproduction, on a timeout or
a crash, the run continues unchanged and the report records why the Tracer was skipped.

Ablation (with vs. without the Tracer, `HARNESS_SPECTRUM=0`): pending the `make eval` benchmark (Step 24).
