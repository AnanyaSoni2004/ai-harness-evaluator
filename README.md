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
