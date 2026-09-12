# Reasoning-layer development

- Offline tests: `python -m unittest discover -s bot-code/tests -v` (stdlib only).
- Four-box demo: `uv run --offline bot-code/main.py --mock --planner deterministic`. Alternatively `python -S bot-code/main.py --mock` works without installed robot or model packages.
- Agent is the default CLI mode. `--mode oneshot` is the separate legacy fixed-grid pipeline, not a mobile fallback.
- Live integration uses `--provider module:factory` returning `agent_adapters.AgentProviders` and an explicit `--box-size` in meters. `FunctionActions` adapts ordinary action functions; they return `FunctionResult`, with possession/readiness supplied separately by `read_state`.
- Provider status/state/observation/stop calls must have bounded latency. Provider cleanup closes observation/transport resources only; it must not implicitly release, park, or disable a loaded robot. A successful action does not by itself establish possession or placement.
- Action and perception implementations are independently owned. Do not import action test scripts from reasoning code or tests. Shared `contracts.py` and `WIRE_FORMAT.md` remain frozen.
- Mock mode does not contact the model even if OPENAI_API_KEY exists, unless `--allow-api-with-mock` is explicitly supplied. Do not run real hardware or paid API calls as part of normal verification.
