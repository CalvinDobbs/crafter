# Crafter LLM reasoning framework — reboot checkpoint

Make a stateful, verified agent loop the default, with an OpenAI-compatible LLM, bounded automatic recovery, and validated movement capabilities.

## Status

Planning is incomplete. The user requested extensive planning and then paused for a reboot. Approval to commit/push this checkpoint is NOT approval to implement the framework. Resume only when asked; resolve the open scope questions before implementation.

Original session plan: `/home/bracketbot/.devin/plans/plan-ba73cd4abe1f9cac.md`. This repository checkpoint preserves the decisions and design and updates the original plan's now-stale perception findings.

## Confirmed user direction

- The agent should be the default, not just an experimental opt-in mode.
- Use the existing OpenAI SDK with a configurable OpenAI-compatible endpoint/model. No LangChain or similar orchestration dependency is needed.
- Support automatic recovery, with bounded retries and explicit state/outcome checks.
- Moving around should be an action the agent can access. Do not silently restrict the project to a parked base.
- Keep implementation paused until the remaining plan is approved.

## Current code and integration points

### main.py

`bot-code/main.py` currently loads a structure, homes, calls `scan_all`, obtains a one-shot plan, and executes alternating pick/place actions. It does not check physical outcomes before continuing and always celebrates. Dependencies are declared through PEP 723 and already include openai.

Proposed integration: default feedback-driven agent loop; explicit legacy `--mode oneshot`. Preserve existing structure/grid UI/skills URL flags. Retain deterministic reasoning inside the agent loop as the API-independent fallback, rather than falling back to blind one-shot execution.

The README reserves main.py changes for owner approval. Permission to change this integration point was asked but not answered before the reboot request.

### planner.py

The deterministic planner fills bottom-up with nearest unused boxes. The LLM planner calls `gpt-4o-mini` through chat completions in JSON-object mode. Existing validation checks target membership/support but does not fully enforce box uniqueness or duplicate-cell rejection. Investigate literal JSON braces in `PROMPT.format` as a likely pre-request formatting failure. Reproduce before making a targeted legacy fix.

### contracts.py

Frozen shared dataclasses: Block, Structure, Detection, Action, Plan. Minecraft x/z are horizontal, y is layer. Robot coordinates are +x forward, +y left, +z up. Do not change contracts without explicit team agreement. Put new agent-specific types in agent-local modules.

### perception.py — updated while planning

The initial session plan described an older scaffold. The current file has substantially changed and must be the new baseline:

- `PerceptionSession` maintains `WorldModel`, observations, and session-local tracks.
- `Pose` and `Odometry` represent base motion, timestamps, validity and map epochs.
- `BuildFrame` includes origin, column/row orientation, marker position and timestamp.
- `cell_center` now requires an oriented BuildFrame; no invented fixed-origin fallback.
- Anchor is now **49**, valid in DICT_4X4_50, rather than the earlier invalid 99.
- Depth uses sparse `idx_2d` correspondence with rectified images.
- Fresh loose candidates, protected build-zone detections, unknowns, and remembered tracks are distinct. Protected does not mean confirmed placed.
- `scan_all` rejects physical sweeps: perception does not move hardware.
- `scan` has a timeout; `verify_pick` is tri-state and never treats mere disappearance as proof of a grasp.
- `verify_place` checks geometric consistency, not motor success.
- `MockSource` supports simulated rotation, translation, visibility, pose loss and a stacking sequence. Reuse these capabilities instead of creating redundant geometry simulation.
- `--self-test` contains synthetic regression tests.

This is drifting planar odometry and an observation map, not SLAM or collision-free navigation. Verify units/calibration and suitability for action gating before live use. The README still describes the older scaffold in places; prefer actual code and recheck after reboot.

### skills_client.py and external mc_skills server

`SkillsClient` uses urllib and rejects outer `ok:false`, but ignores nested action failures. The external server at `/home/bracketbot/bbapps/mc_skills/main.py` can return `{"ok": true, "result": [false, "approach IK failed"]}`. A successful HTTP request is not successful motion.

The server serializes commands. Pick/place return success tuples, but currently ignore lift/retreat failure. State does not provide reliable possession or action-stage IDs. Rotate exists; drive does not. Never automatically repeat a motion POST after a timeout: it may still be executing.

The body server is outside this Git repository and is not included in this checkpoint. Any changes there require explicit scope approval.

### actions/pickup.py

The actions directory is now present. It contains a standalone bimanual pickup UI: down/out/pinch/raise, joint-space motion, direct ownership of both arms, and a mock option. It is not currently an HTTP skill under mc_skills and cannot run alongside another arm writer owner. Resolve whether this is the intended action implementation before building an executor around the older single-arm skills server. Do not start it during planning or checkpoint work.

### Shared VLM plumbing

`/home/bracketbot/bbapps/inference/vlm.py` offers OpenAI/Google/local clients and strict schema examples. Reuse conventions, not the unrelated camera-policy runner. OpenAI compatibility does not guarantee identical schema features on every server.

## Proposed framework

### 1. Typed state and interfaces

Use compact stdlib dataclasses/protocols for configuration, BuildState, Step, observation, action result and final run result. Start with `bot-code/agent.py`; split a backend or simulation module only when warranted.

BuildState owns target cells, confirmed placements, visible loose boxes, reserved/used IDs, current grid/base pose, freshness/epoch, measured column tops, holding state, recent structured history and failure counters. Memory lives in code, not only the model conversation.

Expose `think(state, last_result) -> Step` with OpenAI-compatible and deterministic backends. Inject model, perception, executor, and clock interfaces for offline testing.

### 2. Structured reasoning

Request strict JSON-schema responses where supported. Validate locally regardless of provider guarantees. Handle empty output, refusal, malformed payloads and incomplete responses.

Semantic actions should include pick/place, observe/scan, movement/approach and done, with exact names/parameters finalized around available action capabilities. The LLM selects a known box, target cell or bounded movement intent, not joints, arbitrary URLs or raw motor commands.

Include enabled capabilities and eligible targets in the prompt. Require only a concise decision explanation for diagnostics. Bound request timeout, retries, output size and history. Read credentials from explicitly configured environment variables; never discover or log secrets. Mock execution must not call a paid API unless explicitly requested.

### 3. Validation and preflight

Before homing, validate structure coordinates, duplicates, support and supported build dimensions. Before dispatch, reject non-target/occupied/unsupported cells, reused/protected/missing boxes, stale observations, invalid poses/epochs, unavailable capabilities and unsafe holding-state transitions.

All backends, including deterministic fallback, use the same validator. Refresh/revalidate after model latency rather than dispatching from an old snapshot.

### 4. Execution state machine

Observe -> decide -> validate -> execute one semantic step -> verify -> update state -> repeat.

Track pick/place subphases. A failed place must not permit a new pick. Reserve a box before dispatch; confirm a placement only with sufficient evidence. Keep uncertainty explicit. Report completed, blocked, exhausted or needs-operator outcomes and truthful CLI exit status. Celebrate only when the build is confirmed complete. Narration failures must not repeat physical actions.

### 5. Bounded automatic recovery

- Provider error or illegal proposal: bounded correction, then a validated deterministic decision.
- Box moved/disappeared before dispatch: fresh observation and replan.
- Known pre-grasp rejection: bounded retry/replan; quarantine repeatedly failing candidates.
- Busy: bounded wait and re-observe, without assuming another actor left the world unchanged.
- Missing observation: bounded retry, no fabricated coordinates.
- Ambiguous outcome/timeout after dispatch: do not replay blindly; reconcile only using adequate evidence, otherwise stop for review.
- Known holding after failed placement: retry/replan placement only when action-stage and possession evidence justify it.

Use max steps, per-action budgets and no-progress detection. Do not blindly park or open a potentially loaded gripper in generic cleanup. Define safe stop/cleanup with the actual action owner.

### 6. Movement capability

Movement is in scope conceptually, per the user's explicit direction. Deterministic code must implement bounded semantic movement using available localization, workspace limits, obstacle handling and action feedback.

Reuse the new perception transforms and invalidate/reacquire actionable observations after movement. Never pick remembered detections as if fresh. A pose epoch change must invalidate old action plans. Do not mistake unobserved map space for free space.

Still unresolved: whether to implement drive/navigation/body changes now or use an action layer developed elsewhere; whether movement while carrying is supported; exact reachability and stop conditions. Expose capabilities truthfully and do not advertise live motion that the action owner cannot execute safely.

### 7. Mock world and tests

Reuse MockSource and real world classification for geometry. Add executor-linked simulation that tracks holding, removes picked boxes, changes placement state and injects failures; the current scripted stacking demo alone is not a full action simulator.

Use existing perception self-tests and stdlib unittest for new tests unless further inspection reveals another framework. Fake model responses keep tests offline. No test should create hardware Writers through imports.

## Provisional file scope

- New `bot-code/agent.py`: state, decision interfaces, validation and loop.
- Optional small backend/simulation modules as needed; avoid redundant abstractions.
- New tests for decisions, validation, client results, recovery, mock builds and CLI.
- `bot-code/main.py`: default agent and truthful terminal outcomes, pending owner approval.
- `bot-code/skills_client.py`: outcome normalization/capabilities if this remains the executor boundary.
- `bot-code/actions/pickup.py`: inspect and settle integration; do not modify ownership casually.
- `bot-code/perception.py`: reuse updated APIs; only agreed narrow integration changes.
- `bot-code/planner.py`: targeted reproduced legacy fixes if included.
- External mc_skills: conditional, not approved, not part of this repository push.
- Out of scope unless expanded: contracts changes, Minecraft mod, wire format, structure receiver.

## Acceptance criteria to confirm

- Default CLI runs the feedback-driven agent.
- Deterministic mock build completes with no hardware, network or credentials.
- Fake OpenAI tests cover schema errors/refusals/timeouts/fallback.
- Invalid targets, used/protected boxes, stale geometry and premature done never dispatch motion.
- Placement failures are not counted as success; retries are bounded.
- Unknown holding/dispatch outcome never causes a blind new pick or replay.
- Movement is a validated action with truthful capability/localization gating.
- Explicit one-shot mode remains available.
- Existing user work and frozen contracts are preserved.
- Live API/hardware testing requires separate approval.

## Verification approach

Add failing tests before fixes, then implement in small slices. Proposed commands to confirm against the final design:

- `python -m unittest discover -s bot-code/tests -v`
- `uv run bot-code/perception.py --self-test`
- `uv run bot-code/main.py --mock --planner deterministic`
- `uv run bot-code/main.py --mock --mode oneshot --planner deterministic`
- `git diff --check`

Test partial builds, nested skill failures, model latency/stale poses, timeouts after dispatch, lost localization, known/unknown holding states, bounded recovery, transformed movement and concurrent ownership limitations. Do not run the real robot while verifying a Git checkpoint.

## Open questions after reboot

1. Should the executor use mc_skills or integrate the new bimanual actions module? Which developer owns that integration?
2. Include full mobile execution/navigation now, or consume movement capabilities from another action layer?
3. Approval to change main.py, perception/client and possibly the external body server?
4. Acceptance: offline tests only, real API with mock hardware, or separately supervised robot validation?
5. Existing stacks: explicit initial state, verified resume, or empty-grid precondition?
6. What trusted possession/action-stage evidence supports physical recovery?
7. Confirm workspace limits, collision handling, odometry units/calibration and marker/grid setup.

## Resume instruction

Read this checkpoint and current git state. The perception implementation evolved during this conversation; re-read actual interfaces before coding. Finish clarifying scope and present the final implementation plan for approval. The reasoning framework itself has not been implemented.
