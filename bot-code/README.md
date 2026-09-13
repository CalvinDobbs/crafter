# crafter bot-code — digital-twin block builder

Project-level sequencing, demo definition, and the voice-during-motion plan: **[../PLAN.md](../PLAN.md)**. This file is the robot-app architecture and runbook.

For parallel component development, start with the **[agent component interface v2](#agent-component-interface-v2)** below. It is the current action/perception handoff; legacy body-server endpoints are not themselves a complete agent provider.

## The idea

A player builds a structure in Minecraft; the Fabric mod
(`../minecraft-mod/`, Structure Scanner item) streams it to the bot over
TCP:5005 per `../WIRE_FORMAT.md`. The robot finds physical cardboard boxes,
picks them up, and stacks them so the pile matches the in-game build. It is
NOT teleoperation — the game build is a *target spec*; the robot maintains a
world model, plans which box goes to which cell, and executes pick/place
primitives. An LLM does high-level reasoning; deterministic code does motion.

Demo story: "You build it in Minecraft. It builds it in real life."

## Platform context (read before writing code)

BracketBot: self-balancing 2-wheel base, two 8-DoF arms (index 7 = gripper),
head stereo camera, 2 wrist cameras, mic/speaker, LED. All I/O is **bbos IPC**
— shared-memory topics via `Reader`/`Writer`/`Type`/`Config`.

- **ONE writer per topic.** A second `Writer("arm_right.ctrl")` anywhere else
  raises `RuntimeError` naming the owning PID. Exactly ONE process at a time may
  own `arm_*.ctrl`, `*.torque`, `drive.ctrl`, `led.ctrl`, `speaker.audio`.
  Today that owner is `mc_skills` when it is running; `actions/pickup.py` opens
  arm writers directly and must never run alongside it. Going forward the
  interface-v2 action provider is the designated owner. Do not open these
  Writers from reasoning, perception, panel or test code, and never resolve a
  conflict by killing the owner or restarting a BBOS daemon.
  Details and the per-topic contracts: [actions/motion_and_arms.md](actions/motion_and_arms.md).
- **Readers are unlimited.** Perception can run live on the robot at any time
  without conflicting with anything.
- Arm control = position in motor turns. Cartesian via IK:
  `Config("arm_right").ik.solve(pos[3], quat_xyzw[4]) -> 7 urdf joints` →
  `urdf2q` → turns.
- `camera.points` = **base-frame xyz per pixel** — detections arrive already
  in robot coordinates. Base frame: +x forward, +y left, +z up, meters.
- Homing: use `staged_home_arms`/`park_arms` from
  `bbapps/quest_teleop/scripts/homing.py` (mc_skills does). Leave arms limp.
  Note the converse: the arm daemon cuts torque as soon as its control writer
  disappears, so closing writers drops whatever the arm is holding.
- VLM plumbing exists in `~/bbapps/inference/vlm.py` (multi-provider clients,
  `grab_right_eye`, strict JSON schema) — reuse it, don't rebuild it.
- Full platform doc: `~/bbapps/AGENTS.md`.

## The world model (core concept — read this)

Each `scan()` produces a snapshot in the robot's **current** base frame:

```python
Scan { boxes: [Detection],     # current loose candidates
       protected: [Detection], # inside footprint, NOT proof of placement
       build: BuildFrame|None, # measured origin and oriented column/row axes
       pose: Pose, ts: float }
```

- **Anchor marker** is ArUco id **49**, in `DICT_4X4_50`. The measured
  `BuildFrame` carries orientation; an xyz point is insufficient. Missing/stale
  registration is unknown, never a fixed `GRID_ORIGIN` fallback. Planar odometry
  is not SLAM, free-space certification or collision-free navigation.
- **Grid gate:** detections inside the footprint are protected, not pick
  candidates. `stacked` is a deprecated alias for protected, not an assertion
  that a box has been successfully placed.
- **Dual tracking:** `placed` (agent's authoritative set — what SHOULD be
  there) vs `column_heights()` (measured z per cell from the pointcloud —
  what IS there). Disagreement = fumble → feed back, re-plan.

## Architecture

```text
Minecraft --TCP:5005--> panel.StructureReceiver --+
Fixtures / grid UI ------------------------------+--> Structure --> Agent
                                                                    |
                                           Reasoner <--- state + images + choices
                                                                    |
                                          +-------------------------+------------------+
                                          |                                            |
                                    ActionProvider                            ObservationProvider
                              submit/status/cancel/stop                  observe/sites/images/monitor
                                          |                                            |
                          explicit live component adapters, OR temporary shared mock world
                                   world.actions                              world.observations
```

The live action provider talks to the designated hardware owner; it does not
create another writer. The mock providers have no hardware or network access.

## Files

```
~/bbapps/mc_skills/main.py   body server — ONLY hardware writer owner
bot-code/
  contracts.py               shared types (pure python, no bbos) — FROZEN
  skills_client.py           SkillsClient (HTTP) + MockSkills
  structure_src.py           .nbt import, structure.json, legacy grid web UI
                             (serve_grid_ui defaults to :8005 — the panel's port;
                              pass another port, or the two collide)
  panel.py                   control panel and EOF-framed TCP :5005 receiver
  debug_console.py           manual tool bench behind the panel's debug screen
  perception.py              independently owned sensing/world model
  planner.py                 legacy one-shot plan_build()
  agent.py                   verified per-step reasoning and monitoring
  agent_types.py             pure-stdlib component interface v2
  agent_adapters.py          live factory seam and ordinary-function bridge
  mock_agent_world.py        temporary action/perception contract simulator
  main.py                    default agent CLI; explicit legacy oneshot mode
  web/                       browser-only panel files
    panel.html               markup and controls
    panel.css                responsive viewport layout
    panel.js                 previews, polling and interactions
    debug.js                 debug screen: perception readouts and tool calls
    assets/
      Crafter-transparent.svg  header logo
  actions/                   independently owned hardware action scripts
    pickup.py                joint-space two-arm pickup prototype
    motion_and_arms.md       source-checked base + arm control guide
  tests/                     offline component and UI regression tests
  voice/                     speech packs, playback and voice documentation
  fixtures/                  structure_house.json, world_state.json,
                             scan_sample.json (boxes+anchor+stacked+heights)
```

## Contracts

`contracts.py` is the single source of truth — frozen, changes need team
agreement. Perception-local types (`Scan`, `ANCHOR_ID`, `is_in_grid`,
anchor-relative `cell_center`) currently live in `perception.py`, marked as
candidates for promotion — do NOT edit contracts.py unilaterally.

```python
Block(x, y, z, kind)            # minecraft cell; y = layer (0 = on table)
Structure(blocks)               # target spec; .layers(), .supported()
Detection(id, pos, color, size) # physical box; pos = base-frame xyz meters
Action(kind, box_id, cell)      # 'pick' | 'place'
Plan(actions, narration)
```

Wire → contract: mod sends `{origin,size,count,palette,blocks:[[x,y,z,i]]}`,
MC axes +X east/+Z south/+Y up; coords relative to origin. MC `y` = our layer.
Receiver: accept → **read to EOF** (half-close framing) → parse →
`save_structure`. Backlog ≥8, sends can overlap. See `../WIRE_FORMAT.md`.

## perception.py — Readers-only sensing, not a complete agent provider

- `scan(mock=False) -> Scan` uses ArUco `DICT_4X4_50` (49 = anchor), sparse
  depth/image correspondence and pose-aligned world tracking. Loose, protected,
  unknown and remembered observations are distinct.
- `is_in_grid`, `cell_center`, `resolve_anchor` require an oriented `BuildFrame`;
  `column_heights` measures stack tops from current depth.
- `verify_pick` returns false or unknown; disappearance alone never proves a grasp.
  `verify_place` checks geometric height consistency, not action success or identity.
- `scan_all` is legacy compatibility and rejects physical sweeps. The action
  provider owns movement; perception never commands a sweep.
- **Markerless detection** is available alongside ArUco: `--prepare-detector`
  fetches pinned YOLO-World + CLIP ONNX weights, then `--viz --detector boxes`
  runs local proposals. `GET /objects` exposes session-local track IDs, original
  bboxes, raw detector scores, depth and position uncertainty.
- **Debug viz:** `uv run perception.py --viz [--mock]` → :8007 top-down map
  (robot, grid, loose/stacked markers, anchor) + live head-cam feed.
- **Verified live** on bracketbot-184 with cameras and the box detector: valid
  observations at ~9 FPS capture / ~8 FPS detection, sub-millisecond read
  latency, image ages around 0.4–1.0 s. Possession, occupancy and site safety
  stayed **unknown** — that is correct, not a gap to paper over.
- Prep: print box markers 0..N-1 and anchor 49. Hardcoded positions are mock
  fixtures only, never a substitute for missing live evidence.
- ⚠️ FIRST LIVE CHECK: confirm `camera.points` is pixel-aligned to the left
  half of head rgb (compare shapes) before trusting positions.

### Connect live perception to reasoning

`agent_adapters.PerceptionObservations` connects to the **existing** perception
visualizer's `/scan` and `/frame?view=rect` endpoints (default port 8007). It does
not import `perception.py`, start another detector, own hardware readers/writers,
or call action scripts. Reuse the running perception process; inspect bot
services before starting anything. From Windows, use SSH for bot-side commands,
or forward port 8007 to read the bot's loopback-only service locally.

Inspect a live snapshot from the repository root without an action provider,
robot packages, or a model call:

```bash
python -B -S bot-code/main.py --observe-perception http://127.0.0.1:8007
```

The URL is relative to the machine running this command: on the bot it reaches
the bot's perception process; on a developer PC it needs the corresponding SSH
forward. This mode rejects `--mock`, `--ui`, `--provider`, and legacy execution.
It never moves the robot or reports a completed build. Output includes structured
observations and image metadata, **not image bytes**. Even with a saved API key,
the default `auto` planner does not contact a model in this inspection mode.

To request one real, read-only model decision on the bot, explicitly opt in:

```bash
uv run --offline bot-code/main.py --observe-perception http://127.0.0.1:8007 --planner llm
```

This sends the live image and structured world model to the configured model and
can incur API charges. Only `observe` and `stop` are allowed; neither invokes an
action provider. Optional `--structure PATH --box-size METERS` supplies the target
build as context without attempting execution.

The adapter preserves the useful world-model information rather than reducing
perception to box centers:

- Persistent tracks, original measurement times, current/remembered status and
  loose/protected/unknown classifications. Markerless surface estimates remain
  separate from actionable box geometry; unknown depth and ambiguous identities
  are not repaired or promoted to grasp poses. `observed_current` retains recent
  visual detections even when localization is invalid and metric `current` is false.
- The oriented anchor/build registration in its original base-at-capture frame,
  plus `build_world` transformed into the same session-world frame as box tracks.
  Box size, cell pitch, footprint dimensions, anchor validity and age are retained.
  A registered anchor is **not** a feasibility or clearance certificate.
- Observed surface cells, independent measurement times, height bounds, and a
  whole-map summary. Large maps are bounded to 32 KiB of world-model JSON, favoring
  fresh samples near the robot/build. `coverage` reports included and omitted
  records; omitted or blank cells remain unknown, never free space. Redundant
  precomputed build-cell centers are replaced by their count; axes/pitch remain.
- Detector labels, scores, bounding boxes, session IDs, identity/depth status,
  visible-surface estimates, pose/epoch, calibration diagnostics, stream freshness,
  wheel/IMU telemetry, detector health and warnings.
- A real rectified camera image whose own capture time is bracketed by scans in
  the same map epoch. Unavailable, stale or mismatched images are not substituted
  with simulated frames. Transport failures make observations unavailable and
  retry in the background; they never refresh cached evidence timestamps.

`ObservationSnapshot.world_model_json` is immutable, bounded JSON;
`snapshot.world_model` returns a detached dictionary for inspection. The agent
passes it to the reasoner as `world_model`, alongside the existing inventory and
vision inputs. Empty `{}` remains backward-compatible with other v2 providers.

When the independently developed actions are ready, use this observation provider
in their existing `--provider module:factory` handoff:

```python
from agent_adapters import AgentProviders, PerceptionObservations

observations = PerceptionObservations("http://127.0.0.1:8007")
providers = AgentProviders(actions=your_action_provider, observations=observations,
                           close=observations.close)
```

The factory should return `providers` and also close any action **transport**
resources it owns, without moving or releasing a load. Perception calls read a
background cache (at most 200 ms initial wait); no camera or model startup runs
inside an observation call. This adapter advertises only inventory and images.
Site feasibility, complete cell occupancy, phase-aware safety monitoring and
possession remain unavailable, so the normal build agent still refuses execution
at preflight. Its safety checks and the frozen Minecraft contracts are unchanged.

Focused offline integration checks use a local fake HTTP perception server and
fake model clients; they never contact the bot or a paid API:

```bash
python -B -S -m unittest discover -s bot-code/tests -p test_agent_adapters.py -v
```

## The panel runs on the robot, never on a developer PC

`main.py --ui` refuses to start unless `bbos` is importable, because everything the panel needs
is on the bot: the saved API key, the cameras, the arms, and the TCP:5005 receiver the Minecraft
mod points at. A panel started on a laptop has none of them — it prompts for an API key that
already exists on the robot, and its debug screen can open neither perception nor actions.

Start it on the bot and forward the port:

```bash
ssh bracketbot@100.66.148.86 'uv run --offline /home/bracketbot/crafter/bot-code/main.py --ui'
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8005:127.0.0.1:8005 bracketbot@100.66.148.86
```

Then open <http://127.0.0.1:8005>. Check the bot's ports first with
`ss -ltnp '( sport = :8005 or sport = :5005 )'` and reuse a panel that is already running rather
than starting a second receiver.

`--ui-without-robot` starts one locally anyway. It exists for browser layout work only: no key, no
perception, no actions.


## Debug screen — driving the tools by hand

The panel has a second screen, reached from **Debug console** in the header, for looking at
everything perception reports and calling each of the agent's operations one at a time. It is
`debug_console.py` on the server and `web/debug.js` in the browser; the build flow is untouched.

It calls the component interface, not a parallel implementation: `observe`, `find_build_sites`,
`check_build_site`, and `submit`/`status`/`monitor`/`cancel`/`stop` on the same providers,
inside the same `ActionRequest` envelope `Agent._execute` builds. Arguments are validated the way
the agent's are, so a call that is refused here is refused identically inside a build.

**Nothing on this screen is simulated, deliberately.** A fake reading is indistinguishable from a
real one, and you would tune against it. The default providers are the deployed pair —
`observations.RobotObservations` in this process, `actions.provider.build_providers` behind the
arm switch — and off the robot the screen says it cannot open them rather than showing something
plausible. `test_debug_console` asserts the module cannot even reach a simulator.

The screen shows the whole `ObservationSnapshot`: validity and pose, the camera frame, every
observed box with its eligibility and age, the selected site and its per-cell occupancy, warnings,
and the raw world model. `done` measures the target cells the way `FINAL_VERIFY` would and reports
what the agent would conclude, without confirming anything.

### The two halves open separately

Perception is readers only, so it opens as soon as you look at anything and is safe beside any
hardware owner. The action provider opens `arm_*.ctrl` and `drive.ctrl` the moment it is
constructed — this panel becomes the designated owner — so it waits for a deliberate **Arm**. That
ordering is the reason inspecting perception still works while `mc_skills` holds the writers; if
arming fails because something else owns them, the page says which PID.

Disarming refuses further motions but does **not** close the provider: releasing `arm_*.ctrl` makes
the arm daemon cut torque, and a held box would drop. `stop` is never gated, and reports honestly
that it owns no motors when nothing has been armed.

### Overrides

| Launch | Perception | Motions |
| --- | --- | --- |
| `main.py --ui` | this robot, in process | the real arms, once armed |
| `--ui --debug-perception URL` | a running `perception.py --viz` over HTTP | refused; no action provider |
| `--ui --debug-provider module:factory --box-size M` | that factory's pair | that factory's actions, once armed |

`--debug-perception` is the remote-development path: `PerceptionObservations` against the existing
visualizer (default `:8007`), starting no detector and owning no hardware. It also embeds that
server's own monitor page. `--debug-provider` takes the same `module:factory` spec as `--provider`.
A build that is running refuses manual motions until it is cancelled.

The shell itself never scrolls, so the two debug panes scroll instead; Stop, Arm and Back sit in
the page heading, outside both panes, where they stay reachable.

## Agent component interface v2

This section and the pure-stdlib types in **`agent_types.py`** are the handoff
contract for independently developed actions, perception, and reasoning. They do
not change the frozen `contracts.py` or Minecraft wire format. Both providers
must advertise `INTERFACE_VERSION` (currently **2**); v1 providers fail preflight
rather than silently running without monitoring.

### Ownership and replacement boundary

| Component | Implements | Must not do |
|---|---|---|
| Reasoning (`agent.py`, `agent_backend.py`) | Job ledger, legal choices, retries, monitoring/stop decisions, final verification | Import hardware scripts, invent observations, generate motor setpoints |
| Action owner | `ActionProvider`: semantic motion, live state, action lifecycle and load-preserving stop | Infer possession from a successful command, replay an uncertain request, run a second hardware writer |
| Perception owner | `ObservationProvider`: inventory, site geometry, occupancy, images and phase-aware monitoring | Move hardware, convert unobserved space to free space, refresh old evidence timestamps |
| Temporary simulation (`mock_agent_world.py`) | Separate `world.actions` and `world.observations` providers sharing one simulated world | Claim physical calibration, navigation or camera accuracy has been validated |

Integrators return `AgentProviders(actions, observations, close)` from an explicit
`module:factory`. `close()` closes observation/transport resources only. It must
not release, park, torque off, or otherwise move a loaded robot. The reasoning
code does not import `actions/pickup.py` or `perception.py`; adapters are the seam.
Do not start independent arm-owning scripts beside the designated motion owner.
Never combine live motion with mocked possession, occupancy or safety clearance.
Mixed providers are for offline component tests, not permission to move real hardware.

### Model-visible operations

The model selects a `Step` from the controller's current `allowed_choices`.
Only the controller creates an `ActionRequest`. `reason` is a short explanation;
unused target fields must be `None`.

| Operation | Step fields | Preconditions and required result |
|---|---|---|
| `observe` | none | Read-only refresh; obtains current images and structured evidence |
| `look_around` | `search="materials"` or `"sites"` | Empty grippers; bounded, collision-checked survey; re-observe after settling |
| `select_site` | `site_id` | Measured, feasible candidate fits the entire build; does not move hardware |
| `approach_box` | `box_id`, `site_id`, `cell` | Fresh eligible box, empty grippers, supported empty target; reserve assignment and approach |
| `pickup` | `box_id` | Approached box still current; grasp/lift and independently establish possession |
| `move_to_build` | `box_id`, `site_id`, `cell` | Known held identity; navigate with load-aware clearance and confirm arrival |
| `place` | `box_id`, `site_id`, `cell` | Known held box, confirmed approach and support; lower/release/retreat, then verify placement |
| `done` | none | Request final verification; only the controller can declare completion |
| `stop` | none | Stop this job, retain unresolved load/dispatch obligations and report operator help if needed |

Do not expose raw velocity, joint angles, gripper opening, homing, arbitrary URLs
or shell commands as model tools. Motion primitives and emergency stopping belong
to the action owner, not to model-generated plans.

### Coordinates, identities and time

- Schematic cells are integer **`(x, layer, z)`**; `voxel_size`, `extents`, and
  build dimensions use **`(width, height, depth)`** in that same order.
- Physical positions are **world XYZ meters**, +Z up. Base yaw is in radians.
  Snapshot `base_position`/`base_yaw` locate the robot in that world frame.
- `BuildSite.origin` is the bottom-center of cell `(0,0,0)`. `col` maps schematic
  +x, `row` maps schematic +z, and `cross(col,row)` points up. Both are unit axes.
  `site.cell_center(cell, voxel_size)` supplies the common conversion to a box
  center. Providers must still perform their own live reachability/IK checks.
- Box IDs must remain stable within the session/map epoch, including across a
  grasp and placement. Ambiguous identity is unknown, never a fresh arbitrary ID
  asserted to be the held box. Site IDs likewise identify stable registrations.
- Every pose reset/re-registration changes `epoch`. Never compare or execute
  geometry across differing `(frame_id, epoch)` values.
- Evidence timestamps are **Unix seconds on a shared clock**, at measurement
  time, not HTTP receipt time. Providers must translate device clocks and reject
  excessive skew. Agent deadlines use a separately injected monotonic clock.
- `ActionOutcome.ts` is the last phase transition or terminal completion time.
  `observed_at` is the fresh status-query heartbeat. Repeated terminal queries
  must preserve `ts`; a long-running unchanged phase is not a stale heartbeat.

### ActionProvider

| Method | Contract |
|---|---|
| `capabilities() -> Capabilities` | Truthfully declare operations, status, cancellation, idempotency, possession, carrying, emergency stop, size/height limits and finite action deadline |
| `submit(request) -> ActionReceipt` | Admit at most one motion; return promptly while execution continues; identical request ID/payload returns the same receipt without another motion |
| `lookup(request_id) -> ActionReceipt \| None` | Recover an acknowledgement lost in transport; `None` means no known receipt, not proof that nothing moved |
| `status(action_id) -> ActionOutcome` | Report identity, phase, effects, motion, error code, independent holding evidence and fresh heartbeat |
| `state() -> ExecutorState` | Independently measured readiness, motion, active action, phase and holding; do not derive possession from command completion |
| `cancel(action_id) -> CancellationReceipt` | Interrupt that action without dropping a load; prevent its routine from issuing later commands |
| `stop() -> CancellationReceipt` | Load-preserving stop of the owned motion pipeline, including when no action receipt was received |

`ActionRequest` carries `job_id`, an idempotency `request_id`, `Step`,
`observation_ref`, frame/epoch, submission time, full `BuildRequirements`, and the
validated `BuildSite`/`BoxObservation` when relevant. This avoids hidden shared
lookup tables between developers. `expires_at` is the latest safe **admission**
time, not the motion duration limit. Reject expired admission before effects;
recheck current localization, reachability, obstacles and target identity before
motion. Do not blindly use the carried box's remembered pre-grasp position.

`FunctionActions` is an optional bridge for ordinary callables. In v2 each
callable takes the **full `ActionRequest`**, not just `Step`, and returns
`FunctionResult(success, phase, error_code, effects_started)`. Supply independent
`read_state` and cooperative `stop` callbacks. Its worker thread does not make an
uninterruptible hardware routine safe: the action owner must implement stopping
and its watchdog. `read_state.phase` exposes progress while the function runs.

Typical running phases are `surveying`/`settling`, `approaching`,
`grasping`/`lifting`, `carrying`, and `lowering`/`releasing`/`retreating`.
A completed placement must report `released`, `retreated` or `completed`.
Known recovery codes are `target_lost_before_pick`,
`pickup_rejected_before_grasp`, `blocked_motion`, and
`placement_rejected_before_release`; they authorize recovery only together with
consistent possession, stopped state and the relevant phase/effects evidence.
Exceptions and transport timeouts are **unknown outcomes**, not safe failures.

### ObservationProvider

| Method | Contract |
|---|---|
| `capabilities() -> PerceptionCapabilities` | v2 agent requires inventory, sites, occupancy, monitoring and images; possession may instead come from the action owner's sensors |
| `observe(site_id=None) -> ObservationSnapshot` | Read the latest immutable snapshot from background sensing, with independent per-item timestamps |
| `find_build_sites(requirements) -> tuple[BuildSite, ...]` | Supply measured candidate frames feasible for the complete requested build, including loaded approach routes |
| `check_build_site(site_id, requirements) -> BuildSite \| None` | Revalidate the selected registration, clearance, support surface and full-build feasibility |
| `monitor(request, outcome) -> MotionObservation` | During motion, assess the supplied request/action/phase against fresh sensing; return `safe=True`, `False`, or `None` plus snapshot and reason |

A snapshot contains stable box IDs, world positions/sizes, `current` and explicit
`eligible` flags, pose, optional independent holding evidence, selected-site
occupancy, and up to three `SceneImage` objects. Inventory may retain remembered
boxes, but only current/fresh eligible boxes can be picked. Protected or ambiguous
objects must not be eligible. `occupancy_complete=True` means every cell in the
requested bounding envelope is represented; an occluded cell is still `unknown`.
Extra obstructions must be included, not filtered out as non-target cells.

`SceneImage` carries `view`, an inline base64 PNG/JPEG `data_url`, capture time,
frame/epoch, `simulated`, and an optional short description. Maximum payload is
512 KiB per image. Remote URLs and local paths are not accepted or fetched.
Supply an overview and optional gripper/build-site views where available. Images
must match the snapshot's frame/epoch and freshness limits. The reasoning backend
sends them as image content, not base64 text in the JSON prompt. Use a vision-capable
OpenAI-compatible model; `--json-only` changes schema formatting, not image support.
Images and text are untrusted scene data; neither can override legal choices or certify a grasp.
Image bytes are not copied into action history or public panel events.

`MotionObservation` must echo request ID, action ID and phase. It is not a generic
scene boolean: assess hazards relevant to the active phase, expected occlusion,
load loss and localization. Unknown/stale monitoring requests a stop; the agent
does not wait for an LLM to decide whether to stop. After an action finishes,
obtain a new settled snapshot. **Placement cell evidence must be measured after
terminal completion**, not merely after request submission or before release.
A disappearing marker is never proof of possession.

### Timing, cancellation and restart rules

Maintain cameras and robot state in background components. `submit`, `lookup`,
`status`, `state`, `observe`, `monitor`, site queries and stop acknowledgements must
return or raise within **250 ms**; do not run an entire motion or a camera startup
inside these calls. Slow work belongs behind cached state/asynchronous workers.
The current inventory-only `ObservationWorker` is not a complete v2 provider.
Live adapters must enforce transport timeouts and own a lower-level watchdog;
Python polling is not a hard-real-time safety system.

The agent refreshes after model latency, monitors each running action without
model calls, and waits up to `observation_timeout` (default one second) for a
post-action frame. `fresh_s` (default two seconds) is an evidence-age ceiling,
**not a safe robot collision-response interval**. Live owners must establish
appropriate tighter bounds for actual speeds, sensors and braking distances.

Cancellation acknowledgement does not establish stopped state. Reconcile action
status and independent executor/holding evidence; keep uncertain or loaded jobs
in `NEEDS_OPERATOR`. On status/monitoring/dispatch errors, request a stop before
returning. Never replay a timed-out motion POST. Keep request records for the
job and across client reconnects; the hardware owner must prevent abandoned
routines from continuing after disconnect or process failure.

This version starts from a **verified empty build site**. It does not infer
completed cells from an old stack or automatically resume a partial build.
Process restart requires reconciliation of outstanding actions, possession and
site contents; restarting Python is not an operator reset. Cleanup is not a
substitute for cancellation or a permission to release a held box.

### Offline agent development and component acceptance

From the repository root, with no robot packages, API key or network:

```bash
python -B -S bot-code/main.py --mock --planner deterministic
```

For targeted stdlib tests, run from `bot-code/tests`:

```bash
python -B -S -m unittest test_agent test_agent_adapters test_agent_backend.BackendTests test_agent_cli test_panel.PanelTests -v
```

A direct test harness uses the exact provider split that live components will use:

```python
from agent import Agent
from contracts import Block, Structure
from mock_agent_world import MockAgentWorld

world = MockAgentWorld(3, faults={"place": ["prerelease"]})
agent = Agent(world.actions, world.observations, backend="deterministic",
              clock=world.clock, sleep=world.sleep)
result = agent.run(Structure([Block(0, 0, 0), Block(1, 0, 0), Block(0, 1, 0)]))
assert result.success
```

The mock executes timed phases using virtual time, moves its simulated base,
tracks grasp/release/placement separately, and supplies synthetic top-down PNGs.
It is a temporary **contract/agent simulator**, not a camera, physics, collision
or calibration test. `action_duration` and `action_timeout` are configurable.
Inject per-operation fault sequences with `faults={operation: [fault, ...]}`:
`pregrasp`, `prerelease`, `blocked`, `unknown_holding`, `bad_placement`,
`postrelease`, `submit_timeout`, `running_forever`, `obstacle`, `pose_loss`,
`stale_observation`, and `lost_load`. `None` in a sequence means a normal action.
`no_sites=True`, `world.stale=True`, and `world.cancel_stops=False` cover missing
sites, stale startup sensing and unconfirmed cancellation. Inspect
`world.requests`, `world.monitor_calls`, `world.cancellations` and `world.placed`.

Before replacing either mock, the component owner must demonstrate:

1. Correct units, frames, epochs, evidence timestamps and immutable snapshots.
2. Idempotent admission, rejection of changed duplicate payloads and receipt lookup.
3. Fresh running heartbeats with an unchanged terminal completion timestamp.
4. Bounded calls and load-preserving cancellation that prevents later commands.
5. Truthful unknown possession/occupancy, including occlusion and sensor loss.
6. Phase-aware monitoring that detects hazards while the action is still running.
7. Current camera images plus independently verified post-action evidence.
8. Passing the focused agent tests through the replacement provider; no hardware
   or paid API calls belong in normal test runs. Supervised live acceptance is separate.

`--mock` never calls a model by default. The panel's builds never call one at all. Model calls with simulated tools
require `--allow-api-with-mock`; the panel's existing `--ui` mode is explicitly
real-model/simulated-tools. Offline tests inject fake reasoners/SDK clients.

## mc_skills — body server (:8006)

```bash
uv run ~/bbapps/mc_skills/main.py          # real: claims writers at boot
MOCK=1 uv run ~/bbapps/mc_skills/main.py   # same API, no bbos — laptops
```

Endpoints: `GET /health` `/state`, POST `/home` `/park`
`/goto{pos,quat?,duration}` `/pick{pos}` `/place{pos}` `/gripper{open}`
`/rotate{rad}` `/say{text}` `/celebrate`. The deployed copy also has
`/voice_script{lines}`. There is **no `/drive` endpoint** — it was only ever
proposed; base motion goes through `/rotate` or a `drive.ctrl` writer.
Responses `{ok, result|error}`; serialized by a lock — concurrent calls get
`busy`, except `/say` and `/voice_script`, which do not take the motion lock.
The repo copy at `voice/mc_skills_main.py` and the deployed
`~/bbapps/mc_skills/main.py` differ; diff them before assuming a change is live.

pick = approach +APPROACH_H → descend → grip → lift; place = mirror. TUNE on
robot: `grip_open`/`grip_closed` direction (test `POST /gripper`), `DOWN_QUAT`
top-down orientation, `APPROACH_H`. Add endpoints, don't rename.

## Structure input — the Minecraft side

- `panel.StructureReceiver` receives EOF-framed wire payloads on TCP :5005 in
  `--ui` mode. The panel keeps the latest valid design and freezes an active job's
  design. Empty scans do not connect; silence is not an empty structure.
- `structure_src.load_structure(path)` loads `.json` or `.nbt` (nbtlib).
- The separate legacy click-grid writes `fixtures/structure.json`.

## planner.py — explicit legacy one-shot mode

`plan_build(structure, boxes, backend="auto")` belongs to `--mode oneshot`.
It is not the recovery path beneath the agent and does not provide mobile
execution or the v2 verification guarantees. Agent fallback selects deterministic
legal steps inside the same validated, monitored loop.

## Orchestrator — main.py

Agent is the default mode. It validates the job, checks provider capabilities and
readiness, and runs the feedback loop. Live execution requires both an explicit
`--provider module:factory` and `--box-size` in meters. Hardware preparation and
calibration belong to the action owner, outside model tools. Agent exit status is
zero only for verified completion.

From the repository root:

```bash
python -B -S bot-code/main.py --mock --planner deterministic
```

## Parallel-dev rules (the point of this layout)

1. **One owner per hardware topic, and it is never the reasoning layer.** For the
   legacy path that owner is `mc_skills`; `actions/pickup.py` and any future
   interface-v2 action provider claim `arm_*.ctrl` / `arm_*.torque` directly, so
   only one of them may run at a time. Reasoning code never opens a Writer.
2. Readers are fair game — any process, any time.
3. `contracts.py` frozen; `main.py` is the merge point — both need owner
   sign-off for edits. New functionality = new file or your own file.
4. Self-contained modules: types you need but can't put in contracts yet live
   in YOUR file with a "candidate for contracts.py" note (see perception.py).
5. Every module runs robot-free: `--mock`/`MockSkills`/fixtures. Don't block
   on robot time — it's serialized; coordinate in chat and use the running
   mc_skills server.

## Runbook

```bash
# body — leave running. Claims the hardware writers at boot; check first that
# nothing else owns them, and see "Parallel-dev rules" below.
uv run ~/bbapps/mc_skills/main.py
# perception debug (safe alongside body)
uv run ~/crafter/bot-code/perception.py --viz        # :8007
# pipeline
cd ~/crafter/bot-code && uv run main.py --mock       # then real args
```

`POST /home` is **not** part of that starter block on purpose: it torques the arms
and runs a staged homing trajectory, so it is real motion. Run it deliberately,
with the arms clear and their owner present — never chained onto a server start.

## Tune-before-demo checklist

- [ ] `POST /gripper` direction; `DOWN_QUAT`; `APPROACH_H`
- [ ] `BOX_SIZE`, anchor marker placement at cell (0,0)
- [ ] `camera.points`↔rgb alignment vs tape measure
- [ ] ArUco printed: boxes 0..N-1 + anchor 49
- [ ] receiver listening on :5005 before the mod right-click
- [ ] `OPENAI_API_KEY` if llm/agent backend used

## Fallback ladder — move down, don't get stuck

- structure: mod TCP → grid UI → fixture json
- detection: calibrated live provider; unavailable evidence stops the job. Fixtures are mock-only.
- reasoning: model → deterministic choices in the same agent loop, never blind one-shot execution.
- motion: action-owner implementations must retain the same verification/stop contract; no automatic raw-motion fallback.
