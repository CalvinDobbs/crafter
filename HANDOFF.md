# Crafter handoff

Current checkpoint: GitHub `main` at `d217bfb` or later. This handoff covers the reasoning/perception integration, documentation audit, panel/key work, speaker research, and motion/arm research begun on 2026-09-12.

## Work completed

### Reasoning agent and live perception

Commits:

- `93f05c2` — Give reasoning live perception and persistent world awareness
- `874b43c` — Wait for live camera frames before requesting reasoning

`bot-code/agent_adapters.py` now provides `PerceptionObservations`, a read-only adapter for an existing `perception.py --viz` process. It reads `/scan` and `/frame?view=rect`; it does not import the hardware perception module, open BBOS readers/writers, or invoke robot actions.

The adapter supplies the reasoner with:

- Pose and map epoch.
- Persistent box tracks with measurement times and current/remembered state.
- Loose, protected, and unknown classifications.
- Oriented build-anchor registration in base-at-capture and session-world frames.
- Observed surface cells plus coverage and height summaries.
- Markerless detector objects with identity, confidence, depth, and position uncertainty.
- Sensor, stream, detector, and calibration diagnostics.
- A timestamp-checked real rectified camera image.

The world model is stored as bounded immutable JSON in `ObservationSnapshot.world_model_json`; `snapshot.world_model` returns a detached dictionary. The controller includes it in the model context. Large surface maps are reduced within a 32 KiB budget, and `coverage` reports omitted records. Blank or omitted cells remain unknown, never free.

The adapter intentionally advertises only inventory and images. It does not claim build-site feasibility, complete occupancy, possession, or phase-aware motion monitoring. Consequently, the normal live build agent still fails preflight until independently developed action/perception providers implement those contracts. Existing action admission, cancellation, possession, and placement checks were not weakened.

### Read-only inspection CLI

Run against an already-running perception server:

```bash
python -B -S bot-code/main.py --observe-perception http://127.0.0.1:8007
```

This reports live structured perception and image metadata without image bytes. It does not load an action provider or call a model. The URL is relative to the machine running the command; a developer PC needs an SSH forward to the bot's loopback port 8007.

One explicitly requested read-only model decision can be attempted with:

```bash
uv run --offline bot-code/main.py \
  --observe-perception http://127.0.0.1:8007 \
  --planner llm
```

That mode allows only `observe` and `stop`, never actions. It now waits up to four seconds for an on-demand live image and refuses to spend a model request if no image arrives.

### Live verification on `bracketbot-184`

The integration was tested with live cameras and the TensorRT FP16 box detector. No robot actions were invoked.

Continuous sample test:

- 40/40 post-warm-up observations were valid and accepted by the agent's snapshot validator.
- 24 distinct camera/map captures were observed.
- Median observation read latency: 0.99 ms; p95: 2.85 ms.
- Median image age: approximately 651 ms; p95: approximately 992 ms.
- Camera capture rate: approximately 9.1 FPS.
- Detector rate: approximately 8.0 FPS.
- World-model payloads stayed within the configured budget.
- Unknown possession, occupancy, and site safety remained unknown.

Cold-start CLI test after `874b43c`:

- 3/3 runs passed when the rectified image stream was initially inactive.
- Completion times were 0.70–0.77 seconds.
- Every run returned one real, non-simulated image.
- Image ages were approximately 431–454 ms.
- Each run returned valid pose/world data containing 543–590 observed surface cells.
- Output omitted image bytes, `execution_enabled` was false, and `decision` was null.
- Model requests: 0. Action requests: 0.

The deployed bot checkout reached `874b43c` while preserving uncommitted `bot-code/perception.py` work and `bot-code/summaryofperception.md`. The temporary perception service on 127.0.0.1:8007 was stopped after testing. The existing panel remained available on 127.0.0.1:8005, and its receiver was listening on TCP 5005.

### Model request result

One user-approved real `gpt-4o-mini` request was attempted with:

- One real image.
- Approximately 28 KiB of world-model JSON.
- 721 total observed surface cells, 64 included detailed samples.
- 13 detector objects.
- Only `observe` and `stop` choices.

The request failed with `APIConnectionError`; no model decision was returned. A later non-billable diagnosis confirmed DNS resolution, TCP 443, TLS 1.3, and an unauthenticated OpenAI endpoint response through the SDK transport. The exact transient cause was not established. Do not claim successful live reasoning from this test. No second paid request was made.

### Model key and panel access

The real OpenAI key is stored outside the repository on the bot at:

```text
/home/bracketbot/.config/crafter/openai_api_key
```

Last verified metadata:

- owner: `bracketbot`
- file mode: `600`
- parent directory mode: `700`
- the key loaded successfully without displaying its value

Never print, copy into chat, commit, or retrieve the key. `OPENAI_API_KEY` overrides the saved file when present. GitHub Actions secrets cannot be downloaded for local development. Developers can use the bot-hosted panel through SSH forwarding without copying the key to their PCs. See `README.md`.

Relevant commit:

- `27e2cd7` — Keep shared model credentials on the bot

### Clear blueprint control

The main panel has a Clear blueprint button. It:

- clears the latest blueprint and 3D preview;
- disables Start until a new design arrives;
- preserves model configuration;
- does not cancel or mutate an active build;
- rejects stale browser requests that target an older design;
- prevents a scan started before clearing from restoring the cleared design.

At implementation time, 25 panel tests passed on the bot, including QuickJS checks. No paid model or hardware calls were made by those tests.

Relevant commits:

- `63693b3` — Let users clear a blueprint without disrupting builds
- `d3eb341` — merge preserving concurrent voice work

An eight-box Minecraft design was also observed completing through the real model and simulated tools before a panel restart: 8/8 boxes, 35 model calls, and 33 simulated tool calls. That was an observed integration run, not a deterministic regression test. The restart cleared its in-memory history with user approval.

The frontend has since moved from loose `bot-code/panel.{html,css,js}` files to `bot-code/web/`, with the packaged logo under `bot-code/web/assets/`. The current panel uses a light theme with warm red accents. Use the current paths.

### Speaker playback guide

Added `bot-code/voice/speaker_playback.md`.

Verified from the bot's `bbapps`, BBOS speaker daemon, and deployed `mc_skills`:

- publish to `speaker.audio` with `Type("speaker_audio")`;
- payload is signed PCM16, mono, 16 kHz;
- each publication is shaped `(1600, 1)`, representing 100 ms;
- the speaker daemon applies gain/compression and resamples to the 48 kHz hardware path;
- there must be one owner of the speaker writer;
- integrated speech should use the existing body-server queue rather than opening a competing writer;
- the deployed `/say` endpoint is non-blocking, but HTTP success only acknowledges enqueueing;
- `bbapps/play_sound/main.py` is the minimal WAV sample;
- `bbapps/examples/view_speaker.py` writes synthesized chords despite its name;
- physical audibility was not tested.

Guide validation:

- one Python example parsed;
- six Bash examples passed syntax checks;
- ten local links resolved;
- a bundled WAV was verified as mono, 16 kHz, PCM16;
- no sound, TTS, or hardware service was triggered.

Relevant commit:

- `2e87d3a` — Document speaker playback without competing hardware writers

## Verification commands

Focused portable checks from `bot-code/tests`:

```bash
python -B -S -m unittest \
  test_agent \
  test_agent_adapters \
  test_agent_backend.BackendTests \
  test_agent_cli \
  test_panel.PanelTests -v
```

Relevant results during this work:

- 103 focused tests passed after the initial perception integration.
- 93 reasoning/perception/CLI tests passed after the camera warm-up fix.
- 11 `test_agent_cli` tests passed on the bot after deployment.
- 33 focused backend/panel/CLI tests passed during key setup.
- 25 panel tests passed during the Clear blueprint deployment.

These are historical checkpoint results. Other developers have since changed reasoning, perception, voice, and panel code. Use the current commands in `AGENTS.md` for fresh verification. Routine tests must not make paid model calls or move hardware.

## Important constraints

- Never combine real actions with mocked possession, occupancy, or safety evidence.
- Do not start another hardware writer. The live action provider must use the designated action owner and implement interface v2 truthfully.
- Do not infer possession from a successful action or a disappearing marker.
- Placement evidence must be measured after terminal action completion.
- `ActionOutcome.ts` is immutable transition/completion time; `observed_at` is the heartbeat.
- Perception provider cleanup closes sensing/transport resources only. It must not park, release, or move a loaded robot.
- `contracts.py` and `WIRE_FORMAT.md` remain frozen.
- Preserve uncommitted perception work on the bot. Always inspect the remote tree and services before updating or launching anything.
- The main panel uses a light theme with warm red accents. Preserve native light controls, canvas colors, and viewport containment.

## Current unfinished task: base movement and arm guide

The user requested a Markdown guide in `bot-code/actions/` explaining how to move the base and use the arms, based on the bot's sample apps and `actions/pickup.py`.

No guide file has been created yet. Intended destination:

```text
bot-code/actions/motion_and_arms.md
```

The source review is substantially complete. Write the guide from the findings below, validate links and snippets without running motion, and commit only that guide.

### Sources reviewed on the bot

- `/home/bracketbot/bbapps/AGENTS.md`
- `/home/bracketbot/bbapps/teleop.py`
- `/home/bracketbot/bbapps/examples/view_arms.py`
- `/home/bracketbot/bbapps/examples/view_ik.py`
- `/home/bracketbot/bbapps/quest_teleop/main.py`
- `/home/bracketbot/bbapps/quest_teleop/scripts/homing.py`
- `/home/bracketbot/bbapps/nav/main.py`
- `/home/bracketbot/bbapps/mc_skills/main.py`
- `/home/bracketbot/crafter/bot-code/actions/pickup.py`
- `/home/bracketbot/bbos/bbos/daemons/base/constants.py`
- `/home/bracketbot/bbos/bbos/daemons/base/daemon.py`
- `/home/bracketbot/bbos/bbos/daemons/arm_left/constants.py`
- `/home/bracketbot/bbos/bbos/daemons/arm_right/constants.py`
- `/home/bracketbot/bbos/bbos/daemons/arm_left/daemon.py`

The local and bot copies of `actions/pickup.py` matched during review.

### Base movement findings

Core topic:

```python
Writer("drive.ctrl", Type("drive_ctrl"))
```

Publish:

```python
buffer["twist"] = np.array([v, w], dtype=np.float32)
```

Conventions and limits:

- `v` is forward linear velocity in meters/second.
- `w` is yaw rate in radians/second.
- Positive `v` is forward.
- Positive `w` is intended as CCW/left, but `mc_skills.rotate()` still says to verify the sign on the robot.
- The base daemon clamps `v` to ±0.3 m/s and `w` to ±1.0 rad/s.
- `Config("drive").max_angular_vel` was 0.9 rad/s.
- `nav/main.py` uses much lower autonomous values, approximately 0.08 m/s and 0.15 rad/s.

Safety and timing:

- The baseboard owns the 200 Hz balance loop; apps publish velocity commands.
- `drive.ctrl` is a 10 ms topic.
- The base command timeout is 0.1 seconds. Missing or non-finite commands become zero.
- Publish continuously faster than 10 Hz; the host configuration uses 50 Hz.
- Explicitly publish `[0, 0]` during cancellation/teardown rather than relying only on timeout.
- There can be only one `drive.ctrl` writer. `teleop.py`, `nav/main.py`, Quest teleop, and `mc_skills` compete for it.
- `teleop.py` converts left/right wheel velocity into body twist using half the robot width.
- `nav/main.py` is the autonomy reference: SLAM/map-frame control, smoothing, obstacle clearance, bounds checks, stuck detection, and a 0.4-second manual dead-man timer.
- A field named `twist` does not require base `MODE_TWIST`.
- `MODE_BALANCE=0` is normal balancing mode.
- `MODE_LEAN=1` is specialized.
- `MODE_TWIST=2` is explicitly labeled “velocity, NOT balancing.” Do not select it in a generic action provider without platform-owner approval.
- The base daemon falls back to its configured default mode when mode commands expire.

`mc_skills.rotate(rad)` is an open-loop scan primitive: publish `[0, w]` for a computed duration, then zero. It is not localization-grade navigation.

### Arm control findings

Per-arm topics:

- `<arm>.state`: eight-element position, velocity, torque, temperature, and current arrays
- `<arm>.ctrl`: `pos[8]`, `vel[8]`, `tau[8]`, and `alpha`
- `<arm>.torque`: enable flags, torque-mode flags, compliance, homing, and calibration state
- J0 is the vertical stage.
- J7/index 7 is the gripper.
- `<arm>.ctrl.pos` uses motor turns, not URDF radians.

Ownership and enable sequence:

- Use one owner for each `<arm>.ctrl` and `<arm>.torque` topic.
- Read-only tools must not open writers. `examples/view_arms.py` opens them only with `--control`.
- Before enabling torque: disable first, wait for fresh state, copy the live pose into `ctrl`, flush it, then enable. The OFF-to-ON transition reseeds the daemon command filter and prevents jumps.
- Use `staged_home_arms` and `park_arms` from `bbapps/quest_teleop/scripts/homing.py`; do not casually duplicate or simplify homing.
- The arm daemon disables torque when its control writer disappears.
- The daemon clips commands near live state and into calibrated ranges, but applications must still generate smooth bounded trajectories.

Cartesian IK pattern used by `mc_skills`:

1. Read live motor turns.
2. Convert with `cfg.q2urdf(live)`.
3. Reset IK from the first seven URDF joints.
4. Call `cfg.ik.solve(position_xyz, quaternion_xyzw)`.
5. Insert the seven-joint result into the full URDF vector.
6. Convert back with `cfg.urdf2q(full)`.
7. Preserve the live J7/gripper setpoint.
8. Smoothly ramp the complete motor-turn target.

Coordinates and mirroring:

- `mc_skills.Arm.goto()` expects base-frame XYZ meters and XYZW quaternion.
- Base frame: +x forward, +y left, +z up.
- The agent uses world positions with frame/epoch metadata. The action provider must transform into the current base frame and reject stale/cross-epoch geometry before IK.
- Left and right arms have different `ik_sign` arrays.
- The left gripper uses `gripper_sign=-1`; the right uses `gripper_sign=1`.
- Use each arm's `Config` conversion methods instead of manually negating motor commands.
- Load each robot's calibration from `ranges.calibration.json`; do not copy limits between robots.
- Raw joints, velocities, homing, and arbitrary IK targets must remain behind semantic action operations, not model-visible tools.

### `actions/pickup.py` findings

This is a direct joint-space two-arm cage/lift prototype, not navigation or a complete `ActionProvider`.

Sequence:

1. move J3/elbow to 90 degrees;
2. open J7 grippers;
3. raise J0;
4. spread both arms with J2;
5. lower J0 near the calibrated bottom;
6. optionally pinch with J2 until tracking error suggests contact;
7. hook with an FK-derived J5/J6 blend;
8. close grippers;
9. cradle with more elbow flex;
10. lift J0 and hold.

Properties and limitations:

- It derives inward/outward signs with FK instead of hard-coding mirror signs.
- It loads per-robot calibration and keeps a range margin.
- It self-paces at 200 Hz using `keeptime=False`.
- Contact uses command-versus-live tracking error and bounded travel.
- Contact thresholds and squeeze values are hardware-tuning assumptions, not independent possession evidence.
- Default behavior lowers and holds; `--pickup` opts into cage/lift.
- Torque remains enabled while holding.
- Its `finally` block disables torque and closes writers. If carrying a box, that releases the load.

Do not call `pickup.py` from the reasoning layer and do not use its cleanup as agent `cancel()`/`stop()`. Interface v2 requires a load-preserving stop and truthful independent possession state.

### Action-provider requirements

The implementation must satisfy interface v2 in `bot-code/agent_types.py` and `bot-code/README.md`.

Semantic operations:

- `look_around`
- `approach_box`
- `pickup`
- `move_to_build`
- `place`

Lifecycle requirements:

- idempotent admission and receipt lookup;
- bounded `submit`, `status`, `state`, `cancel`, and `stop`;
- live phase reporting;
- independent possession/readiness evidence;
- no blind replay after unknown/timed-out submission;
- load-preserving cancellation and stop;
- immediate rechecks of localization, obstacles, identity, reachability, frame, and epoch before effects;
- successful command completion is not proof of possession or placement.

`FunctionActions` can adapt ordinary functions, but it does not make a blocking or uninterruptible hardware routine safe. Cooperative cancellation and watchdog behavior remain the action owner's responsibility.

Do not run `pickup.py`, `/home`, `/goto`, `/pick`, `/place`, `/rotate`, or a drive publisher while verifying the guide. Do not include an unguarded copy-paste command that can unexpectedly move the robot. Do not resolve writer conflicts by killing owners or restarting BBOS services.

## Current documentation audit

The user asked to check outdated Markdown files and update or delete them. The audit was started but paused for this handoff. No documentation cleanup edits have been made yet.

Markdown files found:

- `AGENTS.md`
- `README.md`
- `PLAN.md`
- `LLM_REASONING_PLAN.md`
- `WIRE_FORMAT.md`
- `bot-code/README.md`
- `bot-code/voice/command_map.md`
- `bot-code/voice/speaker_playback.md`

### Confirmed stale documents

#### `PLAN.md`

This plan has extensive obsolete current-state claims:

- It says no process listens on TCP 5005; `panel.StructureReceiver` now does.
- It says `structure_src` does not consume the wire format; the panel now parses and validates it directly.
- It describes the stateful reasoning agent, receiver, voice queue, and personality work as future phases even though much of that exists.
- Its voice section says personality is not chosen and `/say` is a print stub; current voice code has a selected pack and non-blocking speech queue with OpenAI TTS/cache support.
- It presents the legacy one-shot planner/body-server pipeline as primary instead of interface-v2 providers and the default agent loop.

Recommended action: replace stale current-state/phase details with a concise roadmap linked to `bot-code/README.md`, or explicitly label the document historical.

#### `LLM_REASONING_PLAN.md`

This pre-implementation checkpoint incorrectly says the reasoning framework, typed interfaces, mock world, validation, recovery, and default agent mode are only proposed. Those are implemented in `agent.py`, `agent_types.py`, `agent_backend.py`, `agent_adapters.py`, `mock_agent_world.py`, and `main.py`.

Recommended action: reduce it to a superseded checkpoint linked to current docs, or delete it after confirming historical context is not needed.

#### `bot-code/voice/command_map.md`

Confirmed stale statements:

- `/say` is described as a stub, but local/deployed code uses `hook.say_text` and `SpeechQueue`.
- It references numbered approval sections that no longer match the current plan.
- It says `pick.descend` and `place.descend` templates are absent, but both exist.
- The route count is inconsistent with the implementation.
- It does not clearly distinguish the default agent from legacy one-shot narration.

Update it against both the current repository copy and deployed `~/bbapps/mc_skills`; do not assume they are identical.

### Documents needing smaller corrections

- `README.md`: point to current interface-v2 status rather than treating `PLAN.md` as authoritative.
- `bot-code/README.md`: update writer ownership language, legacy grid UI port wording, endpoint list, unsafe auto-home runbook, current perception status, and parallel-development rules.
- `bot-code/voice/speaker_playback.md`: mostly current; preserve the warning that physical audibility is unverified.

### Documents to preserve

- `WIRE_FORMAT.md`: frozen protocol specification; sampled claims still match scanner behavior.
- `AGENTS.md`: current project rules; preserve SSH, safety, test, panel, and theme guidance.

## Operational state and access

Bot: `bracketbot-184` at `100.66.148.86`.

SSH from the Windows PC:

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -i ~/.config/devin/crafter-ssh/id_ed25519 \
  bracketbot@100.66.148.86 '<command>'
```

SSH was intermittent. Use bounded options for scripted reads and retry timeouts rather than changing SSH configuration. Non-interactive sessions do not put `uv` on `PATH`; use `/home/bracketbot/.local/bin/uv`.

At the last handoff status check:

- panel: `127.0.0.1:8005`
- Minecraft receiver: `0.0.0.0:5005`
- panel state: receiver listening, model ready, simulation enabled, main view
- no TCP 8006 listener was present at that specific check; earlier checks found `mc_skills`, so always re-inspect
- bot working tree still had modified `bot-code/perception.py` and untracked `bot-code/summaryofperception.md`

Do not overwrite those bot-side files.

## Deferred browser task

A real-browser cancellation check remains pending:

1. send or load a blueprint;
2. start a real-model/simulated-tool build;
3. cancel during a model request and confirm immediate return to main;
4. confirm `worker_busy` remains true only until the in-flight request returns;
5. confirm late results cannot restart or complete the cancelled build;
6. confirm another build can start afterward;
7. recheck Clear blueprint in the current light-theme frontend.

This uses real, billable model calls. Do not run it as an automatic regression test.

## Recommended next steps

1. Create `bot-code/actions/motion_and_arms.md` from the completed research in this handoff.
2. Validate Markdown links and Python/Bash snippets without executing motion.
3. Stage and commit only that guide; inspect for concurrent changes first.
4. Continue the documentation audit, starting with root `README.md` and `bot-code/README.md`.
5. Re-inspect the bot working tree and writer owners before any deployment or physical test.
6. Coordinate the first physical base, arm, pickup, speaker, and cancellation checks with the robot/action owner.

## Key files

- `bot-code/agent.py` — verified agent loop and model context
- `bot-code/agent_types.py` — interface-v2 contracts and bounded world model
- `bot-code/agent_adapters.py` — provider loading, action bridge, and live perception adapter
- `bot-code/agent_backend.py` — strict model schema, image transport, and world-model instructions
- `bot-code/main.py` — default agent, read-only perception CLI, and explicit legacy mode
- `bot-code/perception.py` — independently owned live sensing/world model
- `bot-code/panel.py` — Minecraft receiver and simulated-tools panel backend
- `bot-code/web/` — current panel frontend and packaged assets
- `bot-code/actions/pickup.py` — direct joint-space pickup prototype
- `bot-code/voice/speaker_playback.md` — source-checked speaker guide
- `bot-code/README.md` — current interface/runbook reference, pending audit corrections
- `AGENTS.md` — operational rules and verified commands
