# Crafter development handoff

Last updated: 2026-09-12

## Current state

- Local checkout: clean on `main`, synchronized with `origin/main` before this handoff was created.
- Current local HEAD before this handoff: `d217bfb` (`Unify the panel around a light red-accented theme`).
- Bot: `bracketbot-184` at `100.66.148.86`.
- Main panel is running on the bot:
  - Browser service: `127.0.0.1:8005`
  - Minecraft receiver: `0.0.0.0:5005`
  - `/api/state` was reachable through the existing SSH forward.
  - `receiver.listening=true`, `receiver.port=5005`, `llm_ready=true`, `simulation=true`, `view=main`.
- No process was listening on TCP 8006 at the last status check in this session. Earlier checks found `mc_skills` on 8006, so re-inspect rather than assuming it is running.
- The bot working tree contains concurrent perception work:
  - modified `bot-code/perception.py`
  - untracked `bot-code/summaryofperception.md`
- Do not reset, overwrite, stage, or commit that bot-side perception work.

## SSH access

From the Windows PC, use the dedicated identity:

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -i ~/.config/devin/crafter-ssh/id_ed25519 \
  bracketbot@100.66.148.86 '<command>'
```

SSH is intermittent. Several connections timed out during banner exchange, while retries succeeded. Use bounded connection and server-alive options for scripted reads:

```bash
timeout 25s ssh -n -T \
  -o BatchMode=yes \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=yes \
  -o ConnectTimeout=8 \
  -o ServerAliveInterval=5 \
  -o ServerAliveCountMax=2 \
  -i ~/.config/devin/crafter-ssh/id_ed25519 \
  bracketbot@100.66.148.86 '<read-only command>'
```

Non-interactive SSH does not put `uv` on `PATH`. Use `/home/bracketbot/.local/bin/uv` explicitly.

## Model key and panel

The real OpenAI key is stored outside the repository on the bot:

```text
/home/bracketbot/.config/crafter/openai_api_key
```

Last verified metadata:

- owner: `bracketbot`
- file mode: `600`
- parent directory mode: `700`
- the key loaded successfully without displaying its value

Never print, copy into chat, commit, or retrieve the key. `OPENAI_API_KEY` overrides the saved file when present. See [README.md](README.md) for hidden-input setup and SSH-forwarded panel access.

The main panel deliberately makes real model calls but uses simulated physical tools. Nothing starts until Start is clicked. Before sharing a preview, verify `/api/state` reports TCP receiver 5005 as listening. Do not launch a second receiver on alternate ports and expect the default Minecraft mod to reach it.

## Work completed in this session

### Secure key setup and developer access

- Confirmed GitHub Actions secrets are write-only after creation and cannot be retrieved for ordinary local development.
- Chose bot-hosted secret storage instead of pretending GitHub could distribute the key.
- Saved and verified the key on the bot without exposing it.
- Documented bot-hosted panel access and secret handling.
- Commit: `27e2cd7` (`Keep shared model credentials on the bot`).

### Clear blueprint control

Added a Clear blueprint button to the main screen.

Behavior:

- clears the latest blueprint and preview;
- disables Start until another design arrives;
- preserves the model configuration;
- does not cancel or mutate an active build;
- rejects a stale browser request so it cannot clear a newer design;
- prevents a scan that began before the clear operation from restoring the cleared design.

Verification at implementation time:

- 25 panel tests passed on the bot, including QuickJS checks;
- the updated button and POST-only endpoint were present after restart;
- no paid model call or hardware call was made by those tests.

Commits:

- `63693b3` (`Let users clear a blueprint without disrupting builds`)
- merge preserving concurrent voice work: `d3eb341`

The frontend has since moved from `bot-code/panel.{html,css,js}` to `bot-code/web/`, with the packaged logo under `bot-code/web/assets/`. The current panel uses a light theme with warm red accents. Use the current paths, not the old locations.

### Panel and Minecraft receiver work

Minecraft blueprint delivery was diagnosed end to end. The mod sends EOF-framed UTF-8 JSON to `100.66.148.86:5005`. The apparent loading failure came from viewing an isolated preview that listened on TCP 5015 instead of the main panel receiver on TCP 5005. Use `http://127.0.0.1:8005/` through the main SSH forward, and always confirm `/api/state` reports `receiver.listening=true` and `receiver.port=5005`.

A fragmented five-block payload was sent from the PC to the bot and verified in the panel without starting a build. Receiver tests cover fragmented reads, EOF framing, stale overlapping scans, invalid-message handling, and clear-during-in-flight behavior.

UI commits completed during this work:

- `e4aa925` — viewport-contained layout with internal scrolling for bounded content
- `1b19856` — removed the Latest design and The demo setup cards; expanded the preview
- `603ecf9` — moved frontend files to `bot-code/web/` and installed `Crafter-transparent.svg`
- `d217bfb` — converted the complete panel and canvas renderer to a light theme with brick-red accent `#b84946`

The Chromium regression exercises 80 combinations of viewport and screen state. It checks page overflow, clipped controls, canvas visibility, logo loading, light native controls, light canvas/input backgrounds, the full-width design preview, and the bounded activity feed. Run it only against an isolated unsigned browser profile as documented in `AGENTS.md`.

### Real Minecraft/model simulation observation

Before the panel restart, an eight-box Minecraft design completed successfully through the real model and simulated actions:

- source: Minecraft
- required/placed: 8/8
- model calls: 35
- simulated tool calls: 33

This was an observed successful flow, not a deterministic regression test. The restart cleared its in-memory history as explicitly approved.

### Speaker playback research and guide

Added [bot-code/voice/speaker_playback.md](bot-code/voice/speaker_playback.md).

Verified from the bot’s `bbapps`, BBOS speaker daemon, and deployed `mc_skills`:

- publish to `speaker.audio` with `Type("speaker_audio")`;
- payload is signed PCM16, mono, 16 kHz;
- each publication is shaped `(1600, 1)`, representing 100 ms;
- the speaker daemon applies gain/compression and resamples to the 48 kHz hardware path;
- there must be one owner of the speaker writer;
- integrated speech should use the existing body-server queue rather than opening a competing writer;
- the deployed `/say` endpoint is non-blocking, but an HTTP success only acknowledges enqueueing;
- `bbapps/play_sound/main.py` is the minimal WAV sample;
- `bbapps/examples/view_speaker.py` writes synthesized chords despite its `view_` name;
- physical audibility was not tested.

Validation:

- one Python example parsed;
- six Bash examples passed syntax checks;
- ten local Markdown links resolved;
- a bundled WAV was verified as mono, 16 kHz, PCM16;
- no sound, TTS, or hardware service was triggered.

Commit: `2e87d3a` (`Document speaker playback without competing hardware writers`).

### Concurrent work merged or observed

Other developers added substantial reasoning, voice, perception, and panel updates during this session. Notable current commits include:

- `93f05c2` — persistent world awareness and live perception
- `603ecf9` — panel frontend assets moved under `bot-code/web/`
- `874b43c` — wait for a live camera frame before model reasoning
- `d217bfb` — light panel theme with warm red accents
- `659490b` — OpenAI voice/TTS changes

Do not use older path assumptions from earlier commits without checking current files.

### Reasoning and live-perception integration

Commits:

- `93f05c2` — persistent world awareness and live perception
- `874b43c` — camera warm-up before read-only model reasoning

`bot-code/agent_adapters.py` now provides `PerceptionObservations`, a read-only adapter for an existing `perception.py --viz` process. It consumes `/scan` and `/frame?view=rect` without importing hardware perception, opening BBOS readers/writers, or invoking actions. It carries pose/epoch, persistent tracks, loose/protected/unknown classifications, build-anchor registration, observed surfaces, detector uncertainty, sensor diagnostics, and a timestamp-checked real image into the reasoner's context.

`ObservationSnapshot.world_model_json` is immutable and limited to 32 KiB. Large maps report included/omitted coverage; blank and omitted cells remain unknown. The live adapter advertises only inventory and images. It does not claim site feasibility, complete occupancy, possession, or phase-aware monitoring, so full live execution still fails preflight until the independently developed providers implement those capabilities.

Read-only inspection, with no model or action provider:

```bash
python -B -S bot-code/main.py --observe-perception http://127.0.0.1:8007
```

An explicit `--planner llm` permits one read-only `observe`/`stop` decision and can incur a model charge. The CLI waits for an on-demand live image and refuses to send a model request if no image arrives.

Live TensorRT FP16 verification on `bracketbot-184` used no robot actions:

- 40/40 post-warm-up observations passed the agent snapshot validator across 24 distinct captures.
- Median observation-read latency was 0.99 ms; p95 was 2.85 ms.
- Median image age was approximately 651 ms; p95 was approximately 992 ms.
- Camera and detector rates were approximately 9.1 and 8.0 FPS.
- Three cold-start CLI runs completed in 0.70–0.77 seconds and each returned one real image plus 543–590 observed surface cells.
- Unknown possession, occupancy, and site safety remained unknown.

One user-approved `gpt-4o-mini` request was attempted with a real image and approximately 28 KiB of world-model JSON. It failed with `APIConnectionError`; no model decision was returned. Later non-billable probes confirmed DNS, TCP 443, TLS 1.3, and API endpoint reachability, but the transient cause was not established. No second paid request was made; do not claim that live model reasoning passed.

## Current unfinished task: motion and arm guide

The user asked for a Markdown guide in `bot-code/actions/` explaining how to move the base and use the arms, based on sample apps and `actions/pickup.py`.

No guide file has been created yet. The intended destination is:

```text
bot-code/actions/motion_and_arms.md
```

The source review is substantially complete. Continue by writing the guide from the findings below, then validate links and code-block syntax. Do not run motion commands as part of documentation verification.

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

Temporary read-only copies were downloaded under a Windows temp directory during research. Do not depend on that temporary directory in committed documentation; cite the stable bot paths above.

### Base movement findings

Core control topic:

```python
Writer("drive.ctrl", Type("drive_ctrl"))
```

Write:

```python
buffer["twist"] = np.array([v, w], dtype=np.float32)
```

Conventions:

- `v`: forward linear velocity in meters/second
- `w`: yaw rate in radians/second
- positive `v`: forward
- positive `w`: intended CCW/left turn, but the existing `mc_skills.rotate()` comment still says to verify the sign on the robot
- deployed limits found in configuration:
  - base clamps `v` to ±0.3 m/s
  - base clamps `w` to ±1.0 rad/s
  - `Config("drive").max_angular_vel` was 0.9 rad/s
- use lower action-level limits; `nav/main.py` used approximately 0.08 m/s and 0.15 rad/s for autonomous navigation

Safety and timing:

- The baseboard owns the 200 Hz balance loop; apps publish velocity commands.
- `drive.ctrl` is a 10 ms topic.
- The base daemon’s command timeout is 0.1 seconds. If commands stop or become non-finite, it commands zero.
- Publish continuously faster than 10 Hz; the configuration uses 50 Hz host commands.
- Explicitly send `[0, 0]` on cancellation and teardown rather than relying only on timeout.
- Use one writer. `teleop.py`, `nav/main.py`, Quest teleop, and `mc_skills` all compete for `drive.ctrl`.
- `teleop.py` converts left/right wheel velocities to body twist using half the robot width.
- `nav/main.py` demonstrates SLAM/map-frame control, smoothing, obstacle clearance, bounds checking, stuck detection, and a 0.4-second manual dead-man timer. It is a better autonomy reference than copying open-loop timed driving.
- A field named `twist` does not mean the base must be placed in `MODE_TWIST`.
- `MODE_BALANCE=0` is the normal balancing mode.
- `MODE_LEAN=1` is specialized.
- `MODE_TWIST=2` is explicitly labeled “velocity, NOT balancing” in the base daemon. Do not switch modes in a generic action provider without platform-owner approval.
- The base daemon falls back to its configured default mode when mode commands expire.

The current `mc_skills.rotate(rad)` implementation is open-loop timed rotation at a fixed angular rate. It continuously publishes `[0, w]`, then publishes zero. Treat it as a bounded scan primitive, not localization-grade navigation.

### Arm control findings

Topics per arm:

- `arm_left.state` / `arm_right.state`: position, velocity, torque, temperature, and current for eight motors
- `<arm>.ctrl`: `pos[8]`, `vel[8]`, `tau[8]`, `alpha`
- `<arm>.torque`: enable flags, torque-mode flags, compliance fields, homing/calibration state
- index 7 is the gripper
- J0 is the vertical stage
- control values in `<arm>.ctrl.pos` are motor turns, not URDF radians

Ownership and enable sequence:

- Use one owner for each `<arm>.ctrl` and `<arm>.torque` topic.
- A read-only viewer must not open control writers. `examples/view_arms.py` only opens them with `--control`.
- Before enabling torque: disable first, wait for fresh state, copy the live pose into `ctrl`, flush it, then enable. The OFF-to-ON transition reseeds the daemon command filter and prevents jumps.
- Use `staged_home_arms` and `park_arms` from `bbapps/quest_teleop/scripts/homing.py`; do not duplicate or simplify homing casually.
- The arm daemon immediately disables torque when its control writer disappears.
- The arm daemon clips control targets near the current pose and to calibrated joint ranges, but application code must still generate smooth, bounded trajectories.

Cartesian IK pattern used by `mc_skills`:

1. Read live motor turns.
2. Convert with `cfg.q2urdf(live)`.
3. Reset IK with the first seven URDF joints.
4. Call `cfg.ik.solve(position_xyz, quaternion_xyzw)`.
5. Insert the seven-joint solution into a full URDF vector.
6. Convert back with `cfg.urdf2q(full)`.
7. Preserve the live J7/gripper setpoint.
8. Smoothly ramp the complete motor-turn command.

Coordinates:

- `mc_skills.Arm.goto()` expects base-frame XYZ meters and XYZW quaternion.
- Base frame: +x forward, +y left, +z up.
- The agent interface uses world-frame positions with frame/epoch metadata. A live action provider must transform the current world target into the current base frame and reject stale or cross-epoch geometry before IK.
- Do not expose raw joints, velocities, homing, or arbitrary IK targets to the model. Those belong behind semantic actions.

Mirror handling:

- Left and right arms have different `ik_sign` arrays.
- The left gripper has `gripper_sign=-1`; the right has `gripper_sign=1`.
- Use each arm’s `Config` conversion methods. Do not mirror by manually negating every motor command.
- Load calibrated ranges from each arm’s `ranges.calibration.json`; never copy limits from another robot.

### `actions/pickup.py` findings

The local and bot copies matched at review time.

This is a direct joint-space two-arm cage/lift prototype, not a general navigation provider. Its sequence is:

1. move J3/elbow to 90 degrees;
2. open J7 grippers;
3. raise J0;
4. spread both arms on J2;
5. lower J0 near the calibrated bottom;
6. optionally pinch on J2 until tracking error indicates contact;
7. hook with a J5/J6 blend derived from FK;
8. close grippers;
9. cradle with additional elbow flex;
10. lift J0 and hold.

Important properties:

- It derives inward/outward signs with FK instead of hard-coding mirror signs.
- It loads per-robot calibration and stays inside range margins.
- It self-paces arm control at 200 Hz with `keeptime=False`.
- It detects contact using command-versus-live tracking error and caps travel.
- Tracking-error contact thresholds and squeeze values are hardware-tuning assumptions, not independent possession evidence.
- The default mode lowers and holds; `--pickup` opts into cage/lift.
- Torque stays enabled while holding.
- Its `finally` block disables torque and closes writers. If carrying a box, this releases the load.

Therefore, do not call `pickup.py` directly from the agent reasoning layer and do not use its cleanup as the agent’s `cancel()` or `stop()` behavior. Agent interface v2 requires load-preserving cancellation and truthful independent holding state.

### Agent integration boundary

The action implementation must satisfy `ActionProvider` interface v2 from `bot-code/agent_types.py` and the “Agent component interface v2” section in `bot-code/README.md`.

Required semantic operations:

- `look_around`
- `approach_box`
- `pickup`
- `move_to_build`
- `place`

Required lifecycle behavior:

- idempotent request admission and lookup;
- bounded `submit`, `status`, `state`, `cancel`, and `stop` calls;
- live phase reporting;
- independent possession/readiness evidence;
- no blind replay after an unknown/timed-out submission;
- load-preserving cancellation and stop;
- recheck localization, obstacle clearance, target identity, reachability, and frame/epoch immediately before effects;
- a successful command is not proof of possession or placement.

`FunctionActions` can adapt ordinary functions, but it does not make a blocking or uninterruptible hardware routine safe. The action owner must implement cooperative cancellation and watchdog behavior.

### What the motion guide should and should not contain

Include:

- a read-only inspection checklist;
- topic/type/units tables;
- a base publisher pattern with continuous updates, validation, dead-man behavior, and a guaranteed zero command;
- arm state/control/torque schemas;
- the torque-enable and homing sequence;
- the IK conversion pattern;
- pickup prototype behavior and limitations;
- ownership conflicts and action-provider integration;
- troubleshooting by layer;
- stable source paths reviewed.

Do not:

- claim physical movement or pickup was tested;
- run `pickup.py`, `/home`, `/goto`, `/pick`, `/place`, `/rotate`, or a drive publisher during documentation verification;
- include a copy-paste command that can unexpectedly move the robot without a prominent operator/safety gate;
- tell developers to kill writer owners or restart all BBOS services;
- treat `MODE_TWIST` as ordinary balanced driving;
- import action test scripts into reasoning code;
- overwrite concurrent action/perception work.

## Verification history

Most relevant completed checks:

```bash
# Focused backend/panel/CLI checks on the bot at key-setup time
# Result: 33 passed
python -m unittest test_agent_backend test_panel test_agent_cli -v

# Clear-blueprint panel suite on the bot
# Result: 25 passed
python -m unittest test_panel -v
```

Those are historical checkpoint results. Since then, other developers changed the panel, reasoning, perception, and voice layers. Use the current project commands in [AGENTS.md](AGENTS.md) for fresh verification.

No real hardware motion was run during the speaker or motion research. No sound was played during the speaker research.

## Deferred browser task

A real-browser cancellation check remains pending from the original panel bring-up:

1. send or load a blueprint;
2. start a real-model/simulated-tool build;
3. cancel during a model request and confirm the UI returns immediately to main;
4. confirm `worker_busy` remains true only until the in-flight request returns;
5. confirm late results cannot restart or complete the cancelled build;
6. confirm a new build can start afterward;
7. recheck the Clear blueprint control in the current light-theme frontend.

This requires real, billable model calls. Do not perform it as an automatic regression check.

## Paused task: Markdown audit

The user asked to identify outdated Markdown and update or delete it. The audit was started but no cleanup edits were made before this handoff.

Confirmed stale documents:

- `PLAN.md` says no TCP 5005 receiver exists and treats the agent, receiver, voice queue, and personality as future work. `panel.StructureReceiver`, the default agent, and voice implementation now exist. Replace it with a current roadmap or clearly label it historical.
- `LLM_REASONING_PLAN.md` says implementation is paused and the framework is only proposed. The typed interfaces, mock world, validation, recovery, model backend, and default agent are implemented. Reduce it to a superseded notice or delete it if historical context is unnecessary.
- `bot-code/voice/command_map.md` calls `/say` a stub, says descend templates are absent, and miscounts routes. The current local implementation has 2 GET and 10 POST handlers, non-blocking `hook.say_text`, and both descend templates.

Smaller corrections are needed in root `README.md` and `bot-code/README.md`: stop treating the old plan as authoritative, describe the current interface-v2/live-perception boundary, remove the nonexistent staged `/drive` route, distinguish the legacy grid UI from the main panel on 8005, and replace the runbook command that starts the body server and immediately homes the robot with an inspect-and-coordinate procedure.

Preserve `WIRE_FORMAT.md`; sampled claims still match `StructureScannerItem.java`. Preserve current `AGENTS.md` operational and light-theme rules. `bot-code/voice/speaker_playback.md` is mostly current and should retain its warning that audibility has not been physically verified.

## Safe next steps

1. Create `bot-code/actions/motion_and_arms.md` from the completed research above.
2. Complete the Markdown audit, starting with the two READMEs, then the historical plans and voice command map.
3. Validate Markdown links plus Python/Bash snippets without executing motion.
4. Stage and commit documentation separately; check for concurrent changes first.
5. Re-inspect the bot working tree and running writer owners before any deployment or physical test.
6. Coordinate the first physical base, arm, pickup, speaker, and cancellation checks with the robot/action owner.
