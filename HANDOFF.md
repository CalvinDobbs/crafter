# Crafter handoff

Current checkpoint: GitHub `main` at `aafefd5` or later. This handoff covers the reasoning/perception integration, panel/key work, speaker research, the base/arm guide, and the documentation audit — the last two of which are now complete.

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

Current baseline, run on Windows at `07664c4`:

- 106 focused reasoning/adapter/backend/CLI/panel tests pass with the command above.
- 20 pickup motion tests pass with
  `python -B -m unittest discover -s bot-code/actions -p test_pickup.py` (NumPy
  required; fake bbos and virtual time, no hardware).

The older numbers are historical checkpoints from a moving codebase. On Windows use
the focused selection above rather than full discovery — `AGENTS.md` explains why
`SavedKeyTests` misbehaves there. Routine tests must not make paid model calls or
move hardware.

## Important constraints

- Never combine real actions with mocked possession, occupancy, or safety evidence.
- Do not start another hardware writer. The live action provider must use the designated action owner and implement interface v2 truthfully.
- Do not infer possession from a successful action or a disappearing marker.
- Placement evidence must be measured after terminal action completion.
- `ActionOutcome.ts` is immutable transition/completion time; `observed_at` is the heartbeat.
- Perception provider cleanup closes sensing/transport resources only. It must not park, release, or move a loaded robot.
- The arm daemon cuts torque as soon as its control writer disappears, and re-sends
  that every second. **Closing an `arm_*.ctrl` writer drops whatever the arm is
  holding**, so a load-preserving `cancel()`/`stop()` can never be implemented by
  closing writers or cutting torque. This is the mechanism behind the rule above.
- The two arms do not mirror by a single sign flip: `ik_sign`, `gripper_sign` and the
  J0 term all differ per arm. Convert through each arm's own `q2urdf`/`urdf2q`.
- `contracts.py` and `WIRE_FORMAT.md` remain frozen.
- Preserve uncommitted perception work on the bot. Always inspect the remote tree and services before updating or launching anything.
- The main panel uses a light theme with warm red accents. Preserve native light controls, canvas colors, and viewport containment.

## Completed since the last handoff: motion guide and documentation audit

### Base movement and arm guide — done

`bot-code/actions/motion_and_arms.md` exists and is committed. Every claim in the
previous handoff's research notes was re-verified read-only against the live bot
before being written down, so the guide supersedes those notes; they are not
repeated here.

It covers the BBOS API surface, `drive.ctrl` (fields, clamps, the 100 ms command
timeout, publish rate, balance modes, the other apps competing for the topic, and
`nav/main.py` as the autonomy reference), the arm topics, torque-enable ordering,
homing/parking, command clipping, motor-turns-vs-URDF conversion and per-arm
mirroring, the `mc_skills` IK pattern, a walkthrough of `actions/pickup.py`, and
what all of it means for an interface-v2 action provider.

Validated without motion: 11 local links resolve, 4 Python snippets parse, and the
documented `test_pickup.py` command runs clean (17 tests at the time of writing, no
hardware). No writer was opened and nothing moved.

Two facts from that work are worth repeating anywhere else they matter:

- The arm daemon cuts torque the moment its control writer disappears. **Closing
  writers drops a held load.** A load-preserving `stop()` cannot be built that way.
- The two arms do not mirror by a single sign flip — `ik_sign`, `gripper_sign` and
  the J0 term all differ. Always convert through the arm's own `q2urdf`/`urdf2q`.

`pickup.py` changed three times during this work (wrist preparation moved before the
descent, then to J5 yaw alone; `--lower-only` / `--squeeze` / `--hook` were added).
The action owner is actively developing it. **Read its docstring and `--help` rather
than any prose description, including the guide's.**

### Documentation audit — done

| Document | Outcome |
|---|---|
| `PLAN.md` | Current-state section rewritten; phases marked done/blocked; now defers to `bot-code/README.md` for what exists. Phase 3 (the action provider) is called out as the critical path. |
| `LLM_REASONING_PLAN.md` | Banner marking it superseded, with a table mapping each proposal to where it was implemented. |
| `README.md` | Points at `bot-code/README.md` and `AGENTS.md` instead of treating `PLAN.md` as the status page; states plainly that no action provider exists. |
| `bot-code/README.md` | Removed a runbook line that chained `POST /home` (real arm motion) onto starting the body server; deleted the non-existent `/drive` endpoint; added the deployed-only `/voice_script`; broadened writer-ownership language; flagged the legacy grid UI's port collision with the panel. |
| `bot-code/voice/command_map.md` | Rewritten against both mc_skills copies, which have diverged (deployed has 12 routes, the repo copy 11). `/say` is no longer described as a stub; the descend templates exist. |
| `WIRE_FORMAT.md`, `AGENTS.md` | Preserved. |

`reference/` now vendors a snapshot of the robot's `bbapps` into the repository, so
the sample apps can be read without SSH. It is a snapshot, not a live mirror —
re-check the bot before relying on it.


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

The documentation work is finished. What is left is the thing everything else is
waiting on.

1. **Implement the interface-v2 action provider.** This is the critical path: until
   it exists, a live build fails preflight by design. Start from
   `bot-code/actions/motion_and_arms.md` for how the hardware works and the
   interface-v2 section of `bot-code/README.md` for what the provider must
   guarantee. Possession and placement evidence must be independent and measured,
   not inferred from a command completing.
2. Re-inspect the bot working tree, running services and writer owners before any
   deployment or physical test. Two sessions edited this repository concurrently
   during the last block of work; do not assume the bot matches `main`.
3. Coordinate the first physical base, arm, pickup, speaker and cancellation checks
   with the robot/action owner. None of these have ever been run.
4. The yaw sign on `drive.ctrl` (`positive = CCW`) is still documented as unverified
   in the source. Confirm it on the robot before any autonomous turn.

## Key files

- `bot-code/agent.py` — verified agent loop and model context
- `bot-code/agent_types.py` — interface-v2 contracts and bounded world model
- `bot-code/agent_adapters.py` — provider loading, action bridge, and live perception adapter
- `bot-code/agent_backend.py` — strict model schema, image transport, and world-model instructions
- `bot-code/main.py` — default agent, read-only perception CLI, and explicit legacy mode
- `bot-code/perception.py` — independently owned live sensing/world model
- `bot-code/panel.py` — Minecraft receiver and simulated-tools panel backend
- `bot-code/web/` — current panel frontend and packaged assets
- `bot-code/actions/pickup.py` — direct joint-space pickup prototype, actively changing
- `bot-code/actions/motion_and_arms.md` — source-checked base and arm guide
- `bot-code/voice/speaker_playback.md` — source-checked speaker guide
- `bot-code/README.md` — current interface/runbook reference
- `reference/bbapps/` — vendored snapshot of the robot's sample apps
- `AGENTS.md` — operational rules and verified commands
