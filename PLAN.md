# Crafter — project plan

This is the overall plan: goals and sequencing. It is **not** a status page — for what the code does today, read the sources below.

| Document | Role |
|---|---|
| **[PLAN.md](PLAN.md)** (this file) | Goals, sequencing, voice design, what “done” means |
| [bot-code/README.md](bot-code/README.md) | **Current** robot architecture, interface v2, module owners, runbook |
| [AGENTS.md](AGENTS.md) | Current operational rules and verified commands |
| [README.md](README.md) | Mod: how to build, run, and scan a world |
| [WIRE_FORMAT.md](WIRE_FORMAT.md) | Frozen TCP/JSON contract, game → bot |

This file wins on *what we are building and in what order*. It does not win on current state: where this file and `bot-code/README.md` disagree about what exists, the README and the code win. Wire format wins on the scanner payload. `bot-code/contracts.py` wins on in-process types.

---

## 1. Demo thesis

A person stacks blocks in Minecraft. They right-click the Structure Scanner. The BracketBot looks at real cardboard cubes on the table, assigns each cube to a cell, and stacks them so the physical pile matches the in-game build.

It is **not** teleoperation. The game is a *target spec*. Perception, planning, and pick/place are the robot’s job.

One-line pitch: **You build it in Minecraft. It builds it in real life. It talks while it works.**

---

## 2. What is true today (GitHub `main`)

This section goes stale quickly. It was last checked against the code on 2026-09-12; [bot-code/README.md](bot-code/README.md) is the authority.

**Built**

- Fabric 1.21.11 mod (`minecraft-mod/`, id still `posstream`) scans ±64 blocks around world origin and sends one JSON object over TCP to `100.66.148.86:5005`.
- Wire format is specified, frozen, and captured against a real send.
- `bot-code/panel.py` listens on `0.0.0.0:5005`, reads to EOF, and parses and validates the wire payload directly. `structure_src.py` still offers `.json` / `.nbt` / click-grid ingest as an offline alternative, not as the live path.
- The panel serves a browser UI on `127.0.0.1:8005`: preview, Start, Clear blueprint, masked model-key entry. It makes **real model calls with simulated tools**, and starts nothing until Start is clicked.
- A stateful reasoning agent (`agent.py`) is the default CLI mode, with typed interface-v2 contracts (`agent_types.py`), provider loading and a live read-only perception adapter (`agent_adapters.py`), a strict model backend (`agent_backend.py`), and an offline mock world (`mock_agent_world.py`). `--mode oneshot` is the separate legacy pipeline.
- `perception.py` runs live sensing and a persistent world model on the robot, verified with live cameras and a TensorRT box detector. It went beyond the ArUco plan below: detection is markerless, with identity, confidence, depth and position uncertainty.
- Voice exists and is non-blocking: `SpeechQueue` on a background thread, phase-level events (`pick.approach` / `grasp` / `lift`, `place.approach` / `release` / `retreat`), voice packs with one selected, and OpenAI TTS with a WAV cache.
- Speaker playback is documented from source in [bot-code/voice/speaker_playback.md](bot-code/voice/speaker_playback.md); base and arm control in [bot-code/actions/motion_and_arms.md](bot-code/actions/motion_and_arms.md).

**Not true yet**

- **No action provider.** There is no interface-v2 implementation that moves the real robot, so a live build still fails preflight. `actions/pickup.py` is a joint-space prototype, not a provider.
- Possession, occupancy and build-site safety are **unknown** to the agent, by design — the perception adapter advertises only what it can actually evidence.
- `GRID_ORIGIN` and `BOX_SIZE` are untuned at the table; no physical pick, place, base move or speaker check has been run.
- `mc_skills` is **not fully in this repo**. The voice-side copy lives at `bot-code/voice/mc_skills_main.py`; the deployed body server is on the robot at `~/bbapps/mc_skills`. Treat them as separate and diff before assuming they match.

Hardware daemons on the robot (arms, camera, base, speaker, …) are a platform given; they are not this project’s implementation work except as we call them.

---

## 3. System, in one picture

```
 Minecraft (Structure Scanner)
        TCP :5005   WIRE_FORMAT.md
                │
                v
     bot-code receiver  →  Structure (contracts.py)
                │
 cameras.points + rgb  →  perception  →  Detection[]
                │
                v
            planner  →  Plan {actions, optional flavor lines}
                │
                v
          orchestrator
 treble: narrator (text)     bass: SkillsClient HTTP
                │                         │
                v                         v
         POST /say (queue)         POST /pick /place /home …
                                   (owns Writers; emits motion phases)
```

Rules that do not change:

1. Only `mc_skills` opens hardware `Writer`s (`arm_*.ctrl`, `drive.ctrl`, `speaker.audio`, `led.ctrl`).
2. The LLM (when used) chooses *which box goes where* and may write flavor text. It never emits joint angles or Cartesian setpoints.
3. Every module has a mock path. Robot time is scarce; laptop time is not.
4. `contracts.py` and `WIRE_FORMAT.md` stay frozen unless the team agrees.

---

## 4. Phases

Do these in order. A later phase may start on a laptop while an earlier one is tuned on the robot, but a live demo requires 0–3.

### Phase 0 — Single source of truth — **done**

- Treat this file as the plan.
- Fix the stale sentence in root README (“bot side not in this repo”).
- Copy or submodule `mc_skills` into this repo (or document a hard path + owner) so the body server is not a ghost on one Jetson.

**Done when:** a new teammate can clone `crafter` and know where every process lives.

### Phase 1 — Game → robot structure — **done**

- Add `bot-code/receiver.py`: listen `0.0.0.0:5005`, read to EOF, parse, sanity-check `count`, map palette ids → `Block.kind`, write `fixtures/structure.json` (and/or push to a queue the orchestrator already waits on).
- Convert Minecraft Y-up cells into the existing `Block(x, y, z, kind)` model (Y stays layer height).
- Handle the documented edges: empty scan sends nothing; overlapping clicks; 1500 ms accept budget; never `recv()` once and assume a full message.
- Optional: orchestrator `--listen` waits for the next payload instead of a file.

**Done when:** right-click in the superflat world produces a `Structure` the planner already understands, with the game closed or still open.

**Fallback if this slips:** grid UI or `structure_house.json`. Demo still works; the magic sentence does not.

### Phase 2 — See boxes — **done, and superseded**

`perception.py` went further than this phase asked: detection is markerless rather than ArUco-tagged, and the world model persists box tracks across observations. The original steps are kept for context.

- Print ArUco 4×4 ids 0..N, tape one face per cube.
- Run `perception.py` live; confirm `camera.points` is pixel-aligned with the **left** half of `camera.head.rgb`.
- Hardcoded `world_state.json` remains the fallback.

**Done when:** three marked cubes on the table yield three stable `Detection`s in base frame, within a couple of centimeters, repeatedly.

### Phase 3 — Move boxes — **not started; this is the critical path**

This phase now means **implementing an interface-v2 `ActionProvider`**, not driving the legacy `mc_skills` HTTP endpoints. Read [bot-code/actions/motion_and_arms.md](bot-code/actions/motion_and_arms.md) for how the base and arms actually work, and the interface-v2 section of [bot-code/README.md](bot-code/README.md) for what the provider must guarantee. The original endpoint-level steps below remain a reasonable order for the first physical checks:

1. `POST /gripper` actually opens/closes (flip calibration constants if backwards).
2. `POST /goto` with a tuned `DOWN_QUAT` (top-down EE).
3. Measure `GRID_ORIGIN` and `BOX_SIZE` at the table; edit `contracts.py` once.
4. One pick + one place of a known cube to a known cell.
5. Then `main.py --planner deterministic` with a 2–4 block fixture.

**Done when:** a 3-block, 2-layer structure from a fixture is stacked without a human touching the arms.

### Phase 4 — Closed loop — **blocked on Phase 3**

Phase 1 + 2 + 3 in one run: scan in Minecraft → listen → detect → plan → execute.

**Done when:** a teammate who is not the implementer can build a small shape in-game and get a matching stack on the table.

### Phase 5 — Voice during motion — **done** (design retained below)

### Phase 6 — Personality — **done**

Voice pack, catchphrases, TTS voice, maybe LLM flavor. Does not gate Phases 0–5.

---

## 5. Voice narration — plan (Phase 5)

### Goal

While the robot is **actually moving**, it says what it is doing in plain language. Speech overlaps motion. Silence is worse than a slightly late line; a line that **blocks** a pick is a bug.

Personality (witty pit crew, calm museum guide, etc.) is a skin. It was deliberately not picked during this phase; it was chosen afterwards, in Phase 6.

### What exists

**This phase is built.** The table below is what it looked like beforehand, kept because the design that follows was written against it.

| Layer | Before Phase 5 |
|---|---|
| Planner | Optional one-liner per pick/place *pair* |
| Orchestrator | `skills.say(line)` then blocking `pick`/`place` |
| `mc_skills` `/say` | Prints text; does not write `speaker.audio` |
| Speaker daemon | Real: int16 16 kHz chunks, 100 ms, one `Writer` |

The gap was a *slot* for talking and no overlapping audio path. Today `SpeechQueue` runs speech on a background thread, `phases.py` emits the phase events below, and `tts.py` synthesizes through OpenAI with a WAV cache. The remaining unknown is physical audibility, which has never been checked — see [bot-code/voice/speaker_playback.md](bot-code/voice/speaker_playback.md).

### Design: events, not paragraphs

Narration is driven by **motion phases**, not by a speech at the start of the whole plan.

`mc_skills` already knows the phases inside `/pick` and `/place` (approach → descend → gripper → lift/retreat). It should emit them. The orchestrator cannot, because it is blocked inside one HTTP call for the whole primitive.

**Phase vocabulary (frozen for v1):**

| Event | When | Neutral line (v1 copy) |
|---|---|---|
| `scan.start` | perception / rotate sweep | “Looking for boxes.” |
| `plan.ready` | plan built | “{n} blocks to place.” |
| `home.start` / `home.done` | torque + home | “Waking the arms up.” |
| `pick.approach` | moving to box | “Going to box {id}.” |
| `pick.grasp` | closing gripper | “Picking it up.” |
| `pick.lift` | leaving table | “Got it.” |
| `place.approach` | moving to cell | “Placing on layer {y}.” |
| `place.release` | opening gripper | “Setting it down.” |
| `place.retreat` | backing off | — (optional, often skip) |
| `build.done` | celebrate | “Build complete.” |
| `fail` | any primitive error | “That miss, trying the next one.” |

Planner flavor lines (`Plan.narration`) are **optional extras** queued at `pick.approach` if present. They must not replace the phase line, and they must not be required for a demo.

### Process split (keeps the one-writer rule)

- **`mc_skills` owns playback.** It is the only `speaker.audio` writer. `/say` becomes a **non-blocking enqueue**: generate or fetch PCM, push onto a speech queue, return immediately. The motion lock (`_busy`) must **not** apply to `/say`.
- **`mc_skills` emits phases** on a tiny in-process callback and/or a log line. For v1, the body server itself may call the same enqueue with the neutral template when a phase starts. That way speech happens even if the orchestrator never calls `/say`.
- **`bot-code/narrator.py`** (new, pure, mockable) maps `(event, context) → str`. v1 is a dict of templates. Later a VoicePack object swaps templates + TTS voice. Orchestrator can also POST `/say` for scan/plan lines the body server does not see.

TTS lives behind `/say` so laptop mock mode still prints the same strings.

### Audio behavior

1. **Overlap:** start TTS (or start playing a cached wav) at phase start; do not `join()` before the next IK segment.
2. **Queue policy:** at most one line playing + one pending. A new phase **preempts** the pending line; if the current wav has > ~400 ms left, cut it at a chunk boundary and play the new line. Prefer being slightly terse over talking over yourself.
3. **Length:** ≤ 8 words, ≤ ~2 s of audio. Pick/place segments are a few seconds; a paragraph will spill into the next phase.
4. **Offline-first TTS:** Jetson-local (e.g. Piper or espeak-ng → wav → `speaker.audio` in 1600-sample frames, matching `play_sound`). Cloud TTS is optional later; it adds latency and a demo failure mode. **This is the one item that shipped differently:** `tts.py` synthesizes through OpenAI, backed by the pre-rendered WAV cache in `voice/wavs/` so the demo lines do not depend on a live network call. The latency and failure-mode concerns above are the reason the cache exists.
5. **Cache** by (voice_id, text) so repeated “Picking it up.” is instant.
6. **Failure:** if TTS throws, log and keep moving. Mute is acceptable; a frozen arm is not.
7. **Mock:** `MockSkills.say` already logs; add phase logs in mock `mc_skills` the same way.

### What we are explicitly not doing in Phase 5

- Choosing a character, accent, or joke style.
- LLM-generated live commentary on the hot path (extra latency + new failure mode). If we want punchy lines, generate them **in the planner before motion starts** and attach them to events.
- Lip-sync, LED visemes, or talking during `fail` retries beyond one short line.
- A second process writing `speaker.audio`.

### Personality later (Phase 6)

Introduce a `VoicePack`:

```python
@dataclass
class VoicePack:
    id: str                 # "neutral" | TBD
    templates: dict[str, str]
    tts_voice: str          # piper/espeak voice id
    max_words: int = 8
```

Default pack is `neutral` (the table above). A future pack only replaces strings + voice id. Planner prompt flavor (“pit-crew”) moves into the pack, not into IK.

**Done when (Phase 5):** a real pick/place on the robot produces audible lines **during** approach and grasp, `/say` does not block `/pick`, and `MOCK=1` prints the same event strings.

---

## 6. Risks (and the fallback already in the bot README)

| Risk | Move down, don’t stall |
|---|---|
| Mod / TCP | `.nbt` → grid UI → fixture json |
| ArUco / lighting | color blobs → hardcoded detections |
| LLM planner | deterministic |
| IK grasp | `/goto` waypoints → recorded mimic playback |
| TTS | printed lines + optional beep on LED |

Hardware time is a lock. Coordinate it; use one running `mc_skills`.

---

## 7. Suggested ownership (files, not people titles)

| Surface | Where |
|---|---|
| Scanner item + TCP send | `minecraft-mod/` |
| Wire contract | `WIRE_FORMAT.md` |
| TCP receive + structure ingest | `bot-code/panel.py` (live path), `structure_src.py` (offline) |
| Types / grid math | `bot-code/contracts.py` |
| See boxes | `bot-code/perception.py` |
| Reason and sequence | `bot-code/agent.py`, `agent_types.py`, `agent_adapters.py` |
| Assign boxes (legacy one-shot) | `bot-code/planner.py` |
| Event → text | `bot-code/voice/narrator.py`, `voice/packs.py` |
| Speech and TTS | `bot-code/voice/speech_queue.py`, `voice/tts.py` |
| Move the robot | unowned — the interface-v2 action provider (Phase 3) |
| Move + speak (deployed) | `~/bbapps/mc_skills` on the robot |

Cross-file edits go through the file owner, especially `contracts.py` and `main.py`.

---

## 8. Definition of a shippable demo

Minimum:

1. Superflat world, scanner item, 3–8 block build near origin.
2. Robot detects that many (or more) marked cubes.
3. Deterministic plan, bottom-up, no API key.
4. Physical stack matches occupancy (color matching optional).
5. Neutral voice lines audible during pick and place, not only before.

Stretch (not required): LLM assignment + flavor lines, live color matching, rotate-to-scan, celebration choreography, a named personality.
