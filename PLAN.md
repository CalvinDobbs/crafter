# Crafter — project plan

This is the overall plan. READMEs are not.

| Document | Role |
|---|---|
| **[PLAN.md](PLAN.md)** (this file) | Goals, sequencing, gaps, voice, what “done” means |
| [README.md](README.md) | Mod: how to build, run, and scan a world |
| [WIRE_FORMAT.md](WIRE_FORMAT.md) | Frozen TCP/JSON contract, game → bot |
| [bot-code/README.md](bot-code/README.md) | Robot app architecture, module owners, demo runbook |

If those disagree, this file wins on *what we are building and in what order*. Wire format wins on the scanner payload. `bot-code/contracts.py` wins on in-process types.

---

## 1. Demo thesis

A person stacks blocks in Minecraft. They right-click the Structure Scanner. The BracketBot looks at real cardboard cubes on the table, assigns each cube to a cell, and stacks them so the physical pile matches the in-game build.

It is **not** teleoperation. The game is a *target spec*. Perception, planning, and pick/place are the robot’s job.

One-line pitch: **You build it in Minecraft. It builds it in real life. It talks while it works.**

---

## 2. What is true today (GitHub `main`)

**Working / sketched**

- Fabric 1.21.11 mod (`minecraft-mod/`, id still `posstream`) scans ±64 blocks around world origin and sends one JSON object over TCP to `100.66.148.86:5005`.
- Wire format is specified and has been captured against a real send.
- `bot-code/` has the robot pipeline on paper and in Python: contracts, grid UI / json / nbt ingest, ArUco perception, planner (deterministic + LLM), orchestrator, HTTP skills client.
- Hardware is meant to live in **one** process, `mc_skills` (port 8006), so only one writer owns arms / base / speaker / LED.

**Not true yet**

- Root README still says the bot is “not in this repo.” Ignore that; `bot-code/` is here. The README was not updated after bot code landed.
- **No process listens on tcp/5005.** A scan from the game currently has nowhere to go in this repo.
- `structure_src` still loads `.json` / `.nbt` / a click-grid. It does not speak `WIRE_FORMAT.md`.
- `mc_skills` is **not in this repo**. It lives on the robot at `~/bbapps/mc_skills`. `POST /say` is a print stub (“TTS stub (wav playback hook)”).
- Orchestrator calls `say(line)` **then** `pick` **then** `place`, all blocking. Even with TTS, speech would finish *before* motion, not during it.
- Planner narration is one punchy line per *pair* of pick+place (`"{kind} block, layer {y}, going in"`). No phase-level lines (approaching, grasping, carrying).
- Live perception, gripper direction, `DOWN_QUAT`, `GRID_ORIGIN`, and `BOX_SIZE` are untuned. Demo checklist in `bot-code/README.md` is all unchecked.
- Personality of the voice is **not chosen**. Do not block motion or TTS on that decision.

Hardware daemons on the robot (arms, camera, base, speaker, …) are a platform given; they are not this project’s implementation work except as we call them through `mc_skills`.

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

### Phase 0 — Single source of truth (this week, cheap)

- Treat this file as the plan.
- Fix the stale sentence in root README (“bot side not in this repo”).
- Copy or submodule `mc_skills` into this repo (or document a hard path + owner) so the body server is not a ghost on one Jetson.

**Done when:** a new teammate can clone `crafter` and know where every process lives.

### Phase 1 — Game → robot structure (blocks the live hook)

- Add `bot-code/receiver.py`: listen `0.0.0.0:5005`, read to EOF, parse, sanity-check `count`, map palette ids → `Block.kind`, write `fixtures/structure.json` (and/or push to a queue the orchestrator already waits on).
- Convert Minecraft Y-up cells into the existing `Block(x, y, z, kind)` model (Y stays layer height).
- Handle the documented edges: empty scan sends nothing; overlapping clicks; 1500 ms accept budget; never `recv()` once and assume a full message.
- Optional: orchestrator `--listen` waits for the next payload instead of a file.

**Done when:** right-click in the superflat world produces a `Structure` the planner already understands, with the game closed or still open.

**Fallback if this slips:** grid UI or `structure_house.json`. Demo still works; the magic sentence does not.

### Phase 2 — See boxes

- Print ArUco 4×4 ids 0..N, tape one face per cube.
- Run `perception.py` live; confirm `camera.points` is pixel-aligned with the **left** half of `camera.head.rgb`.
- Hardcoded `world_state.json` remains the fallback.

**Done when:** three marked cubes on the table yield three stable `Detection`s in base frame, within a couple of centimeters, repeatedly.

### Phase 3 — Move boxes

On-robot, serialized, with `mc_skills` already running:

1. `POST /gripper` actually opens/closes (flip calibration constants if backwards).
2. `POST /goto` with a tuned `DOWN_QUAT` (top-down EE).
3. Measure `GRID_ORIGIN` and `BOX_SIZE` at the table; edit `contracts.py` once.
4. One pick + one place of a known cube to a known cell.
5. Then `main.py --planner deterministic` with a 2–4 block fixture.

**Done when:** a 3-block, 2-layer structure from a fixture is stacked without a human touching the arms.

### Phase 4 — Closed loop

Phase 1 + 2 + 3 in one run: scan in Minecraft → listen → detect → plan → execute.

**Done when:** a teammate who is not the implementer can build a small shape in-game and get a matching stack on the table.

### Phase 5 — Voice during motion (specified below)

### Phase 6 — Personality (explicitly later)

Voice pack, catchphrases, TTS voice, maybe LLM flavor. Does not gate Phases 0–5.

---

## 5. Voice narration — plan (Phase 5)

### Goal

While the robot is **actually moving**, it says what it is doing in plain language. Speech overlaps motion. Silence is worse than a slightly late line; a line that **blocks** a pick is a bug.

Personality (witty pit crew, calm museum guide, etc.) is a skin. **Do not pick it in this phase.**

### What exists

| Layer | Today |
|---|---|
| Planner | Optional one-liner per pick/place *pair* |
| Orchestrator | `skills.say(line)` then blocking `pick`/`place` |
| `mc_skills` `/say` | Prints text; does not write `speaker.audio` |
| Speaker daemon | Real: int16 16 kHz chunks, 100 ms, one `Writer` |

So we have a *slot* for talking and no overlapping audio path.

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
4. **Offline-first TTS:** Jetson-local (e.g. Piper or espeak-ng → wav → `speaker.audio` in 1600-sample frames, matching `play_sound`). Cloud TTS is optional later; it adds latency and a demo failure mode.
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
| TCP receive + structure ingest | `bot-code/receiver.py` (new), `structure_src.py` |
| Types / grid math | `bot-code/contracts.py` |
| See boxes | `bot-code/perception.py` |
| Assign boxes | `bot-code/planner.py` |
| Event → text | `bot-code/narrator.py` (new) |
| Sequence the demo | `bot-code/main.py` |
| Move + speak | `~/bbapps/mc_skills` until it lives in this repo |

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
