# crafter bot-code — digital-twin block builder

Project-level sequencing, demo definition, and the voice-during-motion plan: **[../PLAN.md](../PLAN.md)**. This file is the robot-app architecture and runbook.

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
  raises `RuntimeError`. Exactly ONE process owns hardware: `mc_skills`.
  Everyone else calls its HTTP API. Never open Writers on `arm_*.ctrl`,
  `*.torque`, `drive.ctrl`, `led.ctrl`, `speaker.audio` outside it.
- **Readers are unlimited.** Perception can run live on the robot at any time
  without conflicting with anything.
- Arm control = position in motor turns. Cartesian via IK:
  `Config("arm_right").ik.solve(pos[3], quat_xyzw[4]) -> 7 urdf joints` →
  `urdf2q` → turns.
- `camera.points` = **base-frame xyz per pixel** — detections arrive already
  in robot coordinates. Base frame: +x forward, +y left, +z up, meters.
- Homing: use `staged_home_arms`/`park_arms` from
  `bbapps/quest_teleop/scripts/homing.py` (mc_skills does). Leave arms limp.
- VLM plumbing exists in `~/bbapps/inference/vlm.py` (multi-provider clients,
  `grab_right_eye`, strict JSON schema) — reuse it, don't rebuild it.
- Full platform doc: `~/bbapps/AGENTS.md`.

## The world model (core concept — read this)

Each `scan()` produces a snapshot in the robot's **current** base frame:

```python
Scan { boxes: [Detection],   # loose, pickable — outside the grid
       stacked: [Detection], # inside grid footprint = already placed
       anchor: [x,y,z]|None, # base-frame pos of the grid anchor marker
       ts: float }
```

- **Anchor marker** (ArUco id 99) is taped at cell (0,0)'s table spot and
  re-detected every scan → the virtual grid is always resolved in the
  *current* heading. This replaces SLAM: rotating stales old detections, but
  every action re-scans first. Fallback when anchor unseen: fixed
  `GRID_ORIGIN` (fine while the base stays parked).
- **Grid gate:** any detection projecting inside the footprint is `stacked`,
  never a pick candidate. "Don't grab the 3rd block off the stack" is a
  spatial rule in code — the LLM never sees placed boxes as options.
- **Dual tracking:** `placed` (agent's authoritative set — what SHOULD be
  there) vs `column_heights()` (measured z per cell from the pointcloud —
  what IS there). Disagreement = fumble → feed back, re-plan.

## Architecture

```
 minecraft-mod ──TCP:5005──> structure_rx.py ──> structure.json ─┐
   (grid UI / .json fixtures are fallbacks)                       v
 cameras+depth ──> perception.scan() ──> Scan ──> agent/planner ──> steps
       (Readers only)                                │            │
                                                     v            v
                                   orchestrator (main.py)  <── feedback
                                                     │ HTTP
                                                     v
                       ┌─────── mc_skills (body server, :8006) ────────┐
                       │ owns ALL hardware Writers                     │
                       │ /home /park /pick /place /goto /drive         │
                       │ /gripper /rotate /say /celebrate              │
                       └───────────────────────────────────────────────┘
```

## Files

```
~/bbapps/mc_skills/main.py   body server — ONLY hardware writer owner
bot-code/
  contracts.py               shared types (pure python, no bbos) — FROZEN
  skills_client.py           SkillsClient (HTTP) + MockSkills
  structure_src.py           .nbt import, structure.json, grid web UI :8005
  structure_rx.py            STAGED: TCP :5005 receiver per WIRE_FORMAT.md
  perception.py              Scan world model, gate, verify_* — DONE
  planner.py                 one-shot plan_build() — deterministic + LLM
  agent.py                   STAGED: per-step reasoning loop (see below)
  main.py                    orchestrator (`run minecraft` equivalent)
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

## perception.py — DONE (Readers only, live-safe)

- `scan(mock=False) -> Scan` — ArUco `DICT_4X4_50` (marker id = box id, 99 =
  anchor) → pixel mask → median `camera.points` → base-frame pos; grid gate
  splits `boxes`/`stacked`.
- `is_in_grid(pos, anchor)`, `cell_center(x,y,z,anchor)`, `resolve_anchor`,
  `column_heights(cells, anchor)` — measured stack tops (90th-pct z disk).
- `verify_pick(box)` — marker gone/moved from old spot.
  `verify_place(cell, anchor, expected_top)` — measured z ≈ expected.
- `scan_all(skills, mock, sweeps)` — compat wrapper used by main.py.
- **Debug viz:** `uv run perception.py --viz [--mock]` → :8007 top-down map
  (robot, grid, loose/stacked markers, anchor) + live head-cam feed.
- Prep: print markers 0..N-1 + id 99 anchor. Fallbacks: color blobs →
  hardcoded positions.
- ⚠️ FIRST LIVE CHECK: confirm `camera.points` is pixel-aligned to the left
  half of head rgb (compare shapes) before trusting positions.

## agent.py — STAGED (the reasoning loop)

The build memory lives in CODE, not the model's context:

```python
BuildState { placed: set[cell], loose: {id: Detection}, used_boxes: set,
             anchor, column_heights, history }
think(state, last_result) -> Step   # backends: openai | mcp | deterministic
validate(step, state) -> str|None   # reject illegal: occupied/unsupported
                                    # cell, reused/gated box
run(structure, skills, detect_fn, max_steps)
```

LLM ops (strict JSON schema, `reason` field first): `pick_place{box_id,cell,
say}` | `scan` (rotate+re-detect until loose box found) | `approach{box_id}`
(rotate to bearing → `/drive` fwd → re-detect → pick) | `done`.
Per turn: refresh Scan → think → validate → execute → verify_pick/
verify_place → on failure append error to history and loop. `max_steps` caps
runaway. Deterministic think() mirrors `plan_deterministic` — demo survives a
dead API. Rule: LLM chooses WHICH box → WHICH cell; never positions/joints.

## mc_skills — body server (:8006)

```bash
uv run ~/bbapps/mc_skills/main.py          # real: claims writers at boot
MOCK=1 uv run ~/bbapps/mc_skills/main.py   # same API, no bbos — laptops
```

Endpoints: `GET /health` `/state`, POST `/home` `/park`
`/goto{pos,quat?,duration}` `/pick{pos}` `/place{pos}` `/gripper{open}`
`/rotate{rad}` `/say{text}` `/celebrate`; **`/drive{v,w,secs}` staged** for
approach moves. Responses `{ok, result|error}`; serialized by a lock —
concurrent calls get `busy`.

pick = approach +APPROACH_H → descend → grip → lift; place = mirror. TUNE on
robot: `grip_open`/`grip_closed` direction (test `POST /gripper`), `DOWN_QUAT`
top-down orientation, `APPROACH_H`. Add endpoints, don't rename.

## structure_src / structure_rx — the Minecraft side

- `structure_rx.py` (staged, new file — anyone can grab): TCP :5005 server,
  read-to-EOF, wire payload → `save_structure`. Run as thread in main.py or
  standalone. Empty scan never connects — silence isn't data.
- `load_structure(path)`: `.json` or `.nbt` (nbtlib).
- `serve_grid_ui(:8005)`: click-grid → `fixtures/structure.json`. Guaranteed
  fallback if mod plumbing stalls — judges can't tell the difference.

## planner.py — one-shot fallback

`plan_build(structure, boxes, backend="auto")` — used by `--mode oneshot`.
LLM call validated (legal cells, existing ids, support order); any failure →
deterministic. Keep it: it's the safety net under the agent loop.

## orchestrator — main.py

`--mode agent|oneshot` (agent staged): load structure → `skills.home()` →
loop or `plan_build` → execute → celebrate. Missing boxes warn, don't die.

```bash
uv run main.py --mock        # whole pipeline, no robot — DO THIS FIRST
```

## Parallel-dev rules (the point of this layout)

1. **Only mc_skills opens hardware Writers.** Everything else → SkillsClient.
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
# body — leave running
uv run ~/bbapps/mc_skills/main.py && curl -XPOST localhost:8006/home
# perception debug (safe alongside body)
uv run ~/crafter/bot-code/perception.py --viz        # :8007
# pipeline
cd ~/crafter/bot-code && uv run main.py --mock       # then real args
```

## Tune-before-demo checklist

- [ ] `POST /gripper` direction; `DOWN_QUAT`; `APPROACH_H`
- [ ] `BOX_SIZE`, anchor marker placement at cell (0,0)
- [ ] `camera.points`↔rgb alignment vs tape measure
- [ ] ArUco printed: boxes 0..N-1 + anchor 99
- [ ] receiver listening on :5005 before the mod right-click
- [ ] `OPENAI_API_KEY` if llm/agent backend used

## Fallback ladder — move down, don't get stuck

- structure: mod TCP → grid UI → fixture json
- detection: ArUco → color blobs → hardcoded positions
- reasoning: agent loop → oneshot planner → hand-ordered plan.json
- motion: IK pick/place → curl /goto waypoints → mimic record/playback
