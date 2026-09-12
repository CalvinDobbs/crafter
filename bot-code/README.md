# minecraft — digital-twin block builder

Project-level sequencing, demo definition, and the voice-during-motion plan: **[../PLAN.md](../PLAN.md)**. This file is the robot-app architecture and runbook.

## The idea

A player builds a structure in Minecraft (stacking blocks). The robot finds
physical cardboard boxes scattered in front of it, picks them up, and stacks
them so the real-world pile matches the in-game build. It is NOT teleoperation
— the robot never mirrors player movements. The game build is a *target spec*;
the robot perceives the world, plans which box goes to which cell, and executes
pick/place primitives. An LLM does the high-level reasoning (box assignment,
ordering, narration); deterministic code does all motion.

Demo story: "You build it in Minecraft. It builds it in real life."

## Platform context (read this before writing code)

The robot is BracketBot: self-balancing 2-wheel base, two 8-DoF arms
(index 7 = gripper), head stereo camera, 2 wrist cameras, mic/speaker, LED.
All I/O goes through **bbos IPC**: shared-memory topics accessed via
`Reader`/`Writer`/`Type`/`Config` from the `bbos` package.

Critical facts that shape this design:

- **ONE writer per topic.** A second `Writer("arm_right.ctrl", ...)` in any
  other process raises `RuntimeError`. This is why there is exactly ONE
  hardware-owning process (`mc_skills`) and everyone else goes through its
  HTTP API. Never open a `Writer` on `arm_*.ctrl`, `arm_*.torque`,
  `drive.ctrl`, `led.ctrl`, or `speaker.audio` outside `mc_skills`.
- **Readers are unlimited.** Many processes may `Reader(...)` the same topic
  concurrently. Perception work can run live on the robot at any time without
  conflicting with anything.
- Arm control is position control in **motor turns** (`<arm>.ctrl` pos field,
  8 floats). Cartesian moves go through IK: `Config("arm_right").ik.solve(
  pos[3], quat_xyzw[4]) -> 7 urdf joints`, then `urdf2q` -> motor turns.
- `camera.points` publishes **base-frame xyz per pixel** — detection output is
  already in the robot's coordinate frame. No TF math needed.
- Base frame convention: `+x` forward, `+y` left, `+z` up, meters.
- Homing is nontrivial (staged torque enable). Always use
  `staged_home_arms`/`park_arms` from `bbapps/quest_teleop/scripts/homing.py`
  (mc_skills already does). On exit: leave arms limp.
- Full platform details: `~/bbapps/AGENTS.md`.

## Architecture

```
  Minecraft / grid UI ──> structure.json ──┐
                                           v
  cameras+depth ──> perception ──> detections ──> planner ──> Plan
        (Readers only)                (pure fn, LLM optional)   │
                                                              v
                                            orchestrator (main.py)
                                                              │ HTTP
                                                              v
                              ┌─────── mc_skills (body server) ────────┐
                              │ owns ALL hardware Writers              │
                              │ /home /park /pick /place /goto         │
                              │ /gripper /rotate /say /celebrate       │
                              └────────────────────────────────────────┘
```

Every arrow is a JSON or function-call boundary defined in `contracts.py`.
Modules never share state, never import each other's internals, and never
touch hardware outside their lane.

## Files

```
bbapps/mc_skills/main.py     body server — the ONLY hardware writer owner
bbapps/minecraft/
  contracts.py               shared types + grid math (pure python, no bbos)
  skills_client.py           SkillsClient (HTTP) + MockSkills (same interface)
  structure_src.py           .nbt import, structure.json, fallback grid web UI
  perception.py              camera/depth -> Detection list (Readers only)
  planner.py                 plan_build(): LLM backend + deterministic fallback
  main.py                    orchestrator app (`run minecraft`)
  fixtures/                  sample structure + world state for offline dev
```

## Contracts — freeze these first

`contracts.py` is the single source of truth. If you change it, tell the team.

```python
Block(x, y, z, kind)            # minecraft cell; y = layer (0 = on table)
Structure(blocks)               # target spec; .layers(), .supported()
Detection(id, pos, color, size) # a physical box; pos = base-frame xyz meters
Action(kind, box_id, cell)      # 'pick' uses box_id; 'place' uses cell=(x,y,z)
Plan(actions, narration)        # pick/place alternate pairwise + voice lines

cell_center(x, y, z) -> [bx, by, bz]   # minecraft cell -> base-frame position
```

JSON on disk:
- `structure.json`: `{"blocks": [{x,y,z,kind}, ...]}`
- `world_state.json`: `{"detections": [{id, pos:[x,y,z], color, size}, ...]}`

Module boundaries:

| producer | artifact / call | consumer |
|---|---|---|
| structure_src | `load_structure(path) -> Structure` | orchestrator |
| perception | `detect_boxes(mock=?) / scan_all(...) -> list[Detection]` | orchestrator |
| planner | `plan_build(structure, boxes, backend) -> Plan` | orchestrator |
| mc_skills | HTTP API (below) | orchestrator, any dev's curl |

## mc_skills — the body server (port 8006)

```bash
uv run ~/bbapps/mc_skills/main.py          # real: claims all writers at boot
MOCK=1 uv run ~/bbapps/mc_skills/main.py   # fake: same API, no bbos import
```

Endpoints (POST json unless noted): `GET /health`, `GET /state`,
`/home`, `/park`, `/goto {pos,quat?,duration}`, `/pick {pos}`,
`/place {pos}`, `/gripper {open}`, `/rotate {rad}`, `/say {text}`,
`/celebrate`. All responses: `{ok: bool, result|error}`.
Calls are serialized by an internal lock — concurrent requests get
`{ok:false, error:"busy"}`.

Internally: `pick` = approach `+APPROACH_H` above target → descend → close
gripper → lift. `place` = mirror. Gripper open/close are motor-turn constants
loaded from the arm's `ranges.calibration.json` — verify direction with
`POST /gripper` before stacking anything. `DOWN_QUAT` is the top-down EE
orientation — TUNE on robot.

Working on this module: you own `Arm`/`Body` and the endpoint handlers. You
may add endpoints (e.g. `/handoff`, `/wiggle`) — add, don't rename. Anyone
testing needs the robot or `MOCK=1`.

## structure_src — the Minecraft side (no robot needed)

`load_structure(path)`: `.json` -> parse, `.nbt` -> parse structure-block
export (nbtlib; skips air/structure_void, maps `minecraft:foo` -> `foo`).

`serve_grid_ui(port=8005)`: standalone click-grid web page (`uv run
structure_src.py`) — place blocks on an 8x8 grid with a layer selector, SAVE
writes `fixtures/structure.json`. This is the guaranteed input path; a live
mod/websocket hook is a stretch goal that just writes the same file.

## perception — box detection (Readers only, live-safe)

`detect_boxes(mock=False) -> list[Detection]`: grabs `camera.head.rgb` (split
to left via `Config("cam_head").split`) + `camera.points`, runs ArUco
detection (`cv2.aruco`, `DICT_4X4_50`; marker id == box id), takes the median
base-frame position of pointcloud samples under each marker's pixel mask.

`scan_all(skills, mock, sweeps)`: repeat detect + optional `/rotate` between
sweeps to cover more of the room; merges by id.

Prep: print ArUco markers 0..7, tape one face of each box. Fallbacks if
fiducials fail: color/contour blob segmentation, or hardcoded box positions.
Offline dev: `python perception.py --mock` serves `fixtures/world_state.json`.

VERIFY on first live run: that `camera.points` is pixel-aligned to the left
half of the head image (check shapes; swap or index differently if off).

## planner — reasoning (pure functions, laptop-only)

`plan_build(structure, boxes, backend="auto") -> Plan`:
- `"deterministic"`: sorts target cells bottom-up (y, then x, z), assigns the
  nearest unused box by XY distance. Zero dependencies — the demo always has
  this path.
- `"llm"`: one OpenAI call (`OPENAI_API_KEY` env). Prompt gives the block
  list + box positions; response is `{"steps":[{box_id, cell, say}]}`. Output
  is *validated* (legal cells, existing ids, support ordering) and ANY failure
  falls back to deterministic.

Rule: the LLM chooses *which box goes where* and writes narration. It never
emits positions or joint values — coordinates come from `cell_center()` and
IK inside mc_skills. LLM = planner, not controller.

## orchestrator — main.py

State machine: load structure -> `skills.home()` -> `scan_all` ->
`plan_build` -> loop (say -> pick(box.pos) -> place(cell_center(cell))) ->
`celebrate`. Missing boxes are skipped with a warning, not fatal.

```bash
uv run main.py --mock                                   # whole pipeline, no robot
uv run main.py --structure fixtures/structure_house.json
uv run main.py --grid-ui                                # browser build first
uv run main.py --planner deterministic                  # no API key needed
uv run main.py --sweeps 4                               # rotate between scans
```

## Parallel-dev rules (the whole point of this layout)

1. **Only mc_skills opens hardware Writers.** Everything else calls
   `SkillsClient`/`MockSkills`. If your code needs the robot to do something,
   it calls HTTP — or the capability gets added as an endpoint in mc_skills.
2. Readers are fair game anywhere, anytime, in any number of processes.
3. `contracts.py` is frozen unless the team agrees; it has no bbos import so
   it works on laptops.
4. Every module runs without the robot: `--mock` / `MockSkills` / fixtures.
   Do not block on robot time to develop.
5. Own your file. Cross-file edits go through the owner or a message —
   especially `main.py` (the merge point) and `contracts.py` (the contract).
6. Robot hardware time is serialized by nature — coordinate it in chat; use
   the running mc_skills server rather than launching your own arm code.

## Runbook (real run)

```bash
# terminal 1 — the body (leave running all session)
uv run ~/bbapps/mc_skills/main.py
curl -XPOST localhost:8006/home          # energize + home arms once

# terminal 2 — input (pick one)
uv run ~/bbapps/minecraft/structure_src.py     # grid UI on :8005, or use .nbt/.json

# terminal 3 — the brain
cd ~/bbapps/minecraft
uv run main.py --structure fixtures/structure_house.json
```

## Tune-before-demo checklist

- [ ] `POST /gripper {"open":true}` actually opens (flip `grip_open`/
      `grip_closed` in mc_skills if backwards)
- [ ] `POST /goto` reaches the build zone; set `DOWN_QUAT` so the gripper is
      flat/down
- [ ] `contracts.py`: `BOX_SIZE` = real box edge; `GRID_ORIGIN` = measured
      position of cell (0,0,0) center
- [ ] ArUco markers printed + taped, ids 0..N-1
- [ ] `camera.points` alignment sanity-checked against a known box position
- [ ] `OPENAI_API_KEY` set if using the llm backend

## Fallback ladder (if something stalls, move down, don't get stuck)

- structure: live MC hook -> `.nbt` export -> grid UI -> fixture json
- detection: ArUco -> color blobs -> hardcoded positions
- planner: llm -> deterministic -> hand-ordered plan.json
- motion: IK pick/place -> /goto waypoints driven by curl -> mimic
  record/playback of a demonstrated pick
