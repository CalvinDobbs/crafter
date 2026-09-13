# Which robot commands get a voice line

This file is the full map for **§8**. It looks short in git because the first draft was only tables. Every public command on the robot’s body server is listed below.

**Source of truth:** `~/bbapps/mc_skills/main.py` on bracketbot-184 (~325 lines).  
**Client:** `bot-code/skills_client.py` (same names: `home`, `park`, `pick`, `place`, `goto`, `gripper`, `rotate`, `say`, `celebrate`, `health`).

Do **not** edit `mc_skills` until **8.3** is approved.

---

## What “a command” means

Two layers:

1. **HTTP** — what `SkillsClient` / `curl` call (`POST /pick`, …).
2. **Inside pick/place** — several real moves (`goto`, `gripper`) already run in order. Those inner moves are what you hear, not a single “I am picking” at the start.

Helpers that are **not** commands (no speech): `Arm.spec`, `Arm.live`, `Arm._ramp_to`, `get_body`, `_run`, `_shutdown`. They only support `goto` / homing.

---

## Every HTTP route (11)

| # | Method | Path | Speaks? | Event / line | Why |
|---|---|---|---|---|---|
| 1 | GET | `/health` | No | — | Status probe |
| 2 | GET | `/state` | No | — | Returns joint positions |
| 3 | POST | `/home` | Yes | `home.start` → “Staged home, J0 homing.” | `staged_home_arms` (torque on + J0) |
| 4 | POST | `/park` | **Not yet** | propose `home.park` → “Parking the arms.” | `park_arms`; we skipped this in v1 |
| 5 | POST | `/gripper` `open:true` | Yes | `place.release` → “Opening the J7 gripper.” | Direct claw open |
| 6 | POST | `/gripper` `open:false` | Yes | `pick.grasp` → “Closing the J7 gripper.” | Direct claw close |
| 7 | POST | `/goto` | **No** | — | Raw IK move. Pick/place already narrate their own `goto`s. A lone `/goto` would be generic; skip unless we add `goto.generic` later |
| 8 | POST | `/pick` | Yes | **sequence** in the next section | Full grasp |
| 9 | POST | `/place` | Yes | **sequence** below | Full set-down |
| 10 | POST | `/rotate` | Yes | `scan.start` → “Head camera scanning for boxes.” | Base twist (`drive.ctrl`); used while scanning |
| 11 | POST | `/celebrate` | Yes | `build.done` → “Done. Still in balance mode.” | LED flash only |
| 12 | POST | `/say` | Playback only | Does **not** choose a line; plays whatever text it is given | Stub today (prints). Later: speaker |

That is the **entire** public API. There are no other `@app` routes.

---

## Inside `POST /pick` (5 real moves)

Code order in `Body.pick`:

| Step | Code | Speaks? | Event | Line |
|---|---|---|---|---|
| 1 | `gripper(True)` — open claw before reaching | **No** | — | Prep; talking here would say “opening” while starting a pick |
| 2 | `goto(above)` — J0/IK to hover | Yes | `pick.approach` | “J0 slide toward box {id}.” |
| 3 | `goto(pos)` — drop onto the box | Yes | `pick.descend` **(new)** | Propose: “Descending to the box.” |
| 4 | `gripper(False)` — close J7 | Yes | `pick.grasp` | “Closing the J7 gripper.” |
| 5 | `goto(above)` — lift | Yes | `pick.lift` | “J0 vertical stage lifting.” |
| fail | approach or descend IK returns false | Yes | `fail` | “Missed that. Gripper opening.” |

`{id}` is only known if the orchestrator passes it later; inner `pick(pos)` only has xyz today.

## Inside `POST /place` (4 real moves)

| Step | Code | Speaks? | Event | Line |
|---|---|---|---|---|
| 1 | `goto(above)` | Yes | `place.approach` | “Arm bending to layer {y}.” |
| 2 | `goto(pos)` descend | Yes | `place.descend` **(new)** | Propose: “Descending to the cell.” |
| 3 | `gripper(True)` open J7 | Yes | `place.release` | “Opening the J7 gripper.” |
| 4 | `goto(above)` retreat | Yes | `place.retreat` | “Retreating to approach height.” |
| fail | IK fail | Yes | `fail` | Same fail line |

`{y}` needs the orchestrator (cell layer). `place(pos)` is xyz only today.

---

## Gaps (intentional, not missing routes)

| Thing | Covered as a route? | Spoken in v1? | Notes |
|---|---|---|---|
| `/park` | Yes | No | Need a new event if we want a line |
| `/goto` alone | Yes | No | Avoid double-talking with pick/place |
| Pick step 1 open gripper | Yes (inner) | No | Would confuse “opening” at start of pick |
| `pick.descend` / `place.descend` | Yes (inner `goto`) | **Templates not in `narrator.py` yet** | Add at 8.4 if you approve |
| LLM / ElevenLabs | — | No | §6–§7 |
| `main.py` orchestrator `skills.say()` | Different file | Old path | §8 replaces that with per-command lines |

---

## Coverage check

Public HTTP: 2 GETs + 9 POSTs = 11 handlers. All listed.  
`Body` methods: `home`, `park`, `pick`, `place`, `rotate`, `celebrate` — all listed.  
`Arm` motion: `goto`, `gripper` — listed as inner pick/place steps and as their own POST routes.

**How to re-check:** `grep '@app\.' ~/bbapps/mc_skills/main.py` on the robot. If a new route appears, add a row here before 8.4.
