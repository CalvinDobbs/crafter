# Which robot commands get a voice line

Every public command on the body server, and whether it speaks. Checked 2026-09-12.

**Two copies, and they differ.** Diff them before assuming a route exists:

| Copy | Where | Routes |
|---|---|---|
| Deployed | `~/bbapps/mc_skills/main.py` on bracketbot-184 | 12 |
| In-repo (voice work) | [mc_skills_main.py](mc_skills_main.py) | 11 |

The deployed copy additionally has `POST /voice_script` and loads a `voice_script.json` at startup if one sits beside it. The in-repo copy has neither. Re-check with `grep '@app\.' ~/bbapps/mc_skills/main.py` on the robot, and add a row here if a route appears.

**Client:** [skills_client.py](../skills_client.py) — `health`, `home`, `park`, `pick`, `place`, `goto`, `gripper`, `rotate`, `say`, `celebrate`. It has no `voice_script` method.

This maps the **legacy one-shot body-server path**. The default CLI mode is now the reasoning agent, whose model-visible operations are `look_around`, `approach_box`, `pickup`, `move_to_build` and `place` — see [bot-code/README.md](../README.md). Coordinate `mc_skills` edits with its owner; it is shared with other apps on the robot.

---

## What “a command” means

Two layers:

1. **HTTP** — what `SkillsClient` / `curl` call (`POST /pick`, …).
2. **Inside pick/place** — several real moves (`goto`, `gripper`) already run in order. Those inner moves are what you hear, not a single “I am picking” at the start.

Helpers that are **not** commands (no speech): `Arm.spec`, `Arm.live`, `Arm._ramp_to`, `get_body`, `_run`, `_shutdown`. They only support `goto` / homing.

---

## Every HTTP route (12 deployed, 11 in-repo)

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
| 12 | POST | `/say` | Playback only | Does **not** choose a line; plays whatever text it is given | Calls `hook.say_text` → `SpeechQueue`. Non-blocking: it does not take the motion lock |
| 13 | POST | `/voice_script` | No | — | **Deployed copy only.** Loads laptop-generated lines into the narrator; no motion lock |

That is 13 rows for 12 routes: `/gripper` gets two rows because open and close speak different lines. Rows 1–12 are the in-repo copy's 11 routes; row 13 exists only on the robot. No other `@app` routes in either. The neutral lines quoted above match `narrator.py`; `home.park` has no template yet, which is why `/park` is silent.

---

## Inside `POST /pick` (5 real moves)

Code order in `Body.pick`:

| Step | Code | Speaks? | Event | Line |
|---|---|---|---|---|
| 1 | `gripper(True)` — open claw before reaching | **No** | — | Prep; talking here would say “opening” while starting a pick |
| 2 | `goto(above)` — J0/IK to hover | Yes | `pick.approach` | “J0 slide toward box {id}.” |
| 3 | `goto(pos)` — drop onto the box | Yes | `pick.descend` | “Descending to the box.” |
| 4 | `gripper(False)` — close J7 | Yes | `pick.grasp` | “Closing the J7 gripper.” |
| 5 | `goto(above)` — lift | Yes | `pick.lift` | “J0 vertical stage lifting.” |
| fail | approach or descend IK returns false | Yes | `fail` | “Missed that. Gripper opening.” |

`{id}` is only known if the orchestrator passes it later; inner `pick(pos)` only has xyz today.

## Inside `POST /place` (4 real moves)

| Step | Code | Speaks? | Event | Line |
|---|---|---|---|---|
| 1 | `goto(above)` | Yes | `place.approach` | “Arm bending to layer {y}.” |
| 2 | `goto(pos)` descend | Yes | `place.descend` | “Descending to the cell.” |
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
| `pick.descend` / `place.descend` | Yes (inner `goto`) | **Yes** | Templates now exist in `narrator.py` and `flavor.py` |
| LLM flavor lines | — | Yes | `flavor.py`; packs select the wording |
| `main.py` orchestrator `skills.say()` | Different file | Legacy | Superseded by per-command lines and, for the default mode, by the agent loop |

---

## Coverage check

In-repo: 2 GETs + 9 POSTs = 11 handlers. Deployed adds `/voice_script` for 12. All listed.  
`Body` methods: `home`, `park`, `pick`, `place`, `rotate`, `celebrate` — all listed.  
`Arm` motion: `goto`, `gripper` — listed as inner pick/place steps and as their own POST routes.

**How to re-check:** `grep '@app\.' ~/bbapps/mc_skills/main.py` on the robot, and the same against [mc_skills_main.py](mc_skills_main.py). If they diverge further, update the table at the top.
