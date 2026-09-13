# Moving the base and using the arms

Source-checked on `bracketbot-184` (`100.66.148.86`) on 2026-09-12 against the deployed `bbos` daemons, `bbapps` sample apps, and this repository's [pickup.py](pickup.py). The review was read-only: no writer was opened, no motion was commanded, and no daemon or service was started, stopped, or restarted.

Everything below runs **on the bot**. From the Windows PC, connect first with your approved SSH identity; see [the project access notes](../../AGENTS.md). Non-interactive SSH does not put `uv` on `PATH` — use `/home/bracketbot/.local/bin/uv`.

> **Before you command any motion.** The arms and base are shared hardware with a single writer each. Confirm with the robot owner that nothing else owns `drive.ctrl` or `arm_*.ctrl` first, and keep a hand on the power. Nothing in this guide should be pasted into a terminal unread.

## 1. The API surface

Everything goes through BBOS shared-memory topics, not through a robot SDK object:

```python
from bbos import Reader, Writer, Type, Config
```

- Topics are POSIX shared-memory segments at `/dev/shm/<topic.name>`: lock-free ring buffers, **one writer, many readers**. A second `Writer` on a live topic raises `RuntimeError` naming the owning PID.
- `Writer(name, Type("<type>"), keeptime=True)` — write with `with w.buf() as b: b["field"] = ...`.
- `Reader(name)` — `r.ready()` reports fresh data and handles writer death/recreation; then read `r.data["field"]`.
- `keeptime=True` (the default) paces the loop to the type's declared `@realtime(ms)` period. `keeptime=False` plus your own `time.sleep` lets you self-pace, which is what a fixed-rate motion loop usually wants.
- `Config("<daemon>")` resolves that daemon's registered constants. Read limits from `Config` rather than copying the numbers in this guide into your code.

The numbers quoted here are the values that were live on this robot at review time. They are documentation, not a contract.

## 2. Moving the base

### 2.1 The topic

| Property | Value |
|---|---|
| Topic | `drive.ctrl` |
| Type | `Type("drive_ctrl")` |
| Period | 10 ms (`@realtime(ms=10)`) |
| Fields | `twist` `float32[2]`, `twist_torque` `float32[2]` |
| `twist[0]` | forward linear velocity `v`, m/s; positive is forward |
| `twist[1]` | yaw rate `w`, rad/s; positive is intended as CCW / left |

```python
with Writer("drive.ctrl", Type("drive_ctrl")) as w_drive:
    with w_drive.buf() as b:
        b["twist"] = np.array([v, w], dtype=np.float32)
```

`mc_skills.rotate()` documents the yaw sign as "positive rad = CCW (verify sign on robot)". Treat the sign as unverified until you have confirmed it on the physical machine with the owner present.

### 2.2 Limits, timeout and rate

From `Config("base")` and `Config("drive")` in `bbos/daemons/base/constants.py`:

| Constant | Value | Meaning |
|---|---|---|
| `base.v_max` | 0.3 m/s | daemon clamp on `v` |
| `base.w_max` | 1.0 rad/s | daemon clamp on `w` |
| `base.command_hz` | 50.0 | the host publish rate the daemon expects |
| `base.drive_command_timeout_s` | 0.1 | no fresh command within this → the daemon zeroes |
| `drive.max_angular_vel` | 0.9 rad/s | the drive config's own, lower, yaw cap |
| `drive.robot_width` | 0.3275 m | wheel separation |
| `drive.wheel_diam` | 0.165 m | wheel diameter |

The base daemon clamps every command into `±v_max` / `±w_max` and treats a **non-finite** `v` or `w` as zero. It also zeroes when the topic is unreadable or when `drive_command_timeout_s` elapses without a fresh command.

Two consequences for your control loop:

1. **Publish continuously.** A twist is not a destination; it is a velocity that expires in 100 ms. Publish faster than 10 Hz — 50 Hz is the configured host rate, and both `mc_skills.rotate()` and the greeter's WASD thread sit in that range.
2. **Zero explicitly on the way out.** Let the timeout be your backstop, not your brake. Publish `[0.0, 0.0]` on cancellation, on exception, and in your `finally` block before the writer closes.

### 2.3 Balance modes

The baseboard owns a 200 Hz balance loop; apps only publish velocity. Mode lives on a separate topic:

| Property | Value |
|---|---|
| Topic | `base.mode`, 50 ms |
| Fields | `mode` `uint8`, `lean_angle_deg` `float32` |
| `MODE_BALANCE` | `0` — normal balancing; this is `base.default_mode` |
| `MODE_LEAN` | `1` — specialized |
| `MODE_TWIST` | `2` — commented in the source as "velocity, NOT balancing" |

Mode commands expire after `base.mode_command_timeout_s` (0.25 s), after which the daemon falls back to `default_mode` and logs the change. Invalid mode values are ignored.

**Publishing a `twist` field does not require `MODE_TWIST`.** The field name and the mode are unrelated: normal balanced driving publishes `twist` while the base stays in `MODE_BALANCE`. Do not select `MODE_TWIST` from a generic action provider without the platform owner's approval.

### 2.4 Who else wants `drive.ctrl`

At review time these all open the same single-writer topic:

- `bbapps/teleop.py` — keyboard teleop; converts left/right wheel velocities to a body twist with `R = robot_width * 0.5`
- `bbapps/nav/main.py` — autonomous navigation
- `bbapps/quest_teleop/main.py` — Quest teleop
- `bbapps/mc_skills/main.py` — the body server this project talks to
- `bbapps/greeter/main.py`, `bbapps/tune_odrive.py`

Only one may run at a time. **Do not resolve a writer conflict by killing the owner or restarting a BBOS service** — find out what is running and why.

### 2.5 Two reference implementations

`mc_skills.rotate(rad)` is the minimal open-loop primitive: compute a duration from a fixed rotate speed, publish `[0, w]` at 50 Hz for that long, then publish `[0, 0]`. It has no localization feedback, no obstacle check and no arrival criterion. It is a scan gesture, not navigation.

`bbapps/nav/main.py` is the autonomy reference, and is worth reading before writing any `move_to_build`. Its tuned parameters at review time:

| Parameter | Value |
|---|---|
| `SPEED` | 0.08 m/s |
| `MAX_OMEGA` | 0.15 rad/s |
| `GOAL_TOLERANCE` | 0.25 m |
| `CLEAR_MIN` / `CLEAR_FULL` | 0.3 m / 0.6 m wall clearance band |
| `SMOOTH_V` / `SMOOTH_W` | 0.5 / 0.6 velocity low-pass |
| `STUCK_TIME` | 15 s without progress before rotating to rescan |
| `TELEOP_TIMEOUT` | 0.4 s manual dead-man |

Autonomous speeds are roughly a quarter of the daemon's clamp. The clamp is a hardware limit, not a target. Nav also does map-frame pure-pursuit control, obstacle inflation, bounds checks and stuck detection — the parts that make motion safe are the parts that are not the twist publish.

## 3. Using the arms

### 3.1 Topics

Both arms are 8-DoF, addressed as `arm_left` / `arm_right`. **Units on `.ctrl` and `.state` are motor turns, not URDF radians.**

| Topic | Period | Fields |
|---|---|---|
| `<arm>.state` | 15 ms | `pos`, `vel`, `torque`, `temp`, `current` — each `float32[8]` |
| `<arm>.ctrl` | 15 ms | `pos[8]`, `vel[8]`, `tau[8]`, `alpha` |
| `<arm>.torque` | `@state` | `enable` `bool[8]`, `tau_mode` `bool[8]`, `compliance_mode`, `axis_aligned`, `force_only`, `j0_homing`, `calibrating` |
| `<arm>.target` | 15 ms | pre-IK cartesian setpoint; recorded by the dataset daemon, **not read by the arm daemon** |

Joint index conventions:

- **J0** is the vertical lift stage (a carriage, not a revolute joint). `Config("arm_<side>").wheel_radius` (0.0465 m) converts its turns to metres.
- **J2** is the shoulder swing that moves the end effector along base *y*, toward or away from the other arm.
- **J3** is the elbow. `q2urdf(cfg.home)[3]` is +1.571 rad on both arms, so `cfg.home[3]` is the 90-degree elbow pose.
- **J5 / J6** are wrist yaw and pitch.
- **J7 (index 7)** is the gripper. It is not IK-driven.

The arm daemon runs at 150 Hz (`cfg.dt = 1/150`) and applies a per-joint EMA to commanded position (`cfg.lpf_alpha`, 0.2 by default). The `alpha` field on `.ctrl` overrides that smoothing per command.

### 3.2 Ownership, and what happens when you let go

One process owns `<arm>.ctrl` and `<arm>.torque`. Read-only tools must open **no** writers — `bbapps/examples/view_arms.py` is the model: without `--control` it opens readers only and never touches the arms; writers and the "Enable torque" checkbox appear only with the flag.

The arm daemon watches for its control writer. When that reader becomes unreadable it writes `Torque_Enable = 0` to every motor immediately, then re-sends that at 1 Hz.

> **This is the single most important fact for a loaded arm.** Closing the control writer — including by crashing, or by exiting a `with` block — drops the arm limp. If the arm is holding a box, the box falls. A load-preserving `stop()` cannot be implemented by closing writers.

### 3.3 Enabling torque without a jump

Only an OFF→ON transition reseeds the daemon's command filter to the live pose. The sequence, from `homing.staged_home_arms` and reproduced in [pickup.py](pickup.py):

1. Write `enable = False` for every joint.
2. Wait for fresh `<arm>.state` (`r_state.ready()`, then read `data["pos"]`).
3. Copy the **live** pose into `<arm>.ctrl.pos`.
4. Keep flushing that command for ~0.1 s (`ENABLE_FLUSH_S`).
5. Write `enable = True`.
6. Allow ~0.3 s for the daemon's mode-switch stall before moving (`ENABLE_SETTLE_S`).

Skipping the disable step means there is no OFF→ON edge, the filter is never reseeded, and the arm snaps from wherever it hangs to whatever stale command is in the buffer.

### 3.4 Homing and parking

Use `bbapps/quest_teleop/scripts/homing.py`:

- `staged_home_arms(specs, settle_s=0.4, rate_hz=200.0)` — homes every arm in two torque-enable stages, together, blocking. Each spec is a dict of `cfg`, an open `r_state`, `w_ctrl`, `w_torque`, and optional `tau_mode`.
- `park_arms(specs, ...)` — the shutdown reverse: descend, straighten, lower J0, cut torque. Pass the **same** startup `waypoints`; they are reversed internally.

Do not casually duplicate or simplify these. They handle limp-joint settling, the daemon's mode-switch stall, Catmull-Rom waypoint splines and per-segment durations. `PARK_STRAIGHTEN_S` exists because a bent arm dropped without straightening swings into the table.

### 3.5 Command clipping is not a safety system

The daemon clips each commanded position to the live position ±0.5 turns (`limits.clip_target`) and then into the calibrated range, logging `[CLIP]` when it does. That prevents a single wild setpoint from becoming a full-speed slew; it does **not** make a fast ramp safe.

Applications still generate their own smooth, bounded trajectories, and still wait for joints to actually arrive. `pickup.py`'s `settle_joint` is the pattern: hold the command until the joint is within tolerance, or stops moving (a hard stop or an obstacle), or times out.

### 3.6 Motor turns vs URDF, and mirroring

`.ctrl`/`.state` are in motor turns. IK works in URDF radians/metres. Each arm's `Config` carries the conversion:

```python
cfg = Config("arm_left")
urdf = cfg.q2urdf(turns)    # motor turns  -> URDF radians / metres
turns = cfg.urdf2q(urdf)    # URDF         -> motor turns
```

The arms are mirror images in motor-turn space, and the mirroring is **not** a single sign flip:

| | `arm_left` | `arm_right` |
|---|---|---|
| `ik_sign` (joints 0–6) | `[-1, 1, -1, 1, 1, -1, -1]` | `[-1, -1, -1, -1, 1, -1, 1]` |
| `gripper_sign` (J7) | `-1` | `1` |
| J0 in `q2urdf` | `q[0] * wheel_radius` | `-q[0] * wheel_radius` |

Always convert with the arm's own `q2urdf`/`urdf2q`. Never negate a motor command by hand to "make it the other arm", and never reuse one arm's constants for the other.

Per-robot joint ranges live in `~/bbos/bbos/daemons/arm_<side>/ranges.calibration.json` as `cal_min` / `cal_max` in motor turns. Load them from the robot you are running on. **Do not copy calibration between robots** — these are per-machine measurements, and `pickup.py` additionally keeps a `RANGE_MARGIN` of 0.02 turns inside every calibrated edge.

### 3.7 Cartesian moves: the IK pattern

`mc_skills.Arm.goto(pos, quat, secs)` is the reference. The order matters:

```python
live = self.live()                       # 1. live motor turns
q_now = self.cfg.q2urdf(live)[:7]        # 2. to URDF
self.cfg.ik.reset(list(q_now))           # 3. seed IK from where the arm actually is
sol = self.cfg.ik.solve(list(pos), list(quat))
if sol is None:
    return False                         #    unreachable: report it, do not approximate
full = self.cfg.q2urdf(live)
full[:7] = np.asarray(sol)[:7]           # 4. seven solved joints into the full vector
target = self.cfg.urdf2q(full)           # 5. back to motor turns
target[GRIPPER_IDX] = live[GRIPPER_IDX]  # 6. preserve the live gripper setpoint
self._ramp_to(target, secs)              # 7. ramp, never step
```

`pos` is base-frame XYZ in metres; `quat` is XYZW. Base frame is **+x forward, +y left, +z up**.

`cfg.ik.solve()` returns seven joints or `None`. `cfg.ik.fk(q)` gives forward kinematics — `pickup.py` uses it to *derive* directional signs rather than hard-coding them, which is the right instinct when the two arms mirror.

The reasoning layer works in world positions carrying frame and epoch metadata. **The action provider transforms into the current base frame and rejects stale or cross-epoch geometry before calling IK.** A pose solved against a map epoch that has since changed is a confident move to the wrong place.

## 4. What `pickup.py` actually does

[pickup.py](pickup.py) is a direct joint-space two-arm cage-and-lift prototype. It is not navigation, and it is not an `ActionProvider`. Its stage order and mode flags are under active development by the action owner — check the module docstring and `--help` rather than trusting any description, including this one.

Stages, in order:

1. Disable torque, flush ctrl to the live pose, enable torque (section 3.3).
2. Elbow (J3) to 90 degrees, on a quintic ease, held through the later stages so the forearm clears the table.
3. Grippers (J7) open to `GRIP_OPEN_FRAC` of the calibrated travel.
4. J0 to the top.
5. Spread: both arms swing J2 outward to the calibrated edge, straddling a box wider than the shoulders.
6. Prepare the wrists **while still at the top**: J5 yaw alone rotates inward toward the box, preserving the J6 pitch and open J7 claw setpoints. `--hook` caps J5 travel; `--hook 0` skips the stage, as does `--lower-only`. Yaw alone can change hand height, so this rotation happens before descent rather than sweeping across the box at floor level.
7. J0 down to the calibrated bottom plus `--bottom-margin` turns back toward the top (default: 1.0 turns). **That margin is a lift offset, not a measured floor clearance** — nothing here senses the floor. `--lower-only` stops and holds at this pose.
8. Pinch: J2 creeps inward until per-arm tracking error crosses `PINCH_CONTACT_ERR`, then holds `--squeeze` turns past contact, capped by calibration. **This is the default stopping point** — it holds the squeeze until Ctrl+C or `--hold`.
9. With `--pickup` only: close the grippers, cradle with extra elbow flex, then shoot J0 back to the top and hold.

Properties worth copying:

- Inward/outward signs are derived from FK per arm, never hard-coded.
- Per-robot calibration is loaded, with a range margin held inside every edge.
- It self-paces at 200 Hz with `keeptime=False`.
- Contact is detected per arm, so an off-centre box is still held from both sides.
- Travel is capped by both `max_travel` and the calibrated range of every moving joint.

Properties that disqualify it as an action implementation:

- **Contact thresholds and squeeze values are hardware-tuning assumptions, not possession evidence.** A tracking-error rise means something resisted the joint. It does not mean a box is held.
- Torque stays enabled through the hold, and the `finally` block disables torque and closes the writers. **If a box is held, exiting drops it.**
- It is a blocking script driven by Ctrl+C, with no receipt, no phase reporting and no cancellation contract.

Do not call `pickup.py` from the reasoning layer, and do not reuse its cleanup as an agent `cancel()` or `stop()`.

Its offline tests run against a fake `bbos` on virtual time and move no hardware:

```bash
python -B -m unittest discover -s bot-code/actions -p test_pickup.py -v
```

NumPy is required. Keep these separate from the reasoning tests.

## 5. Carrying this into an action provider

The provider must satisfy interface v2 — see [agent_types.py](../agent_types.py) and the "Agent component interface v2" section of [bot-code/README.md](../README.md). Model-visible operations stay semantic: `look_around`, `approach_box`, `pickup`, `move_to_build`, `place`. Raw joint targets, velocities, homing and arbitrary IK poses stay behind them.

Mapping this guide onto the lifecycle requirements:

| Requirement | What it means here |
|---|---|
| Bounded `status`/`state`/`stop` | never block them behind a blocking motion routine |
| Load-preserving cancel and stop | do not close `arm_*.ctrl`, and do not cut torque, while a box is held (section 3.2) |
| Independent possession evidence | not tracking error, not "the command completed", not a marker that vanished |
| Placement evidence after terminal completion | measured after the action reports terminal, never inferred from it |
| No blind replay | a timed-out or unknown `submit` must not be re-sent as motion |
| Immediate rechecks before effects | localization, obstacles, identity, reachability, frame and epoch — re-checked at the moment of the effect, not at admission |

`FunctionActions` can adapt ordinary functions to the interface, but it does not make a blocking, uninterruptible hardware routine safe. Cooperative cancellation, watchdogs and the zero-twist / hold-torque teardown remain the action owner's responsibility.

## 6. Do not, while verifying this guide

- Do not run `pickup.py`, or the body server's `/home`, `/goto`, `/pick`, `/place` or `/rotate` endpoints.
- Do not open a `drive.ctrl` or `arm_*.ctrl` writer to "just check".
- Do not resolve a writer conflict by killing the owning process or restarting a BBOS daemon.
- Do not combine real motion with mocked possession, occupancy or safety evidence.
- Do not treat any number in this guide as verified on *your* robot without re-reading its `Config` and calibration.

## Sources reviewed

On the bot, read-only, 2026-09-12:

- `bbos/bbos/daemons/base/constants.py`, `daemon.py`
- `bbos/bbos/daemons/arm_left/constants.py`, `daemon.py`, `limits.py`
- `bbos/bbos/daemons/arm_right/constants.py`
- `bbapps/AGENTS.md`
- `bbapps/teleop.py`
- `bbapps/nav/main.py`
- `bbapps/mc_skills/main.py`
- `bbapps/examples/view_arms.py`
- `bbapps/quest_teleop/main.py`, `quest_teleop/scripts/homing.py`

In this repository:

- [pickup.py](pickup.py), [test_pickup.py](test_pickup.py)
- [agent_types.py](../agent_types.py), [bot-code/README.md](../README.md)

Related: [playing sound through the speakers](../voice/speaker_playback.md).
