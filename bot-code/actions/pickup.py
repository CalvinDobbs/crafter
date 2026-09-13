# /// script
# dependencies = [
#   "bbos",
#   "numpy",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Initialize, spread, prepare the wrists, reach down, squeeze, then lift and hold at shoulder level.

Every run first reaches the same calibrated home angles with the lift raised and
claws open. This is a pose reset for an already-homed, unloaded robot, not encoder
homing. By default, prepare the wrists before descent, bring the arms inward, and
lift back to shoulder level while maintaining the squeeze. Hold there until Ctrl+C
(or --hold SECS). Torque stays on during the lift and hold; exiting disables torque
and releases the box. --lower-only skips wrist preparation, grasping and the final
lift, but not initialization or elbow extension. --pickup adds grip and cradle before
lifting. The bottom margin is a lift offset, not a measured floor distance.

Joint-space, no IK. Stages:
  1. initialize: elbow (J3) -> calibrated 90 deg, grippers (J7) open, then lift (J0) -> top;
     while raised, return J1, J2, J5 and J6 to each arm's configured home angles and
     roll J4 a mirrored 90 deg from home so the claw openings are vertical.
     Verify every joint reached the reference pose before starting the pickup.
  2. spread: both arms swing J2 (the sideways swing) OUT to the calibrated edge of their range,
     so the forearms straddle a box much wider than the shoulders
  3. prepare wrists: rotate J6 inward toward the box while J0 stays at the top, preserving
     the initialized J4 roll, J5 and open J7 claw setpoints; --hook caps J6 travel and 0 skips this stage
  4. extend elbows (J3) by --elbow-extension turns from home while raised (default: 30 deg);
     wait for both arms to arrive, then lower (J0) -> bottom minus --bottom-margin turns
     toward the top, stopping and settling at each hard-coded J0_WAYPOINT_FRACS stop on the way
     down instead of running the whole height in one sweep; --lower-only holds this reach pose
  5. cage the box, keeping the prepared wrist angles:
       a. pinch: J2 brings the elbows and forearms inward until each forearm meets the box side
          (tracking error rises), then holds --squeeze turns past contact through the lift
       b. with --pickup, grip: grippers close (on a rim/corner if there is one; otherwise they just stiffen the
          hand into a solid paddle -- the daemon's J7 current-relief loop keeps the grip gentle)
       c. with --pickup, cradle: the elbows flex a little from the reach pose, lifting the front edge of the box so it
          tilts back against the upper arms and the weight rests on the forearms
  6. lift J0 back up to the top (shoulder level) through the same hard-coded stops, settling at
     each one, preserving the grasp, and hold until termination
All other joints hold their initialized pose. Range edges come from the per-robot
ranges.calibration.json (motor turns, the arm_ctrl.pos frame); the "down", "outward" and
"inward" signs are per-arm (the arms are mirror images in motor-turn space) and are derived
from FK, never hard-coded.

Usage:  uv run pickup.py [--arm left|right|both] [--speed TURNS_PER_S] [--hold SECS]
                         [--bottom-margin TURNS] [--elbow-extension TURNS] [--spread TURNS] [--squeeze TURNS]
                         [--lower-only | --pickup] [--hook TURNS] [--cradle TURNS]
"""
import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np

from bbos import Reader, Writer, Type, Config

J0 = 0
SHOULDER = 1            # shoulder pitch: raises / lowers the whole arm, elbow tip included
SWING = 2               # shoulder swing: moves the EE along base y (toward / away from the other arm)
ELBOW = 3               # q2urdf(cfg.home)[3] is +1.571 rad on both arms: home[3] IS the 90 deg elbow
WRIST_YAW = 5           # turns the hand about base z: +-0.25 turns = +-90 deg, sweeps the EE along base y
WRIST_PITCH = 6         # tips the hand about base y; blended with J5 so the hook stays level once J2 is swung out
GRIPPER = 7
WRIST_ROLL = 4
WRIST_ROLL_TURNS = 0.25
RATE_HZ = 200.0
J0_SPEED = 0.4          # turns/s along the lift (matches homing.J0_PARK_DOWN_SPEED)
J0_BOTTOM_MARGIN = 1.0
# Hard-coded intermediate J0 stops for both vertical moves (descent to the low pose and the loaded
# ascent), as fractions of the travel from the start pose to that move's target. The lift stops and
# settles at each one instead of running the whole height in a single sweep, so a motor that trips
# its current protection is waited for at the next stop rather than falling a whole lift behind.
# Fractions, not absolute turns: mirrored arms with unequal travel still cross every stop together.
J0_WAYPOINT_FRACS = (0.25, 0.5, 0.75, 1.0)
J0_WAYPOINT_SETTLE_S = 0.3  # dwell at each intermediate stop before the next segment starts
ELBOW_EXTENSION = 45.0 / 360.0  # straighter elbows reach farther forward, but on their own they only
                                # move the low elbow tip forward -- SHOULDER_LIFT is what raises it
SHOULDER_LIFT = 15.0 / 360.0    # turns J1 lifts with the extended reach so the elbow tip clears the
                                # chassis; paired with the extension, not a substitute for it
SHOULDER_SPEED = 0.05           # turns/s; J1 carries the whole arm, so ease it like the elbow
ELBOW_CLEARANCE = 2.5 / 360.0  # extra right-arm J3 reach: its forearm rides a lower chassis member
ELBOW_EXTENSION_SPEED = 0.05
INITIALIZE_SPEED = 0.08
LIFT_SPEED = 0.4        # turns/s for the loaded ascent
LIFT_ACCEL = 0.1
LIFT_FEEDBACK_MAX_AGE = 0.25
LIFT_LOG_INTERVAL_S = 0.5
ELBOW_SPEED = 0.15      # turns/s bending the elbow (~0.25 turns in ~1.7 s)
ELBOW_SETTLE_S = 0.5    # let the forearm stop swinging before the lift moves
TOP_SETTLE_S = 0.5      # rest at the top before descending
ENABLE_FLUSH_S = 0.1    # ctrl flushed to the live pose before torque comes on
ENABLE_SETTLE_S = 0.3   # daemon mode-switch stall after torque on
ARRIVE_TOL = 0.03       # turns; joint counts as arrived within this of its command
ARRIVE_TIMEOUT_S = 6.0  # give up waiting for a joint to arrive after this
STALL_EPS = 0.003       # turns; less motion than this for STALL_S = the joint is stuck/at a stop
STALL_S = 0.75
RANGE_MARGIN = 0.02     # turns; stay this far inside every calibrated joint range
SPREAD_SPEED = 0.15     # turns/s swinging the whole arm out (J2 carries the arm, so ease it)
SPREAD_SETTLE_S = 0.5   # let the arms stop swinging before the lift moves
PINCH_SPEED = 0.05      # turns/s J2 creep toward the box
PINCH_CONTACT_ERR = 0.010   # turns of J2 tracking error that counts as touching the box (no-load ~0.002);
                            # detecting earlier stops the swing on less deflection, so less force
PINCH_SQUEEZE = 0.01   # turns commanded past the contact point: a steady spring squeeze. TUNE on box
PINCH_SETTLE_S = 0.5    # let the squeeze load up before hooking
PINCH_CENTER_MARGIN = 0.03  # turns of J2 kept outboard of home: the travel backstop, not a force
# J2 force-regulated pinch. J2 cannot run in tau mode (arm_left.tau_mode_allowed is False for the
# ST-3120 joints), so the grip stays in position mode and the measured motor current is the force
# signal. Current at a stalled J2 also carries gravity and arm geometry, so CALIBRATE these at the
# actual grasp pose: log current[SWING] on a free-air pinch first for the no-load baseline.
PINCH_CURRENT_A = 0.3   # A of J2 current that counts as pressing the box (clamp is ~1.96 A). TUNE
PINCH_CURRENT_DB = 0  # A deadband around the hold current, so the regulator does not chatter
PINCH_RELIEF_STEP = 5e-4    # turns/tick the J2 goal walks to shed or restore grip (as j7_relief_step)
PINCH_RELIEF_MAX = 0.05     # turns; cap on the accumulated backoff from the contact command
HOOK_SPEED = 0.08       # turns/s wrist creep, hand turning in across the front of the box
HOOK_MAX_TRAVEL = 0.22  # turns (~80 deg) of wrist travel when nothing stops the hand earlier
HOOK_CONTACT_ERR = 0.015    # turns of wrist tracking error that counts as the hand touching the box
HOOK_SQUEEZE = 0.01     # turns past the hook contact point. TUNE on box
GRIP_SPEED = 0.4        # turns/s closing / opening the gripper (~0.33 turns in <1 s)
GRIP_OPEN_FRAC = 0.67    # how far toward the calibrated open stop the jaws open: fully wide, so the
                        # open hands can palm a box from both sides (clamped RANGE_MARGIN inside the
                        # stop, since initialization requires the jaws to actually reach the command)
GRIP_SETTLE_S = 0.5     # let the jaws seat before tilting
CRADLE_TILT = 0.03      # turns (~11 deg) of extra elbow flex from the reach pose to lift the box's front edge. TUNE
CRADLE_SPEED = 0.05     # turns/s; slow so the box rolls back onto the forearms, not out of them
CRADLE_SETTLE_S = 0.5   # let the load settle on the forearms before lifting

_stop = False


def _sigint(*_):
    global _stop
    _stop = True


signal.signal(signal.SIGINT, _sigint)


def smoothstep(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def smootherstep(s):
    """Quintic ease: zero velocity and acceleration at both ends, for joints that jerk."""
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (s * (6.0 * s - 15.0) + 10.0)


def load_cal(side):
    cal = json.loads(Path(
        f"/home/bracketbot/bbos/bbos/daemons/arm_{side}/ranges.calibration.json"
    ).read_text())
    return np.array(cal["cal_min"], dtype=np.float64), np.array(cal["cal_max"], dtype=np.float64)


def j0_extremes(cfg, cal_min, cal_max):
    """(top, bottom) J0 positions in motor turns for this arm."""
    ends = np.array([cal_min[J0], cal_max[J0]])
    # The first startup waypoint is always below home, so it gives the "down" sign.
    down = float(np.sign(float(cfg.startup_waypoints[0][J0]) - float(cfg.home[J0]))) or 1.0
    return float(ends[np.argmin(ends * down)]), float(ends[np.argmax(ends * down)])


def j0_low_target(top, bottom, margin):
    return float(bottom + np.sign(top - bottom) * min(margin, abs(top - bottom)))


def j0_waypoint(start, target, frac):
    """The intermediate J0 stop ``frac`` of the way from ``start`` to ``target``. A full fraction
    returns ``target`` itself, so the last stop is the calibrated end and not float arithmetic."""
    return float(target if frac >= 1.0 else start + frac * (target - start))


def elbow_extension_for(side, extension, clearance=ELBOW_CLEARANCE):
    """Per-arm J3 extension. The right forearm has to clear a lower chassis member, so it
    reaches farther than the left; ``reach_elbow`` still keeps it short of straight."""
    return extension + (clearance if side == "right" else 0.0)


def up_sign(cfg, turns, joint):
    """Sign of a step on ``joint`` that raises this arm's EE (base z up)."""
    q = np.asarray(turns, dtype=np.float64).copy()
    p0, _ = cfg.ik.fk(list(cfg.q2urdf(q.copy())[:7]))
    q[joint] += 0.05
    p1, _ = cfg.ik.fk(list(cfg.q2urdf(q.copy())[:7]))
    return float(np.sign(p1[2] - p0[2])) or 1.0


def inward_sign(cfg, turns, joint):
    """Sign of a step on ``joint`` that moves this arm's EE toward the robot centreline (base y -> 0)."""
    q = np.asarray(turns, dtype=np.float64).copy()
    p0, _ = cfg.ik.fk(list(cfg.q2urdf(q.copy())[:7]))
    q[joint] += 0.05
    p1, _ = cfg.ik.fk(list(cfg.q2urdf(q.copy())[:7]))
    return -float(np.sign((p1[1] - p0[1]) * p0[1])) or 1.0


class Arm:
    def __init__(self, side):
        self.side = side
        self.cfg = Config(f"arm_{side}")
        self.dof = self.cfg.dof
        self.r_state = Reader(f"arm_{side}.state", keeptime=False)
        self.w_ctrl = Writer(f"arm_{side}.ctrl", Type("arm_ctrl"), keeptime=False)
        self.w_torque = Writer(f"arm_{side}.torque", Type("arm_torque"))
        self.cal_min, self.cal_max = load_cal(side)
        self.lo, self.hi = np.minimum(self.cal_min, self.cal_max), np.maximum(self.cal_min, self.cal_max)
        self.top, self.bottom = j0_extremes(self.cfg, self.cal_min, self.cal_max)
        self.elbow_90 = float(self.cfg.home[ELBOW])
        # The gripper's calibrated stop at ~0 turns is the jaws closed; the far stop is fully open
        # (quest_teleop's GRIPPER_OPEN_POS lands ~1/3 of the way toward it).
        ends = np.array([self.cal_min[GRIPPER], self.cal_max[GRIPPER]])
        self.grip_closed = float(ends[np.argmin(np.abs(ends))])
        open_stop = float(ends[np.argmax(np.abs(ends))])
        self.grip_open = float(np.clip(self.grip_closed + GRIP_OPEN_FRAC * (open_stop - self.grip_closed),
                                       self.edge(GRIPPER, -1), self.edge(GRIPPER, 1)))
        self.cmd = None

    def edge(self, joint, sign, margin=RANGE_MARGIN):
        """The calibrated end of ``joint``'s range in the ``sign`` direction, ``margin`` inside it."""
        return self.hi[joint] - margin if sign > 0 else self.lo[joint] + margin

    def initial_pose(self):
        pose = np.asarray(self.cfg.home, dtype=np.float64).copy()
        if pose.shape != (self.dof,) or not np.all(np.isfinite(pose)):
            raise ValueError(f"{self.side}: invalid calibrated home pose")
        pose[J0], pose[GRIPPER] = self.top, self.grip_open
        pose[WRIST_ROLL] += float(np.sign(self.elbow_90) or 1.0) * WRIST_ROLL_TURNS
        if not np.all(np.isfinite(pose)) or np.any(pose < self.lo) or np.any(pose > self.hi):
            raise ValueError(f"{self.side}: initialization pose is outside calibrated ranges")
        if not self.edge(WRIST_ROLL, -1) <= pose[WRIST_ROLL] <= self.edge(WRIST_ROLL, 1):
            raise ValueError(f"{self.side}: wrist roll target is too close to a calibrated stop")
        return pose

    def reach_elbow(self, extension):
        if not 0 <= extension <= abs(self.elbow_90):
            raise ValueError(f"{self.side}: elbow extension must stay between home and straight")
        return self.cradle(-extension)

    def cradle(self, tilt, *, start=None):
        """Flex ``tilt`` turns from ``start`` (default: 90 deg), respecting calibrated elbow stops."""
        target = (self.elbow_90 if start is None else start) + float(np.sign(self.elbow_90) or 1.0) * tilt
        if not self.edge(ELBOW, -1) <= target <= self.edge(ELBOW, 1):
            raise ValueError(f"{self.side}: elbow target is outside calibrated range margins")
        return target

    def live(self):
        while not self.r_state.ready():
            if _stop:
                raise KeyboardInterrupt
            time.sleep(0.005)
        return np.array(self.r_state.data["pos"], dtype=np.float64)

    def latest_state(self):
        self.r_state.ready()
        return self.r_state.data.copy() if self.r_state.readable and self.r_state.data is not None else None

    def write_cmd(self, pos):
        self.cmd = np.asarray(pos, dtype=np.float64)
        with self.w_ctrl.buf() as b:
            b["pos"][:] = self.cmd.astype(np.float32)
            b["vel"][:] = np.zeros(self.dof, dtype=np.float32)
            b["tau"][:] = np.zeros(self.dof, dtype=np.float32)
            b["alpha"] = 0.0

    def set_torque(self, on):
        with self.w_torque.buf() as b:
            b["enable"][:] = np.full(self.dof, on, dtype=np.bool_)
            b["tau_mode"][:] = np.zeros(self.dof, dtype=np.bool_)
            b["compliance_mode"] = False

    def close(self):
        self.w_ctrl.__exit__(None, None, None)
        self.w_torque.__exit__(None, None, None)


def hold(arms, secs):
    dt = 1.0 / RATE_HZ
    t0 = time.monotonic()
    while not _stop and (secs is None or time.monotonic() - t0 < secs):
        for a in arms:
            a.write_cmd(a.cmd)
        time.sleep(dt)


def ramp_joint(arms, joint, targets, speed, ease=smoothstep):
    """Ease ``joint`` of every arm to its target; all other joints hold."""
    dt = 1.0 / RATE_HZ
    froms = [a.cmd.copy() for a in arms]
    dist = max(abs(t - p0[joint]) for p0, t in zip(froms, targets))
    duration = max(dist / speed, 1e-3)
    t0 = time.monotonic()
    while not _stop:
        t = time.monotonic() - t0
        f = ease(t / duration)
        for a, p0, target in zip(arms, froms, targets):
            pos = p0.copy()
            pos[joint] = p0[joint] + f * (target - p0[joint])
            a.write_cmd(pos)
        if t >= duration:
            break
        time.sleep(dt)


def sample_lifts(arms):
    states = [a.latest_state() for a in arms]
    now = np.datetime64(round(time.time() * 1e9), "ns")
    ages = [float((now - s["timestamp"]) / np.timedelta64(1, "s")) if s is not None else np.inf
            for s in states]
    return states, ages


def valid_lift_sample(state, age):
    return (state is not None and 0 <= age <= LIFT_FEEDBACK_MAX_AGE
            and np.all(np.isfinite(state["pos"]))
            and np.all(np.isfinite([state["vel"][J0], state["current"][J0]])))


def lifts_arrived(arms, states, ages):
    return all(valid_lift_sample(s, age) and abs(s["pos"][J0] - a.cmd[J0]) < ARRIVE_TOL
               for a, s, age in zip(arms, states, ages))


def log_lifts(arms, states, ages, phase, elapsed):
    details, heights = [], []
    for a, state, age in zip(arms, states, ages):
        if state is None:
            details.append(f"{a.side} cmd={a.cmd[J0]:+.3f} feedback=missing")
            continue
        valid = valid_lift_sample(state, age)
        details.append(f"{a.side} cmd={a.cmd[J0]:+.3f} pos={state['pos'][J0]:+.3f}"
                       f" vel={state['vel'][J0]:+.3f} cur={state['current'][J0]:+.2f}A"
                       f" age={age:.3f}s {'fresh' if valid else 'invalid/stale'}")
        if valid:
            heights.append(np.sign(a.bottom - a.top) * (state["pos"][J0] - a.top)
                           * 2 * np.pi * a.cfg.wheel_radius)
    skew = f"{max(heights) - min(heights):.4f}m" if len(heights) == len(arms) else "unavailable"
    print(f"[pickup] lift {phase} wall={time.time():.3f} t={elapsed:.3f}s "
          + " | ".join(details) + f" | height_skew={skew}", flush=True)


def settle_lift(arms, timeout=ARRIVE_TIMEOUT_S, *, phase="settling"):
    t0 = next_log = time.monotonic()
    while not _stop:
        for a in arms:
            a.write_cmd(a.cmd)
        states, ages = sample_lifts(arms)
        now = time.monotonic()
        arrived = lifts_arrived(arms, states, ages)
        finished = arrived or now - t0 >= timeout
        if finished or now >= next_log:
            status = ("ready" if arrived else "incomplete") if finished else "waiting"
            log_lifts(arms, states, ages, f"{phase} {status}", now - t0)
            next_log = now + LIFT_LOG_INTERVAL_S
        if finished:
            return arrived
        time.sleep(1.0 / RATE_HZ)
    return False


def lift_segment(arms, targets, speed, accel, label):
    """Timed quintic on J0 to ``targets``, one segment of the ascent. Publishing never waits on
    feedback; the caller settles at the waypoint afterwards."""
    froms = [a.cmd.copy() for a in arms]
    dist = max(abs(t - p0[J0]) for p0, t in zip(froms, targets))
    # 1.875 = 15/8, smootherstep's peak derivative: anything smaller undersizes the ramp and the
    # real peak speed overshoots `speed`. Use --lift-speed to go faster, not this factor.
    duration = max(1.875 * dist / speed, np.sqrt((10 / np.sqrt(3)) * dist / accel))
    print(f"[pickup] lift {label} start wall={time.time():.3f} duration={duration:.3f}s "
          f"peak_speed<={speed:.3f} turns/s acceleration<={accel:.3f} turns/s^2", flush=True)
    t0 = next_log = time.monotonic()
    while not _stop:
        now = time.monotonic()
        elapsed = now - t0
        f = smootherstep(elapsed / duration) if duration > 0 else 1.0
        for a, p0, target in zip(arms, froms, targets):
            pos = p0.copy()
            pos[J0] = target if elapsed >= duration else p0[J0] + f * (target - p0[J0])
            a.write_cmd(pos)
        if now >= next_log or elapsed >= duration:
            states, ages = sample_lifts(arms)
            log_lifts(arms, states, ages, "ramp end" if elapsed >= duration else "ramping", elapsed)
            next_log = now + LIFT_LOG_INTERVAL_S
        if elapsed >= duration:
            return True
        time.sleep(1.0 / RATE_HZ)
    return False


def lift_to_shoulder(arms, speed=LIFT_SPEED, accel=LIFT_ACCEL):
    """Climb to shoulder level through the hard-coded J0_WAYPOINT_FRACS stops, settling at each one.
    A waypoint that cannot be verified is reported and does not stop the climb (no retries), so an
    incomplete arrival still ends with the shoulder-level command published and held."""
    if _stop:
        return False
    if not settle_lift(arms, timeout=LIFT_FEEDBACK_MAX_AGE, phase="start check"):
        if not _stop:
            print("[pickup] lift not started: J0 feedback is missing, invalid, stale, or differs from the held pose", flush=True)
        return False
    starts = [a.cmd[J0] for a in arms]
    stops = len(J0_WAYPOINT_FRACS)
    settled = True
    for i, frac in enumerate(J0_WAYPOINT_FRACS, 1):
        targets = [j0_waypoint(start, a.top, frac) for a, start in zip(arms, starts)]
        label = f"segment {i}/{stops} ({frac:.2f} of the climb)"
        if not lift_segment(arms, targets, speed, accel, label):
            return False
        # The last stop is shoulder level, so it keeps the plain "settling" phase name.
        if not settle_lift(arms, phase="settling" if i == stops else f"waypoint {i}/{stops}"):
            settled = False
        if _stop:
            return False
        if i < stops:
            hold(arms, J0_WAYPOINT_SETTLE_S)
    return settled


def settle_joint(arms, joint, timeout=ARRIVE_TIMEOUT_S, *, require_arrival=False):
    """Hold the command until ``joint`` arrives on every arm, or stops moving (a hard stop /
    obstacle), or ``timeout``. The daemon clips ctrl to +-0.5 turns of the live position, so a
    fast ramp can outrun the joint; this is where it catches up. With ``require_arrival``,
    a stall or timeout without reaching the command fails instead of allowing the next stage."""
    dt = 1.0 / RATE_HZ
    t0 = time.monotonic()
    last_pos = [a.live()[joint] for a in arms]
    last_move = [t0] * len(arms)
    while not _stop:
        now = time.monotonic()
        done = True
        all_arrived = True
        for i, a in enumerate(arms):
            a.write_cmd(a.cmd)
            p = a.live()[joint]
            if abs(p - last_pos[i]) > STALL_EPS:
                last_pos[i], last_move[i] = p, now
            arrived = abs(p - a.cmd[joint]) < ARRIVE_TOL
            stalled = now - last_move[i] > STALL_S
            if stalled and not arrived:
                print(f"[pickup] {a.side}: J{joint} stalled at {p:+.3f} (cmd {a.cmd[joint]:+.3f})", flush=True)
            done &= arrived or stalled
            all_arrived &= arrived
        if done or now - t0 > timeout:
            if require_arrival and not all_arrived:
                raise RuntimeError(f"[pickup] J{joint} did not reach its target on every arm")
            break
        time.sleep(dt)


def initialize_pose(arms, targets, lift_speed):
    stages = [(ELBOW, ELBOW_SPEED, ELBOW_SETTLE_S),
              (GRIPPER, GRIP_SPEED, GRIP_SETTLE_S), (J0, lift_speed, TOP_SETTLE_S)]
    stages.extend((joint, INITIALIZE_SPEED, 0.0) for joint in (SHOULDER, SWING, WRIST_ROLL, WRIST_YAW, WRIST_PITCH))
    print("[pickup] initialization: calibrated home, raised lift, open claws, 90 deg wrist roll", flush=True)
    for joint, speed, pause in stages:
        if _stop:
            return False
        for a, target in zip(arms, targets):
            print(f"[pickup] {a.side}: initialize J{joint} {a.cmd[joint]:+.3f} -> {target[joint]:+.3f}", flush=True)
        ramp_joint(arms, joint, [target[joint] for target in targets], speed,
                   ease=smoothstep if joint in (J0, GRIPPER) else smootherstep)
        if _stop:
            return False
        settle_joint(arms, joint, require_arrival=True)
        if pause:
            hold(arms, pause)
    hold(arms, TOP_SETTLE_S)
    if _stop:
        return False
    for a, target in zip(arms, targets):
        if not np.all(np.abs(a.live() - target) < ARRIVE_TOL):
            raise RuntimeError(f"[pickup] {a.side}: initialization pose was not reached; pickup cancelled")
    print("[pickup] initialization complete", flush=True)
    return True


def pinch_direction(arm):
    """Joint-space step (turns) that swings J2 toward the centreline."""
    d = np.zeros(arm.dof)
    d[SWING] = inward_sign(arm.cfg, arm.cmd, SWING)
    return d


def hook_direction(arm, keep_height=False, *, joint=WRIST_YAW):
    """Rotate the selected wrist joint inward, holding other joints and claw extension.
    Rolled claws use J6 instead of J5. Prepare while raised because hand height can change.
    The optional keep_height blend retains the unrolled J5/J6 level sweep."""
    if not keep_height:
        d = np.zeros(arm.dof)
        d[joint] = inward_sign(arm.cfg, arm.cmd, joint)
        return d
    q = np.asarray(arm.cmd, dtype=np.float64)
    p0, _ = arm.cfg.ik.fk(list(arm.cfg.q2urdf(q.copy())[:7]))
    cols = []
    for j in (WRIST_YAW, WRIST_PITCH):
        lo, hi = q.copy(), q.copy()
        lo[j] -= 0.01
        hi[j] += 0.01
        plo, _ = arm.cfg.ik.fk(list(arm.cfg.q2urdf(lo)[:7]))
        phi, _ = arm.cfg.ik.fk(list(arm.cfg.q2urdf(hi)[:7]))
        cols.append((np.asarray(phi) - np.asarray(plo)) / 0.02)
    jac = np.array(cols).T                                      # d(EE xyz) / d(J5, J6)
    ab = np.linalg.solve(jac[1:], [-np.sign(p0[1]) or -1.0, 0.0])  # y inward, z zero
    d = np.zeros(arm.dof)
    d[WRIST_YAW], d[WRIST_PITCH] = ab / np.abs(ab).max()
    return d


def creep_to_contact(arms, direction, speed, contact_err, squeeze, max_travel, label, current_a=None):
    """Creep every arm along its joint-space ``direction`` (largest component 1 turn) until it
    meets the box, then hold a fixed squeeze past the contact point. Contact is the tracking
    error along the direction rising above ``contact_err``, or -- with ``current_a`` -- the
    measured current on the moving joint reaching that many amps, which stops on force instead of
    a position proxy and adds no positional squeeze past it. Each arm detects it on its own, so an
    off-centre box is still held from both sides. Travel is capped by ``max_travel`` (a scalar or
    one value per arm) and by the calibrated range of every moving joint."""
    dt = 1.0 / RATE_HZ
    starts = [a.cmd.copy() for a in arms]
    dirs = [direction(a) for a in arms]
    caps = max_travel if np.ndim(max_travel) else [max_travel] * len(arms)
    limits = []
    for a, d, cap in zip(arms, dirs, caps):
        moving = np.nonzero(d)[0]
        room = [(a.edge(j, d[j]) - a.cmd[j]) / d[j] for j in moving]
        limits.append(max(min(min(room), cap), 0.0))
        joints = " ".join(f"J{j}{d[j]:+.2f}" for j in moving)
        print(f"[pickup] {a.side}: {label} along {joints} from {a.cmd[moving]}, max travel {limits[-1]:.3f}", flush=True)
    contact = [None] * len(arms)
    t0 = time.monotonic()
    while not _stop:
        travel = speed * (time.monotonic() - t0)
        for i, (a, p0, d) in enumerate(zip(arms, starts, dirs)):
            pos = a.cmd.copy()
            if contact[i] is None:
                pos = p0 + d * min(travel, limits[i])
                live = a.live()
                err = float(d @ (pos - live)) / float(d @ d)
                if current_a is not None:
                    state = a.latest_state()
                    joint = int(np.argmax(np.abs(d)))
                    if state is not None and abs(float(state["current"][joint])) > current_a:
                        contact[i] = live
                        print(f"[pickup] {a.side}: {label} contact at {abs(float(state['current'][joint])):.2f}A "
                              f"after {min(travel, limits[i]):.3f}, holding without extra squeeze", flush=True)
                        a.write_cmd(pos)
                        continue
                if err > contact_err:
                    contact[i] = live
                    pos = p0 + d * np.clip(min(travel, limits[i]) - err + squeeze, 0.0, limits[i])
                    print(f"[pickup] {a.side}: {label} contact after {min(travel, limits[i]) - err:.3f}, holding +{squeeze:.3f}", flush=True)
                elif travel >= limits[i]:
                    contact[i] = live
                    print(f"[pickup] {a.side}: {label} no contact within travel limit, holding at {limits[i]:.3f}", flush=True)
            a.write_cmd(pos)
        if all(c is not None for c in contact):
            break
        time.sleep(dt)


def regulate_pinch(arms, secs, target_a=PINCH_CURRENT_A, deadband=PINCH_CURRENT_DB):
    """Hold the grasp at a current, not a fixed position preload. While J2 presses harder than
    ``target_a`` the goal walks back toward the measured position (shedding current, so heat, the
    way the daemon's J7 relief loop does); below the band the backoff relaxes to re-grip. The
    backoff is capped by PINCH_RELIEF_MAX and never pushes past the original contact command."""
    dt = 1.0 / RATE_HZ
    holds = [a.cmd.copy() for a in arms]
    signs = [inward_sign(a.cfg, a.cmd, SWING) for a in arms]
    bias = [0.0] * len(arms)
    t0 = time.monotonic()
    while not _stop and (secs is None or time.monotonic() - t0 < secs):
        for i, (a, held, inward) in enumerate(zip(arms, holds, signs)):
            state = a.latest_state()
            if state is not None and np.isfinite(state["current"][SWING]):
                current = abs(float(state["current"][SWING]))
                if current > target_a + deadband:
                    bias[i] = min(bias[i] + PINCH_RELIEF_STEP, PINCH_RELIEF_MAX)
                elif current < target_a - deadband:
                    bias[i] = max(bias[i] - PINCH_RELIEF_STEP, 0.0)
            pos = held.copy()
            pos[SWING] = held[SWING] - inward * bias[i]
            a.write_cmd(pos)
        time.sleep(dt)
    for a, held, inward, backoff in zip(arms, holds, signs, bias):
        print(f"[pickup] {a.side}: pinch holding J2 {a.cmd[SWING]:+.3f} "
              f"(contact {held[SWING]:+.3f}, backoff {backoff:.4f} turns) at ~{target_a:.2f}A", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["left", "right", "both"], default="both")
    ap.add_argument("--speed", type=float, default=J0_SPEED, help="J0 descent speed, turns/s")
    ap.add_argument("--lift-speed", type=float, default=LIFT_SPEED,
                    help="maximum final-ascent command speed, turns/s (default: %(default)s)")
    ap.add_argument("--lift-accel", type=float, default=LIFT_ACCEL,
                    help="maximum final-ascent command acceleration, turns/s^2; requires load validation (default: %(default)s)")
    ap.add_argument("--hold", type=float, default=None,
                    help="seconds to hold the final pose before disabling torque (default: until Ctrl+C)")
    ap.add_argument("--bottom-margin", type=float, default=J0_BOTTOM_MARGIN,
                    help="J0 turns above the calibrated bottom; larger values stop higher (default: %(default)s)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--lower-only", action="store_true", help="hold the spread low pose without grasping")
    mode.add_argument("--pickup", action="store_true", help="add gripper closure and cradle after the inward grasp, before the final lift")
    ap.add_argument("--spread", type=float, default=None,
                    help="cap the outward J2 swing at this many turns from the start pose (default: full calibrated range)")
    ap.add_argument("--hook", type=float, default=HOOK_MAX_TRAVEL,
                    help="max inward J6 wrist rotation in turns before descent; roll, J5 and claws hold, 0 disables (default: %(default)s)")
    ap.add_argument("--squeeze", type=float, default=PINCH_SQUEEZE,
                    help="extra inward J2 turns past detected contact, capped by calibration (default: %(default)s)")
    ap.add_argument("--elbow-extension", type=float, default=ELBOW_EXTENSION,
                    help="J3 extension from the 90-degree home bend in turns before descent; 0 disables (default: %(default)s)")
    ap.add_argument("--shoulder-lift", type=float, default=SHOULDER_LIFT,
                    help="J1 raise in turns held with the extended reach so the elbow tip clears the chassis; 0 disables (default: %(default)s)")
    ap.add_argument("--pinch-current", type=float, default=PINCH_CURRENT_A,
                    help="J2 current in amps that stops the pinch and is then held; needs calibration at the grasp pose (default: %(default)s)")
    ap.add_argument("--pinch-margin", type=float, default=PINCH_CENTER_MARGIN,
                    help="turns of J2 travel backstop kept outboard of home (default: %(default)s)")
    ap.add_argument("--elbow-clearance", type=float, default=ELBOW_CLEARANCE,
                    help="extra right-arm J3 extension in turns so its forearm clears the lower chassis (default: %(default)s)")
    ap.add_argument("--cradle", type=float, default=CRADLE_TILT, help="extra elbow flex from the reach pose in turns after gripping; 0 disables")
    args = ap.parse_args()
    for name in ("speed", "bottom_margin", "lift_speed", "lift_accel"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            ap.error(f"--{name.replace('_', '-')} must be finite and greater than zero")
    if not np.isfinite(args.pinch_current) or args.pinch_current <= 0:
        ap.error("--pinch-current must be finite and greater than zero")
    for name in ("hook", "squeeze", "elbow_extension", "elbow_clearance", "pinch_margin", "shoulder_lift"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            ap.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.hold is not None and (not np.isfinite(args.hold) or args.hold < 0):
        ap.error("--hold must be finite and nonnegative")
    hold_description = "until Ctrl+C" if args.hold is None else f"for {args.hold:.1f}s"
    sides = ["left", "right"] if args.arm == "both" else [args.arm]

    arms = [Arm(s) for s in sides]
    try:
        initial_targets = [a.initial_pose() for a in arms]
        elbow_targets = [a.reach_elbow(elbow_extension_for(a.side, args.elbow_extension, args.elbow_clearance))
                         for a in arms]
        cradle_targets = ([a.cradle(args.cradle, start=target) for a, target in zip(arms, elbow_targets)]
                          if args.pickup and args.cradle > 0 else [])
        # Only an OFF->ON transition reseeds the daemon's command filter, so disable first
        # and flush ctrl to the live pose before energizing.
        for a in arms:
            a.set_torque(False)
            a.write_cmd(a.live())
            print(f"[pickup] {a.side}: J0 {a.cmd[J0]:+.3f}  top {a.top:+.3f}  bottom {a.bottom:+.3f}"
                  f"  elbow {a.cmd[ELBOW]:+.3f} -> {a.elbow_90:+.3f}"
                  f"  grip {a.cmd[GRIPPER]:+.3f} open {a.grip_open:+.3f} closed {a.grip_closed:+.3f}", flush=True)
        hold(arms, ENABLE_FLUSH_S)
        if _stop:
            return
        for a in arms:
            a.set_torque(True)
        hold(arms, ENABLE_SETTLE_S)

        # J3 carries the forearm, so it eases on a quintic (as in homing.py). It is held at
        # 90 deg during setup, then extended while raised before the lift descends.
        if not initialize_pose(arms, initial_targets, args.speed):
            return

        # Swing out to the far edge of the J2 range so the forearms clear the width of the box.
        # "Outward" is the opposite of the FK-derived inward sign; the daemon clips to the range anyway.
        spread_targets = []
        for a in arms:
            out = -inward_sign(a.cfg, a.cmd, SWING)
            target = a.edge(SWING, out)
            if args.spread is not None:
                target = a.cmd[SWING] + out * min(args.spread, abs(target - a.cmd[SWING]))
            spread_targets.append(target)
            print(f"[pickup] {a.side}: spread J2 {a.cmd[SWING]:+.3f} -> {target:+.3f}", flush=True)
        print("[pickup] spread", flush=True)
        ramp_joint(arms, SWING, spread_targets, SPREAD_SPEED, ease=smootherstep)
        settle_joint(arms, SWING)
        hold(arms, SPREAD_SETTLE_S)

        if not args.lower_only and args.hook > 0:
            print("[pickup] prepare grasp: rotate rolled wrists inward (J6), holding roll, J5 and claw extension", flush=True)
            creep_to_contact(arms, lambda a: hook_direction(a, joint=WRIST_PITCH),
                             HOOK_SPEED, HOOK_CONTACT_ERR, HOOK_SQUEEZE,
                             max_travel=args.hook, label="rolled wrist inward")
            settle_joint(arms, WRIST_PITCH)
            hold(arms, TOP_SETTLE_S)
        if _stop:
            return

        # Raise the shoulders before straightening the elbows. The extension alone only swings the
        # low elbow tip forward; lifting J1 is what gets it above the chassis it was catching on.
        if args.shoulder_lift > 0:
            print(f"[pickup] reach: raise shoulders (J1) {args.shoulder_lift * 360:.1f} deg while raised", flush=True)
            shoulder_targets = []
            for a in arms:
                up = up_sign(a.cfg, a.cmd, SHOULDER)
                target = float(np.clip(a.cmd[SHOULDER] + up * args.shoulder_lift,
                                       a.edge(SHOULDER, -1), a.edge(SHOULDER, 1)))
                shoulder_targets.append(target)
                print(f"[pickup] {a.side}: raise J1 {a.cmd[SHOULDER]:+.3f} -> {target:+.3f} "
                      f"(up is J1{up:+.0f})", flush=True)
            ramp_joint(arms, SHOULDER, shoulder_targets, SHOULDER_SPEED, ease=smootherstep)
            if _stop:
                return
            settle_joint(arms, SHOULDER, require_arrival=True)
            hold(arms, ELBOW_SETTLE_S)
        if _stop:
            return

        if args.elbow_extension > 0:
            print(f"[pickup] reach: extend elbows {args.elbow_extension * 360:.1f} deg while raised", flush=True)
            for a, target in zip(arms, elbow_targets):
                print(f"[pickup] {a.side}: extend J3 {a.cmd[ELBOW]:+.3f} -> {target:+.3f}", flush=True)
            ramp_joint(arms, ELBOW, elbow_targets, ELBOW_EXTENSION_SPEED, ease=smootherstep)
            if _stop:
                return
            settle_joint(arms, ELBOW, require_arrival=True)
            hold(arms, ELBOW_SETTLE_S)
        if _stop:
            return

        low_targets = [j0_low_target(a.top, a.bottom, args.bottom_margin) for a in arms]
        for a, target in zip(arms, low_targets):
            print(f"[pickup] {a.side}: low J0 target {target:+.3f} (bottom {a.bottom:+.3f}, margin {args.bottom_margin:.3f})", flush=True)
        print(f"[pickup] J0 -> low pose through {len(J0_WAYPOINT_FRACS)} stops", flush=True)
        descent_starts = [a.cmd[J0] for a in arms]
        for i, frac in enumerate(J0_WAYPOINT_FRACS, 1):
            stops = [j0_waypoint(start, target, frac)
                     for start, target in zip(descent_starts, low_targets)]
            print(f"[pickup] descent stop {i}/{len(J0_WAYPOINT_FRACS)} ({frac:.2f}): "
                  + "  ".join(f"{a.side} J0 {a.cmd[J0]:+.3f} -> {stop:+.3f}"
                              for a, stop in zip(arms, stops)), flush=True)
            ramp_joint(arms, J0, stops, args.speed)
            settle_joint(arms, J0)
            if _stop:
                return
            if i < len(J0_WAYPOINT_FRACS):
                hold(arms, J0_WAYPOINT_SETTLE_S)
        for a in arms:
            print(f"[pickup] {a.side}: J0 at {a.live()[J0]:+.3f}", flush=True)

        if args.lower_only:
            print(f"[pickup] holding low pose {hold_description}; torque drops on exit", flush=True)
            hold(arms, args.hold)
            return

        print("[pickup] grasp: swing elbows inward (J2), keeping the extended elbow reach pose (J3)", flush=True)
        # The forearm hits the chassis once J2 swings inward past its calibrated home, so the
        # pinch stops there: a box that never triggered contact detection would otherwise keep
        # squeezing into the robot and jam the lift.
        pinch_limits = []
        for a in arms:
            inward = inward_sign(a.cfg, a.cmd, SWING)
            pinch_limits.append(max(inward * (float(a.cfg.home[SWING]) - a.cmd[SWING])
                                    - args.pinch_margin, 0.0))
            print(f"[pickup] {a.side}: pinch stops {args.pinch_margin:.3f} turns outboard of J2 "
                  f"{a.cfg.home[SWING]:+.3f} (home), {pinch_limits[-1]:.3f} turns inward", flush=True)
        creep_to_contact(arms, pinch_direction, PINCH_SPEED, PINCH_CONTACT_ERR, args.squeeze,
                         max_travel=pinch_limits, label="pinch", current_a=args.pinch_current)
        regulate_pinch(arms, PINCH_SETTLE_S, args.pinch_current)

        if _stop:
            return

        if args.pickup:
            print("[pickup] cage: grippers -> closed", flush=True)
            ramp_joint(arms, GRIPPER, [a.grip_closed for a in arms], GRIP_SPEED)
            hold(arms, GRIP_SETTLE_S)
            if _stop:
                return
            if cradle_targets:
                print("[pickup] cage: cradle tilt from extended reach pose", flush=True)
                ramp_joint(arms, ELBOW, cradle_targets, CRADLE_SPEED, ease=smootherstep)
                hold(arms, CRADLE_SETTLE_S)
        if _stop:
            return

        print("[pickup] final stage: J0 -> shoulder level, maintaining grasp", flush=True)
        arrived = lift_to_shoulder(arms, args.lift_speed, args.lift_accel)
        if _stop:
            return
        states, ages = sample_lifts(arms)
        arrived = arrived and lifts_arrived(arms, states, ages)
        for a, state, age in zip(arms, states, ages):
            if not valid_lift_sample(state, age):
                print(f"[pickup] {a.side}: final pose feedback unavailable or invalid/stale", flush=True)
                continue
            live = state["pos"]
            print(f"[pickup] {a.side}: J0 at {live[J0]:+.3f}  J2 {live[SWING]:+.3f} (cmd {a.cmd[SWING]:+.3f})"
                  f"  J4 {live[WRIST_ROLL]:+.3f} (cmd {a.cmd[WRIST_ROLL]:+.3f})"
                  f"  J6 {live[WRIST_PITCH]:+.3f} (cmd {a.cmd[WRIST_PITCH]:+.3f})"
                  f"  elbow {live[ELBOW]:+.3f}  grip {live[GRIPPER]:+.3f}", flush=True)
        if arrived:
            print("[pickup] shoulder level reached on every arm", flush=True)
        else:
            print("[pickup] lift incomplete: shoulder-level arrival not verified; no application retry", flush=True)
        target_description = "shoulder-level target" if arrived else "existing J0 targets"
        print(f"[pickup] holding {target_description} and grasp {hold_description}; torque drops on exit and releases the box", flush=True)
        hold(arms, args.hold)
    finally:
        # Leave the arms limp on exit.
        for a in arms:
            a.set_torque(False)
        for a in arms:
            a.close()


if __name__ == "__main__":
    main()
