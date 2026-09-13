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
     wait for both arms to arrive, then lift (J0) -> bottom minus --bottom-margin turns
     toward the top; --lower-only holds this extended reach pose
  5. cage the box, keeping the prepared wrist angles:
       a. pinch: J2 brings the elbows and forearms inward until each forearm meets the box side
          (tracking error rises), then holds --squeeze turns past contact through the lift
       b. with --pickup, grip: grippers close (on a rim/corner if there is one; otherwise they just stiffen the
          hand into a solid paddle -- the daemon's J7 current-relief loop keeps the grip gentle)
       c. with --pickup, cradle: the elbows flex a little from the reach pose, lifting the front edge of the box so it
          tilts back against the upper arms and the weight rests on the forearms
  6. lift J0 back up to the top (shoulder level), preserving the grasp, and hold until termination
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
ELBOW_EXTENSION = 30.0 / 360.0
ELBOW_EXTENSION_SPEED = 0.05
INITIALIZE_SPEED = 0.08
LIFT_SPEED = 0.4        # turns/s for the loaded ascent
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
PINCH_CONTACT_ERR = 0.015   # turns of J2 tracking error that counts as touching the box (no-load ~0.002)
PINCH_SQUEEZE = 0.03    # turns commanded past the contact point: a steady spring squeeze. TUNE on box
PINCH_SETTLE_S = 0.5    # let the squeeze load up before hooking
HOOK_SPEED = 0.08       # turns/s wrist creep, hand turning in across the front of the box
HOOK_MAX_TRAVEL = 0.22  # turns (~80 deg) of wrist travel when nothing stops the hand earlier
HOOK_CONTACT_ERR = 0.015    # turns of wrist tracking error that counts as the hand touching the box
HOOK_SQUEEZE = 0.01     # turns past the hook contact point. TUNE on box
GRIP_SPEED = 0.4        # turns/s closing / opening the gripper (~0.33 turns in <1 s)
GRIP_OPEN_FRAC = 0.6    # how far toward the calibrated open stop the jaws open (~quest_teleop's wide-open)
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
        self.r_state = Reader(f"arm_{side}.state")
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
        self.grip_open = self.grip_closed + GRIP_OPEN_FRAC * (float(ends[np.argmax(np.abs(ends))]) - self.grip_closed)
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
    stages.extend((joint, INITIALIZE_SPEED, 0.0) for joint in (1, SWING, WRIST_ROLL, WRIST_YAW, WRIST_PITCH))
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


def creep_to_contact(arms, direction, speed, contact_err, squeeze, max_travel, label):
    """Creep every arm along its joint-space ``direction`` (largest component 1 turn) until it
    meets the box, then hold a fixed squeeze past the contact point. Contact is the tracking
    error along the direction rising above ``contact_err``; each arm detects it on its own, so an
    off-centre box is still held from both sides. Travel is capped by ``max_travel`` and by the
    calibrated range of every moving joint."""
    dt = 1.0 / RATE_HZ
    starts = [a.cmd.copy() for a in arms]
    dirs = [direction(a) for a in arms]
    limits = []
    for a, d in zip(arms, dirs):
        moving = np.nonzero(d)[0]
        room = [(a.edge(j, d[j]) - a.cmd[j]) / d[j] for j in moving]
        limits.append(max(min(min(room), max_travel), 0.0))
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["left", "right", "both"], default="both")
    ap.add_argument("--speed", type=float, default=J0_SPEED, help="J0 descent speed, turns/s")
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
    ap.add_argument("--cradle", type=float, default=CRADLE_TILT, help="extra elbow flex from the reach pose in turns after gripping; 0 disables")
    args = ap.parse_args()
    for name in ("speed", "bottom_margin"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            ap.error(f"--{name.replace('_', '-')} must be finite and greater than zero")
    for name in ("hook", "squeeze", "elbow_extension"):
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
        elbow_targets = [a.reach_elbow(args.elbow_extension) for a in arms]
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
        print("[pickup] J0 -> low pose", flush=True)
        ramp_joint(arms, J0, low_targets, args.speed)
        settle_joint(arms, J0)
        for a in arms:
            print(f"[pickup] {a.side}: J0 at {a.live()[J0]:+.3f}", flush=True)

        if args.lower_only:
            print(f"[pickup] holding low pose {hold_description}; torque drops on exit", flush=True)
            hold(arms, args.hold)
            return

        print("[pickup] grasp: swing elbows inward (J2), keeping the extended elbow reach pose (J3)", flush=True)
        creep_to_contact(arms, pinch_direction, PINCH_SPEED, PINCH_CONTACT_ERR, args.squeeze,
                         max_travel=np.inf, label="pinch")
        hold(arms, PINCH_SETTLE_S)

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
        ramp_joint(arms, J0, [a.top for a in arms], LIFT_SPEED)
        if _stop:
            return
        settle_joint(arms, J0)
        if _stop:
            return
        for a in arms:
            live = a.live()
            print(f"[pickup] {a.side}: J0 at {live[J0]:+.3f}  J2 {live[SWING]:+.3f} (cmd {a.cmd[SWING]:+.3f})"
                  f"  J4 {live[WRIST_ROLL]:+.3f} (cmd {a.cmd[WRIST_ROLL]:+.3f})"
                  f"  J6 {live[WRIST_PITCH]:+.3f} (cmd {a.cmd[WRIST_PITCH]:+.3f})"
                  f"  elbow {live[ELBOW]:+.3f}  grip {live[GRIPPER]:+.3f}", flush=True)
        print(f"[pickup] holding shoulder-level target and grasp {hold_description}; torque drops on exit and releases the box", flush=True)
        hold(arms, args.hold)
    finally:
        # Leave the arms limp on exit.
        for a in arms:
            a.set_torque(False)
        for a in arms:
            a.close()


if __name__ == "__main__":
    main()
