# /// script
# dependencies = [
#   "bbos",
#   "numpy",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Long-lived arm and base control for the interface-v2 action provider.

The motion idioms here (the OFF->ON torque edge, easing ramps, settle-on-arrival,
creep-to-contact by tracking error, FK-derived per-arm signs) follow actions/pickup.py,
which stays the standalone prototype and is not imported: it installs a SIGINT handler at
import time, which would steal the host process's handler.

Two things differ from pickup.py, and both exist because a provider outlives one motion:

  - Writers are opened once for the process and never closed. The arm daemon cuts torque
    the moment its control writer disappears, so closing writers drops a held box. Rig.close
    is therefore shutdown-only and refuses to run while a load may be held.
  - A publisher thread per subsystem republishes the current command forever. Motion routines
    only move the target; the publisher keeps the arms commanded at 200 Hz and the base twist
    refreshed at 50 Hz (a twist expires in 100 ms). Cancelling stops the routine and leaves the
    publisher holding position, which is what makes a load-preserving stop possible.

Cancellation is a threading.Event passed in by the caller, never a module global.
"""
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

from bbos import Reader, Writer, Type, Config

J0 = 0
SWING = 2               # shoulder swing: moves the EE along base y (toward / away from the other arm)
ELBOW = 3               # q2urdf(cfg.home)[3] is +1.571 rad on both arms: home[3] IS the 90 deg elbow
WRIST_YAW = 5           # turns the hand about base z
WRIST_PITCH = 6         # tips the hand about base y
GRIPPER = 7

ARM_RATE_HZ = 200.0     # arm command republish rate
DRIVE_RATE_HZ = 50.0    # base.command_hz; a twist expires after drive_command_timeout_s = 0.1
MOTION_RATE_HZ = 200.0  # rate at which motion routines advance their targets

ENABLE_FLUSH_S = 0.1    # ctrl flushed to the live pose before torque comes on
ENABLE_SETTLE_S = 0.3   # daemon mode-switch stall after torque on
ARRIVE_TOL = 0.03       # turns; joint counts as arrived within this of its command
ARRIVE_TIMEOUT_S = 6.0  # give up waiting for a joint to arrive after this
STALL_EPS = 0.003       # turns; less motion than this for STALL_S = the joint is stuck/at a stop
STALL_S = 0.75
RANGE_MARGIN = 0.02     # turns; stay this far inside every calibrated joint range
GRIP_OPEN_FRAC = 0.6    # how far toward the calibrated open stop the jaws open

# Driving. Limits borrowed from bbapps/nav, which is the autonomy reference on this robot;
# they sit well under the daemon's clamps (v_max 0.3 m/s, w_max 0.9 rad/s) because the clamp
# is a hardware limit, not a target.
# MEASURED on bracketbot-184, 2026-09-13, against the raw IMU heading:
#   drive.ctrl twist[1] = +0.30 rad/s  ->  imu.orientation yaw  +33.3 deg  (counter-clockwise)
#   drive.ctrl twist[1] = -0.30 rad/s  ->  imu.orientation yaw  -29.7 deg  (clockwise)
# So positive really is CCW, as motion_and_arms.md:44 documents, and no correction is needed.
#
# An earlier survey concluded the opposite because it measured the turn with perception's
# base_yaw, which carries the OPPOSITE sign convention to the raw IMU. That is self-consistent
# inside perception -- pose and box positions share it, so world_to_base stays correct -- but it
# is not the convention drive.ctrl speaks, and it is not a sign to calibrate a command against.
# Trust the IMU for what the base physically did.
YAW_COMMAND_SIGN = 1.0

YAW_TOL = 0.03          # rad; a turn is finished within this of its target
IMU_STALE_S = 0.5       # hold the last heading this long; past it the stream is dead
YAW_SETTLE_S = 0.3      # let the base stop coasting before the angle is believed
DRIVE_SPEED = 0.08      # m/s creeping toward a target
DRIVE_OMEGA = 0.15      # rad/s turning to face one
ALIGN_TOL = 0.05        # rad; inside this the base is considered pointed at the target
RANGE_TOL = 0.02        # m; inside this the standoff is reached
ARRIVED_MARGIN = 0.15   # m past the standoff within which losing sight counts as arriving
STUCK_S = 15.0          # no range progress for this long = give up rather than grind
STUCK_EPS = 0.01        # m of range change that counts as progress
DRIVE_TIMEOUT_S = 120.0


NUL = bytes([0])          # /proc/<pid>/cmdline separates arguments with it
CONTROL_TOPICS = ("drive.ctrl", "arm_left.ctrl", "arm_right.ctrl",
                  "arm_left.torque", "arm_right.torque")


class Cancelled(Exception):
    """Raised inside a motion routine when its cancel event is set."""


class HardwareBusy(RuntimeError):
    """Another live process already owns a control topic.

    Carries who, so the answer is to go and talk to them. Never kill the owner: its teardown
    drops whatever its arms are holding, and a daemon restart is worse.
    """


def topic_owners(topics=CONTROL_TOPICS):
    """Which live processes have each control topic mapped. Read-only; opens nothing.

    Walks /proc for the mapping rather than asking bbos, because bbos only reports a conflict
    once you have already tried to take the topic -- by which point a long startup has been paid
    for, and the answer arrives as an opaque pid. Returns {} off Linux so tests can run anywhere.
    """
    owners = {}
    proc = Path("/proc")
    if not proc.is_dir():
        return owners
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            maps = (entry / "maps").read_text()
        except OSError:
            continue            # the process exited, or is not ours to inspect
        for topic in topics:
            if f"/dev/shm/{topic}" in maps:
                try:
                    cmd = (entry / "cmdline").read_bytes().replace(NUL, b" ").decode().strip()
                except OSError:
                    cmd = "?"
                owners.setdefault(topic, []).append((int(entry.name), cmd or "?"))
    return owners


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


def inward_sign(cfg, turns, joint):
    """Sign of a step on ``joint`` that moves this arm's EE toward the robot centreline (base y -> 0)."""
    q = np.asarray(turns, dtype=np.float64).copy()
    p0, _ = cfg.ik.fk(list(cfg.q2urdf(q.copy())[:7]))
    q[joint] += 0.05
    p1, _ = cfg.ik.fk(list(cfg.q2urdf(q.copy())[:7]))
    return -float(np.sign((p1[1] - p0[1]) * p0[1])) or 1.0


class Arm:
    """One arm's readers, writers and calibrated limits, held open for the process lifetime."""

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
        ends = np.array([self.cal_min[GRIPPER], self.cal_max[GRIPPER]])
        self.grip_closed = float(ends[np.argmin(np.abs(ends))])
        self.grip_open = self.grip_closed + GRIP_OPEN_FRAC * (float(ends[np.argmax(np.abs(ends))]) - self.grip_closed)
        self.cmd = None
        self.torque_on = False

    def edge(self, joint, sign, margin=RANGE_MARGIN):
        """The calibrated end of ``joint``'s range in the ``sign`` direction, ``margin`` inside it."""
        return self.hi[joint] - margin if sign > 0 else self.lo[joint] + margin

    def cradle(self, tilt):
        """Elbow target ``tilt`` turns past 90 deg: 'more flex' is the sign of home[ELBOW] on this arm."""
        return self.elbow_90 + float(np.sign(self.elbow_90) or 1.0) * tilt

    def j0_meters_to_turns(self, metres):
        """Signed J0 travel in turns for a height change in metres, using this arm's lift radius."""
        return float(metres) / float(self.cfg.wheel_radius)

    def live(self, cancel=None, timeout=2.0):
        t0 = time.monotonic()
        while not self.r_state.ready():
            if cancel is not None and cancel.is_set():
                raise Cancelled("cancelled waiting for arm state")
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"arm_{self.side}.state went stale")
            time.sleep(0.005)
        return np.array(self.r_state.data["pos"], dtype=np.float64)

    def ee_height(self):
        """Base-frame z of this arm's end effector at its current command, via FK."""
        q = np.asarray(self.cmd, dtype=np.float64)
        pos, _ = self.cfg.ik.fk(list(self.cfg.q2urdf(q.copy())[:7]))
        return float(pos[2])

    def set_target(self, pos):
        """Move the commanded pose. The publisher thread is what actually writes it."""
        self.cmd = np.asarray(pos, dtype=np.float64)

    def publish(self):
        if self.cmd is None:
            return
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
        self.torque_on = bool(on)


class Rig:
    """Owns both arms and drive.ctrl, and republishes their commands for the process lifetime.

    This is the single designated hardware writer. It must not run beside mc_skills or
    actions/pickup.py, which open the same topics.
    """

    def __init__(self, sides=("left", "right"), log=print):
        self.log = log
        self.require_free(sides)
        self.arms = [Arm(s) for s in sides]
        self.by_side = {a.side: a for a in self.arms}
        self.w_drive = Writer("drive.ctrl", Type("drive_ctrl"), keeptime=False)
        self._yaw = None
        try:
            self._r_imu = Reader("imu.orientation", keeptime=False)
        except Exception:       # no IMU: turns fall back to open-loop timing
            self._r_imu = None
        self.cfg_drive = Config("drive")
        self._twist = np.zeros(2, dtype=np.float64)
        self._twist_lock = threading.Lock()
        self._running = threading.Event()
        self._threads = []

    @staticmethod
    def require_free(sides=("left", "right")):
        """Refuse to become the motion owner while somebody else already is.

        Checked before a single writer is opened, so a conflict costs a second rather than a
        detector warmup, and says who holds what instead of naming a bare pid.
        """
        wanted = ["drive.ctrl"] + [f"arm_{s}.{k}" for s in sides for k in ("ctrl", "torque")]
        held = topic_owners(tuple(wanted))
        if not held:
            return
        lines = [f"  {topic}: pid {pid} ({cmd})"
                 for topic, entries in sorted(held.items()) for pid, cmd in entries]
        advice = (
            "Only one process may own these at a time. Coordinate with whoever is running it "
            "and wait for them to finish. Do NOT kill the owner or restart a bbos daemon: its "
            "teardown drops whatever its arms are holding.")
        raise HardwareBusy("another process already owns the robot's control topics:"
                           + "".join(chr(10) + line for line in lines)
                           + chr(10) * 2 + advice)

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        """Seed every arm command from its live pose, bring torque on, and start publishing.

        Only an OFF->ON torque transition reseeds the daemon's command filter to the live pose;
        without it the arm snaps from wherever it hangs to whatever stale command is buffered.
        """
        for a in self.arms:
            a.set_torque(False)
            a.set_target(a.live())
            a.publish()
        self._running.set()
        self._threads = [
            threading.Thread(target=self._publish_arms, name="armctl-arms", daemon=True),
            threading.Thread(target=self._publish_drive, name="armctl-drive", daemon=True),
        ]
        for t in self._threads:
            t.start()
        time.sleep(ENABLE_FLUSH_S)
        for a in self.arms:
            a.set_torque(True)
        time.sleep(ENABLE_SETTLE_S)
        self.log("[armctl] rig started: " + ", ".join(a.side for a in self.arms))

    def close(self, force=False):
        """Shutdown only. Closing arm writers drops whatever the arms are holding."""
        if not force:
            raise RuntimeError("Rig.close drops a held load; pass force=True at process shutdown")
        self.set_twist(0.0, 0.0)
        self._running.clear()
        for t in self._threads:
            t.join(timeout=1.0)
        for a in self.arms:
            a.set_torque(False)
        for a in self.arms:
            a.w_ctrl.__exit__(None, None, None)
            a.w_torque.__exit__(None, None, None)
        self.w_drive.__exit__(None, None, None)

    # -- publishers --------------------------------------------------------

    def _publish_arms(self):
        dt = 1.0 / ARM_RATE_HZ
        while self._running.is_set():
            for a in self.arms:
                a.publish()
            time.sleep(dt)

    def _publish_drive(self):
        """Republish the twist at 50 Hz. A twist is a velocity that expires, not a destination."""
        dt = 1.0 / DRIVE_RATE_HZ
        while self._running.is_set():
            with self._twist_lock:
                twist = self._twist.copy()
            with self.w_drive.buf() as b:
                b["twist"][:] = twist.astype(np.float32)
            time.sleep(dt)
        with self.w_drive.buf() as b:   # explicit zero on the way out; the timeout is the backstop
            b["twist"][:] = np.zeros(2, dtype=np.float32)

    # -- base --------------------------------------------------------------

    def measured_yaw(self):
        """Body yaw in RADIANS from the IMU, or None when it is unavailable.

        imu.orientation.rpy is published in DEGREES, and unbounded rather than wrapped. Reading it
        as radians makes every increment ~57x too large, which is how a survey once accumulated
        tens of thousands of degrees from a quarter turn. Measured on bracketbot-184: a stationary
        robot's yaw moved 2.5 degrees over 4 seconds, so this is precise enough to close a loop on
        a quarter turn and not precise enough to trust over a long one.

        Read-only: readers are unlimited, so this competes with nothing.
        """
        if self._r_imu is None:
            return None
        try:
            if self._r_imu.ready():
                self._yaw = (math.radians(float(np.asarray(self._r_imu.data["rpy"],
                                                           dtype=np.float64)[2])), time.monotonic())
        except Exception:
            pass
        if self._yaw is None:
            return None
        # ready() is edge-triggered: it is True once per new sample, so a loop polling faster than
        # the IMU publishes sees False most ticks. Hold the last reading rather than reporting the
        # heading as unknown between samples, but stop trusting it if the stream actually dies.
        value, seen = self._yaw
        return value if time.monotonic() - seen <= IMU_STALE_S else None

    def turn_by(self, radians, rate, cancel=None, log=None):
        """Rotate by a measured angle rather than for a computed duration.

        Open-loop timing under-rotated badly in the first live survey -- eight 45 degree steps
        covered 137 degrees, not 360 -- because a commanded rate is not an achieved rate: the base
        spends much of a short step accelerating. Closing the loop on the IMU removes the guess.
        Yaw is accumulated from wrapped increments, so it stays correct across the +-pi seam.
        """
        log = log or self.log
        start = self.measured_yaw()
        if start is None:                      # no IMU: fall back to timing, and say so
            log("[armctl] no IMU yaw; turning open-loop, angle is approximate")
            self.set_twist(0.0, math.copysign(rate, radians))
            dwell(abs(radians) / rate, cancel)
            self.stop_base()
            return None
        turned, last = 0.0, start
        deadline = time.monotonic() + abs(radians) / rate * 4.0 + 10.0
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    raise Cancelled("cancelled mid-turn")
                now_yaw = self.measured_yaw()
                if now_yaw is not None:
                    step = (now_yaw - last + math.pi) % (2 * math.pi) - math.pi
                    turned += step
                    last = now_yaw
                remaining = radians - turned
                if abs(remaining) <= YAW_TOL or time.monotonic() > deadline:
                    break
                # ease down over the last part so the base does not overshoot and hunt
                self.set_twist(0.0, math.copysign(min(rate, max(0.08, abs(remaining))), remaining))
                _tick(cancel)
        finally:
            self.stop_base()
        dwell(YAW_SETTLE_S, cancel)
        settled = self.measured_yaw()
        if settled is not None:
            turned += (settled - last + math.pi) % (2 * math.pi) - math.pi
        return turned

    def set_twist(self, v, w):
        """Set the commanded body twist. ``w`` is positive-CCW; the hardware's inversion is
        applied here so it is corrected in exactly one place."""
        v_max = float(self.cfg_drive.max_linear_vel)
        w_max = float(self.cfg_drive.max_angular_vel)
        v = 0.0 if not np.isfinite(v) else float(np.clip(v, -v_max, v_max))
        w = 0.0 if not np.isfinite(w) else float(np.clip(w, -w_max, w_max))
        with self._twist_lock:
            self._twist[:] = (v, YAW_COMMAND_SIGN * w)

    def stop_base(self):
        self.set_twist(0.0, 0.0)

    def carry_height(self):
        """Base-frame z of the cradle: where a held box's centre sits right now.

        Measured from the arms' own FK rather than assumed from J0, so it stays correct
        whatever the elbows are doing. Both forearms carry the box, so they agree.
        """
        heights = [a.ee_height() for a in self.arms if a.cmd is not None]
        if not heights:
            return float("nan")
        return sum(heights) / len(heights)

    def carry_centre(self):
        """Base-frame xyz midway between the two forearms: where a held box sits.

        Derived from the arms' own FK at their current command, so it tracks the spread and
        cradle rather than assuming a fixed pose. With one arm there is no sandwich and no
        meaningful centre, so this reports None.
        """
        points = []
        for arm in self.arms:
            if arm.cmd is None:
                return None
            pos, _ = arm.cfg.ik.fk(list(arm.cfg.q2urdf(np.asarray(arm.cmd, dtype=np.float64).copy())[:7]))
            points.append(np.asarray(pos, dtype=np.float64))
        if len(points) < 2:
            return None
        return tuple(np.mean(points, axis=0).tolist())

    def hold(self, cancel=None):
        """Freeze the arms at their current command and stop the base. Torque stays on."""
        self.stop_base()
        for a in self.arms:
            if a.cmd is not None:
                a.set_target(a.cmd)


class Stuck(RuntimeError):
    """The base stopped making progress toward its target."""


def drive_to_standoff(rig, target_fn, standoff, cancel=None, log=print,
                      speed=DRIVE_SPEED, omega=DRIVE_OMEGA, timeout=DRIVE_TIMEOUT_S):
    """Turn to face a target, then creep toward it until ``standoff`` metres remain.

    ``target_fn`` returns the target in the CURRENT base frame and is called every cycle, so
    the robot steers to where the target is now rather than to an integrated pose. There is no
    SLAM here: dead reckoning drifts, and a pose epoch roll is fatal to the job, so a remembered
    goal is worse than useless. If target_fn cannot produce a fresh target it raises, and that
    ends the drive rather than letting it continue blind.

    The base is zeroed on every exit path. The daemon's 100 ms command timeout is a backstop,
    not the brake.
    """
    t0 = time.monotonic()
    best_range, last_progress = float("inf"), t0
    last_distance = float("inf")
    try:
        while True:
            if cancel is not None and cancel.is_set():
                raise Cancelled("cancelled while driving")
            try:
                target = target_fn()
            except Exception:
                # Losing the target is fatal at range and expected on arrival: a floor-level box
                # drops out of a forward-looking camera once the robot is nearly on top of it.
                # Only the second reading counts as arriving, and only because the distance
                # measured a moment ago says the robot is already there.
                if last_distance <= standoff + ARRIVED_MARGIN:
                    log(f"[armctl] target left view at {last_distance:.2f}m, inside the standoff; "
                        f"treating as arrived")
                    return last_distance
                raise
            bearing = math.atan2(target[1], target[0])
            distance = math.hypot(target[0], target[1])
            last_distance = distance

            if distance < best_range - STUCK_EPS:
                best_range, last_progress = distance, time.monotonic()
            elif time.monotonic() - last_progress > STUCK_S:
                raise Stuck(f"no progress for {STUCK_S:.0f}s at {distance:.2f}m")
            if time.monotonic() - t0 > timeout:
                raise Stuck(f"drive exceeded {timeout:.0f}s")

            if abs(bearing) > ALIGN_TOL:
                # Turn in place first. Creeping while badly misaligned arcs the robot around
                # the target instead of closing on it.
                rig.set_twist(0.0, math.copysign(min(omega, abs(bearing)), bearing))
            elif distance > standoff + RANGE_TOL:
                rig.set_twist(min(speed, distance - standoff), 0.0)
            else:
                log(f"[armctl] standoff reached at {distance:.3f}m (target {standoff:.3f}m)")
                return distance
            _tick(cancel)
    finally:
        rig.stop_base()


# -- motion primitives -----------------------------------------------------
#
# Each advances the arms' targets; the rig's publisher is what reaches hardware.
# All of them raise Cancelled promptly so the caller can leave the load held.


def _tick(cancel, dt=1.0 / MOTION_RATE_HZ):
    if cancel is not None and cancel.is_set():
        raise Cancelled("cancelled mid-motion")
    time.sleep(dt)


def dwell(secs, cancel=None):
    """Hold the current targets for ``secs``, staying responsive to cancellation."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        _tick(cancel)


def ramp_joint(arms, joint, targets, speed, ease=smoothstep, cancel=None):
    """Ease ``joint`` of every arm to its target; all other joints hold."""
    froms = [a.cmd.copy() for a in arms]
    dist = max(abs(t - p0[joint]) for p0, t in zip(froms, targets))
    duration = max(dist / speed, 1e-3)
    t0 = time.monotonic()
    while True:
        t = time.monotonic() - t0
        f = ease(t / duration)
        for a, p0, target in zip(arms, froms, targets):
            pos = p0.copy()
            pos[joint] = p0[joint] + f * (target - p0[joint])
            a.set_target(pos)
        if t >= duration:
            return
        _tick(cancel)


def settle_joint(arms, joint, timeout=ARRIVE_TIMEOUT_S, cancel=None, log=print):
    """Hold until ``joint`` arrives on every arm, or stops moving (a hard stop / obstacle),
    or ``timeout``. The daemon clips ctrl to +-0.5 turns of the live position, so a fast ramp
    can outrun the joint; this is where it catches up."""
    t0 = time.monotonic()
    last_pos = [a.live(cancel)[joint] for a in arms]
    last_move = [t0] * len(arms)
    while True:
        now = time.monotonic()
        done = True
        for i, a in enumerate(arms):
            p = a.live(cancel)[joint]
            if abs(p - last_pos[i]) > STALL_EPS:
                last_pos[i], last_move[i] = p, now
            arrived = abs(p - a.cmd[joint]) < ARRIVE_TOL
            stalled = now - last_move[i] > STALL_S
            if stalled and not arrived:
                log(f"[armctl] {a.side}: J{joint} stalled at {p:+.3f} (cmd {a.cmd[joint]:+.3f})")
            done &= arrived or stalled
        if done or now - t0 > timeout:
            return
        _tick(cancel)


def pinch_direction(arm):
    """Joint-space step (turns) that swings J2 toward the centreline."""
    d = np.zeros(arm.dof)
    d[SWING] = inward_sign(arm.cfg, arm.cmd, SWING)
    return d


def spread_direction(arm):
    """Joint-space step that swings J2 away from the centreline."""
    return -pinch_direction(arm)


def hook_direction(arm):
    """Joint-space step that rotates J5 inward toward the box, pitch and claw untouched.

    Yaw alone can change hand height, which is why this is prepared while raised rather
    than swept across the box at floor level.
    """
    d = np.zeros(arm.dof)
    d[WRIST_YAW] = inward_sign(arm.cfg, arm.cmd, WRIST_YAW)
    return d


def j0_low_target(arm, margin):
    """J0's calibrated bottom held ``margin`` turns back toward the top.

    That margin is a lift offset, not a measured floor clearance: nothing here senses the floor.
    """
    top, bottom = arm.top, arm.bottom
    return float(bottom + np.sign(top - bottom) * min(margin, abs(top - bottom)))


def creep_to_contact(arms, direction, speed, contact_err, squeeze, max_travel, label,
                     cancel=None, log=print):
    """Creep every arm along its joint-space ``direction`` until it meets the box, then hold a
    fixed squeeze past the contact point. Contact is the tracking error along the direction
    rising above ``contact_err``; each arm detects it on its own, so an off-centre box is still
    held from both sides. Travel is capped by ``max_travel`` and by every moving joint's range.

    A tracking-error rise means something resisted the joint. It is not possession evidence.
    """
    starts = [a.cmd.copy() for a in arms]
    dirs = [direction(a) for a in arms]
    limits = []
    for a, d in zip(arms, dirs):
        moving = np.nonzero(d)[0]
        room = [(a.edge(j, d[j]) - a.cmd[j]) / d[j] for j in moving]
        limits.append(max(min(min(room), max_travel), 0.0))
        joints = " ".join(f"J{j}{d[j]:+.2f}" for j in moving)
        log(f"[armctl] {a.side}: {label} along {joints}, max travel {limits[-1]:.3f}")
    contact = [None] * len(arms)
    t0 = time.monotonic()
    while True:
        travel = speed * (time.monotonic() - t0)
        for i, (a, p0, d) in enumerate(zip(arms, starts, dirs)):
            if contact[i] is not None:
                continue
            pos = p0 + d * min(travel, limits[i])
            live = a.live(cancel)
            err = float(d @ (pos - live)) / float(d @ d)
            if err > contact_err:
                contact[i] = live
                pos = p0 + d * np.clip(min(travel, limits[i]) - err + squeeze, 0.0, limits[i])
                log(f"[armctl] {a.side}: {label} contact after {min(travel, limits[i]) - err:.3f}, holding +{squeeze:.3f}")
            elif travel >= limits[i]:
                contact[i] = live
                log(f"[armctl] {a.side}: {label} no contact within travel limit, holding at {limits[i]:.3f}")
            a.set_target(pos)
        if all(c is not None for c in contact):
            return
        _tick(cancel)
