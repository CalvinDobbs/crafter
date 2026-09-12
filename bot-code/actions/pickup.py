# /// script
# dependencies = [
#   "bbos",
#   "numpy",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Drive the arms' vertical stage (J0) from the top-most to the bottom-most point, once.

Joint-space, no IK: the elbow (J3) is first bent to 90 deg and held there for the rest of
the run so the forearm clears the table; J0 is the lift; every other joint holds its live
pose. Top/bottom come from the per-robot ranges.calibration.json (motor turns, the
arm_ctrl.pos frame), and the "down" sign is per-arm (the arms are mirror images in
motor-turn space).

Usage:  uv run pickup.py [--arm left|right|both] [--speed TURNS_PER_S]
"""
import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np

from bbos import Reader, Writer, Type, Config

J0 = 0
ELBOW = 3               # q2urdf(cfg.home)[3] is +1.571 rad on both arms: home[3] IS the 90 deg elbow
RATE_HZ = 200.0
J0_SPEED = 0.4          # turns/s along the lift (matches homing.J0_PARK_DOWN_SPEED)
ELBOW_SPEED = 0.15      # turns/s bending the elbow (~0.25 turns in ~1.7 s)
ELBOW_SETTLE_S = 0.5    # let the forearm stop swinging before the lift moves
TOP_SETTLE_S = 0.5      # rest at the top before descending
BOTTOM_SETTLE_S = 0.5   # rest at the bottom before letting go
ENABLE_FLUSH_S = 0.1    # ctrl flushed to the live pose before torque comes on
ENABLE_SETTLE_S = 0.3   # daemon mode-switch stall after torque on

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


def j0_extremes(cfg, side):
    """(top, bottom) J0 positions in motor turns for this arm."""
    cal = json.loads(Path(
        f"/home/bracketbot/bbos/bbos/daemons/arm_{side}/ranges.calibration.json"
    ).read_text())
    ends = np.array([cal["cal_min"][J0], cal["cal_max"][J0]], dtype=np.float64)
    # The first startup waypoint is always below home, so it gives the "down" sign.
    down = float(np.sign(float(cfg.startup_waypoints[0][J0]) - float(cfg.home[J0]))) or 1.0
    bottom = ends[np.argmax(ends * down)]
    top = ends[np.argmin(ends * down)]
    return float(top), float(bottom)


class Arm:
    def __init__(self, side):
        self.side = side
        self.cfg = Config(f"arm_{side}")
        self.dof = self.cfg.dof
        self.r_state = Reader(f"arm_{side}.state")
        self.w_ctrl = Writer(f"arm_{side}.ctrl", Type("arm_ctrl"), keeptime=False)
        self.w_torque = Writer(f"arm_{side}.torque", Type("arm_torque"))
        self.top, self.bottom = j0_extremes(self.cfg, side)
        self.elbow_90 = float(self.cfg.home[ELBOW])
        self.cmd = None

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
    while time.monotonic() - t0 < secs and not _stop:
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["left", "right", "both"], default="both")
    ap.add_argument("--speed", type=float, default=J0_SPEED, help="J0 speed, turns/s")
    args = ap.parse_args()
    sides = ["left", "right"] if args.arm == "both" else [args.arm]

    arms = [Arm(s) for s in sides]
    try:
        # Only an OFF->ON transition reseeds the daemon's command filter, so disable first
        # and flush ctrl to the live pose before energizing.
        for a in arms:
            a.set_torque(False)
            a.write_cmd(a.live())
            print(f"[pickup] {a.side}: J0 {a.cmd[J0]:+.3f}  top {a.top:+.3f}  bottom {a.bottom:+.3f}"
                  f"  elbow {a.cmd[ELBOW]:+.3f} -> {a.elbow_90:+.3f}", flush=True)
        hold(arms, ENABLE_FLUSH_S)
        for a in arms:
            a.set_torque(True)
        hold(arms, ENABLE_SETTLE_S)

        # J3 carries the forearm, so it eases on a quintic (as in homing.py). It is held at
        # 90 deg by every later ramp, which only moves J0 and keeps the rest of the command.
        print("[pickup] elbow -> 90 deg", flush=True)
        ramp_joint(arms, ELBOW, [a.elbow_90 for a in arms], ELBOW_SPEED, ease=smootherstep)
        hold(arms, ELBOW_SETTLE_S)

        print("[pickup] J0 -> top", flush=True)
        ramp_joint(arms, J0, [a.top for a in arms], args.speed)
        hold(arms, TOP_SETTLE_S)

        print("[pickup] J0 -> bottom", flush=True)
        ramp_joint(arms, J0, [a.bottom for a in arms], args.speed)
        hold(arms, BOTTOM_SETTLE_S)
        for a in arms:
            live = a.live()
            print(f"[pickup] {a.side}: J0 at {live[J0]:+.3f}  elbow at {live[ELBOW]:+.3f}", flush=True)
    finally:
        # Leave the arms limp on exit.
        for a in arms:
            a.set_torque(False)
        for a in arms:
            a.close()


if __name__ == "__main__":
    main()