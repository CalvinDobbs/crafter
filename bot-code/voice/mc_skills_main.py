# /// script
# dependencies = [
#   "bbos",
#   "fastapi",
#   "uvicorn",
#   "numpy",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""mc_skills — the ONLY process allowed to open hardware Writers.

Everything that moves the robot lives behind this HTTP API, so at most one
process ever owns arm_*.ctrl / *.torque / drive.ctrl / speaker.audio / led.ctrl.
Everyone else calls skills_client.SkillsClient (or runs this file with
MOCK=1, which serves the same API without touching bbos — for off-robot dev).

Endpoints (all JSON):
  GET  /health
  GET  /state            -> arm joint positions etc.
  POST /home             -> staged torque enable + home both arms (blocking)
  POST /park             -> descend + torque off (blocking)
  POST /gripper   {open: bool, arm: "right"|"left"}
  POST /goto      {pos:[x,y,z], quat:[x,y,z,w]|null, duration: s, arm}
  POST /pick      {pos:[x,y,z], arm}   -> approach, descend, close, lift
  POST /place     {pos:[x,y,z], arm}   -> approach, descend, open, retreat
  POST /rotate    {rad: float}         -> open-loop base spin (for scanning)
  POST /say       {text: str}          -> TTS stub (wav playback hook)
  POST /celebrate {}                   -> LED + wiggle stub

Run:  uv run ~/bbapps/mc_skills/main.py            (on the robot)
      MOCK=1 uv run ~/bbapps/mc_skills/main.py     (laptop / no hardware)
"""
import os
import sys
import time
import signal
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Voice queue — same directory as this file when run from bot-code/voice/ or installed
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hook import narrate
import hook
import moves
try:
    from packs import set_pack
    set_pack(os.environ.get("VOICE_PACK", "neutral"))
except Exception as e:
    print(f"[mc_skills] voice pack init: {e}", flush=True)

MOCK = os.environ.get("MOCK") == "1"
ARM = os.environ.get("MC_ARM", "right")   # build arm
PORT = int(os.environ.get("MC_PORT", "8006"))

GRIPPER_IDX = 7
APPROACH_H = 0.12        # m above target for the approach waypoint
LIFT_H = 0.15            # m to lift after grasp
GRASP_SECS = 0.8         # dwell while closing/releasing gripper
MOVE_SECS = 2.0          # default segment duration
# EE orientation for top-down grasps: gripper axis (body +z) -> world -z.
# xyzw. TUNE on the robot — try [1,0,0,0] vs [0,1,0,0] (they differ by roll).
DOWN_QUAT = [1.0, 0.0, 0.0, 0.0]
ROTATE_SPEED = 0.5       # rad/s for /rotate

if not MOCK:
    sys.path.insert(0, "/home/bracketbot/bbapps/quest_teleop")
    from bbos import Reader, Writer, Type, Config
    from scripts.homing import staged_home_arms, park_arms, smoothstep
    import json
    from pathlib import Path

app = FastAPI()
_busy = threading.Lock()   # serialize all motion commands


# ---------------------------------------------------------------------------
# hardware layer (skipped entirely under MOCK)
# ---------------------------------------------------------------------------

class Arm:
    def __init__(self, side):
        self.side = side
        self.cfg = Config(f"arm_{side}")
        self.r_state = Reader(f"arm_{side}.state")
        self.w_ctrl = Writer(f"arm_{side}.ctrl", Type("arm_ctrl"))
        self.w_torque = Writer(f"arm_{side}.torque", Type("arm_torque"))
        cal = json.loads(Path(
            f"/home/bracketbot/bbos/bbos/daemons/arm_{side}/ranges.calibration.json"
        ).read_text())
        self.grip_open = float(cal["cal_min"][GRIPPER_IDX])   # TUNE: flip these
        self.grip_closed = float(cal["cal_max"][GRIPPER_IDX]) # if /gripper is backwards
        self.enabled = False

    def spec(self):
        return dict(cfg=self.cfg, r_state=self.r_state,
                    w_ctrl=self.w_ctrl, w_torque=self.w_torque)

    def live(self):
        while not self.r_state.ready():
            time.sleep(0.005)
        return np.array(self.r_state.data["pos"], dtype=np.float64)

    def _ramp_to(self, target_turns, secs):
        """Joint-space smoothstep ramp of the whole 8-dof command."""
        w = self.w_ctrl
        saved = w._keeptime
        w._keeptime = False
        dt = 1.0 / 200.0
        try:
            start = self.live()
            t0 = time.monotonic()
            while True:
                t = time.monotonic() - t0
                f = smoothstep(t / max(secs, 1e-3))
                cmd = start + f * (np.asarray(target_turns) - start)
                with w.buf() as b:
                    b["pos"][:] = cmd.astype(np.float32)
                    b["vel"][:] = np.zeros(self.cfg.dof, dtype=np.float32)
                    b["tau"][:] = np.zeros(self.cfg.dof, dtype=np.float32)
                    b["alpha"] = 0.0
                if t >= secs:
                    break
                time.sleep(dt)
        finally:
            w._keeptime = saved

    def goto(self, pos, quat, secs):
        """IK solve at pos/quat (base frame), ramp arm joints there, hold gripper."""
        live = self.live()
        q_now = self.cfg.q2urdf(live)[:7]
        self.cfg.ik.reset(list(q_now))
        sol = self.cfg.ik.solve(list(pos), list(quat))
        if sol is None:
            return False
        full = self.cfg.q2urdf(live)
        full[:7] = np.asarray(sol)[:7]
        target = self.cfg.urdf2q(full)
        target[GRIPPER_IDX] = live[GRIPPER_IDX]   # hold gripper through the move
        self._ramp_to(target, secs)
        return True

    def gripper(self, open_, secs=GRASP_SECS):
        live = self.live()
        t = live.copy()
        t[GRIPPER_IDX] = self.grip_open if open_ else self.grip_closed
        self._ramp_to(t, secs)


class Body:
    """Owns every hardware Writer. Instantiated lazily on first real call."""
    def __init__(self):
        self.arm = Arm(ARM)
        self.other = Arm("left" if ARM == "right" else "right")
        self.w_drive = Writer("drive.ctrl", Type("drive_ctrl"))
        self.w_led = Writer("led.ctrl", Type("led_ctrl"))
        try:
            self.w_speaker = Writer("speaker.audio", Type("speaker_audio"))
            def _sink(frame: bytes):
                with self.w_speaker.buf() as b:
                    b["audio"] = np.frombuffer(frame, dtype=np.int16).reshape(-1, 1)
            hook.set_audio_sink(_sink)
            print("[mc_skills] speaker.audio writer connected", flush=True)
        except Exception as e:
            print(f"[mc_skills] speaker.audio init warning: {e}", flush=True)
        self.homed = False

    def home(self):
        narrate("home.start")
        staged_home_arms([self.arm.spec(), self.other.spec()])
        self.homed = True

    def park(self):
        park_arms([self.arm.spec(), self.other.spec()])
        self.homed = False

    def pick(self, pos):
        a = self.arm
        above = [pos[0], pos[1], pos[2] + APPROACH_H]
        a.gripper(True)
        narrate("pick.approach")
        if not a.goto(above, DOWN_QUAT, MOVE_SECS):
            narrate("fail")
            return False, "approach IK failed"
        narrate("pick.descend")
        if not a.goto(pos, DOWN_QUAT, MOVE_SECS * 0.7):
            narrate("fail")
            return False, "descend IK failed"
        narrate("pick.grasp")
        a.gripper(False)
        narrate("pick.lift")
        a.goto(above, DOWN_QUAT, MOVE_SECS * 0.7)
        return True, "ok"

    def place(self, pos):
        a = self.arm
        above = [pos[0], pos[1], pos[2] + APPROACH_H]
        narrate("place.approach")
        if not a.goto(above, DOWN_QUAT, MOVE_SECS):
            narrate("fail")
            return False, "approach IK failed"
        narrate("place.descend")
        if not a.goto(pos, DOWN_QUAT, MOVE_SECS * 0.7):
            narrate("fail")
            return False, "descend IK failed"
        narrate("place.release")
        a.gripper(True)
        narrate("place.retreat")
        a.goto(above, DOWN_QUAT, MOVE_SECS * 0.7)
        return True, "ok"

    def rotate(self, rad):
        """Open-loop in-place spin. Positive rad = CCW (verify sign on robot)."""
        narrate("scan.start")
        dt = abs(rad) / ROTATE_SPEED
        w = ROTATE_SPEED * (1 if rad >= 0 else -1)
        t0 = time.monotonic()
        while time.monotonic() - t0 < dt:
            with self.w_drive.buf() as b:
                b["twist"] = np.array([0.0, w], dtype=np.float32)
            time.sleep(0.02)
        with self.w_drive.buf() as b:
            b["twist"] = np.array([0.0, 0.0], dtype=np.float32)

    def celebrate(self):
        narrate("build.done")
        for _ in range(4):
            for rgb in ((0, 255, 0), (0, 0, 255)):
                with self.w_led.buf() as b:
                    b["rgb"] = np.array(rgb, dtype=np.uint8)
                    b["brightness"] = -1
                    b["period_ms"] = 0
                time.sleep(0.25)


_body: Body | None = None
_body_lock = threading.Lock()


def get_body() -> Body:
    global _body
    with _body_lock:
        if _body is None:
            _body = Body()
        return _body


def _shutdown(*_):
    try:
        if _body is not None and _body.homed:
            _body.park()
    finally:
        sys.exit(0)


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class PosReq(BaseModel):
    pos: list[float]
    arm: str | None = None

class GotoReq(PosReq):
    quat: list[float] | None = None
    duration: float = MOVE_SECS

class GripReq(BaseModel):
    open: bool
    arm: str | None = None

class TextReq(BaseModel):
    text: str

class RotReq(BaseModel):
    rad: float


def _run(fn, *a, **kw):
    """Serialize motion; return a uniform response dict."""
    if MOCK:
        print(f"[mock] {fn.__name__}{a}{kw}", flush=True)
        return {"ok": True, "mock": True}
    if not _busy.acquire(blocking=False):
        return {"ok": False, "error": "busy"}
    try:
        return {"ok": True, "result": fn(*a, **kw)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        _busy.release()


@app.get("/health")
def health():
    return {"ok": True, "mock": MOCK, "arm": ARM,
            "homed": (_body.homed if _body else False)}

@app.get("/state")
def state():
    if MOCK:
        return {"ok": True, "mock": True}
    b = get_body()
    return {"ok": True, "pos": b.arm.live().tolist()}

@app.post("/home")
def home():
    if MOCK:
        moves.home()
        return {"ok": True, "mock": True}
    return _run(lambda: get_body().home())

@app.post("/park")
def park():
    if MOCK:
        moves.park()
        return {"ok": True, "mock": True}
    return _run(lambda: get_body().park())

@app.post("/gripper")
def gripper(r: GripReq):
    if MOCK:
        moves.gripper(r.open)
        return {"ok": True, "mock": True}
    narrate("place.release" if r.open else "pick.grasp")
    return _run(lambda: get_body().arm.gripper(r.open))

@app.post("/goto")
def goto(r: GotoReq):
    q = r.quat or DOWN_QUAT
    return _run(lambda: get_body().arm.goto(r.pos, q, r.duration))

@app.post("/pick")
def pick(r: PosReq):
    if MOCK:
        moves.pick(r.pos)
        return {"ok": True, "mock": True}
    return _run(lambda: get_body().pick(r.pos))

@app.post("/place")
def place(r: PosReq):
    if MOCK:
        moves.place(r.pos)
        return {"ok": True, "mock": True}
    return _run(lambda: get_body().place(r.pos))

@app.post("/rotate")
def rotate(r: RotReq):
    if MOCK:
        moves.rotate(r.rad)
        return {"ok": True, "mock": True}
    return _run(lambda: get_body().rotate(r.rad))

@app.post("/celebrate")
def celebrate():
    if MOCK:
        moves.celebrate()
        return {"ok": True, "mock": True}
    return _run(lambda: get_body().celebrate())

@app.post("/say")
def say(r: TextReq):
    # Non-blocking voice playback: does NOT acquire _busy.
    hook.say_text(r.text)
    return {"ok": True}


if __name__ == "__main__":
    print(f"[mc_skills] arm={ARM} mock={MOCK} port={PORT}", flush=True)
    if not MOCK:
        get_body()   # open writers up front so a conflict fails fast
        print("[mc_skills] hardware claimed; call POST /home to energize", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="error")
