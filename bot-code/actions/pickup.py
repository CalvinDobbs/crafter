# /// script
# dependencies = [
#   "bbos",
#   "fastapi",
#   "uvicorn",
#   "wsproto",
#   "numpy",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""pickup — bimanual box pickup with a web control panel.

Phases (joint space, both arms together — no IK):
    1  DOWN    J0 vertical stage all the way down (grippers at floor level)
    2  OUT     J2 swings both arms sideways, away from the body centre
    3  PINCH   J2 swings back inward past neutral, squeezing the box between
               the forearms/grippers; grippers close
    4  RAISE   J0 back up to the top (shoulder level), still pinching
The final pose is held until Release / exit (torque off, arms limp).

Nothing moves until you click a button. This process owns arm_left.* and
arm_right.* writers: stop mc_skills / quest_teleop / mimic first.

    uv run pickup.py            # http://<robot-ip>:8009
    uv run pickup.py --mock     # UI only, no hardware
"""
import argparse
import asyncio
import json
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

PORT = 8009
GRIPPER_IDX = 7
J0, J2 = 0, 2
RATE_HZ = 200.0
HOLD_HZ = 50.0
J0_MARGIN_TURNS = 0.05          # stay off the hard stops
CAL_DIR = Path("/home/bracketbot/bbos/bbos/daemons")

# ---- behaviour parameters (editable live in the UI) ------------------------
DEFAULTS = {
    "down_frac": 0.95,      # 0 = home height .. 1 = bottom of the J0 stage (1.0 ~ grippers on the floor)
    "j0_speed": 0.4,        # turns/s for the vertical stage (park uses 0.4)
    "out_deg": 25.0,        # phase 2: swing outward by this much (per arm)
    "pinch_deg": 8.0,       # phase 3: swing inward PAST neutral by this much
    "out_sign": 1.0,        # flip to -1 if "out" moves the arms inward on your robot
    "grip_close": 1.0,      # gripper fraction at pinch (0 open .. 1 closed)
    "grip_open": 0.0,       # gripper fraction during down/out
    "raise_frac": 0.0,      # phase 4 J0 height (0 = top / shoulder level)
    "move_secs": 2.0,       # duration of the out / pinch swings
    "settle_secs": 0.5,     # pause between phases in "Run all"
}
PHASES = ("down", "out", "pinch", "raise")

_stop = False


def _sig(*_):
    global _stop
    _stop = True


def smoothstep(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


# ---------------------------------------------------------------------------
# hardware
# ---------------------------------------------------------------------------

class Arm:
    def __init__(self, side):
        from bbos import Reader, Writer, Type, Config
        self.side = side
        self.cfg = Config(f"arm_{side}")
        self.dof = self.cfg.dof
        self.r_state = Reader(f"arm_{side}.state")
        self.w_ctrl = Writer(f"arm_{side}.ctrl", Type("arm_ctrl"))
        self.w_torque = Writer(f"arm_{side}.torque", Type("arm_torque"))
        cal = json.loads((CAL_DIR / f"arm_{side}" / "ranges.calibration.json").read_text())
        lo, hi = np.asarray(cal["cal_min"], float), np.asarray(cal["cal_max"], float)
        self.home = np.asarray(self.cfg.home, dtype=np.float64)
        # J0: 0 is the top; the calibrated end far from 0 is the bottom.
        self.j0_bottom = lo[J0] if abs(lo[J0]) > abs(hi[J0]) else hi[J0]
        self.j0_bottom -= np.sign(self.j0_bottom) * J0_MARGIN_TURNS
        # TUNE: flip if the gripper moves the wrong way.
        self.grip_open, self.grip_closed = float(lo[GRIPPER_IDX]), float(hi[GRIPPER_IDX])
        # "out" = away from the body centre: +y for left, -y for right (URDF rad on J2).
        self.out_dir = 1.0 if side == "left" else -1.0
        self.cmd = None

    def spec(self):
        return dict(cfg=self.cfg, r_state=self.r_state, w_ctrl=self.w_ctrl, w_torque=self.w_torque)

    def live(self):
        while not self.r_state.ready():
            time.sleep(0.005)
        return np.array(self.r_state.data["pos"], dtype=np.float64)

    def peek(self):
        self.r_state.ready()
        return np.array(self.r_state.data["pos"], dtype=np.float64)

    def write(self, pos):
        self.cmd = np.asarray(pos, dtype=np.float64)
        with self.w_ctrl.buf() as b:
            b["pos"][:] = self.cmd.astype(np.float32)
            b["vel"][:] = np.zeros(self.dof, dtype=np.float32)
            b["tau"][:] = np.zeros(self.dof, dtype=np.float32)
            b["alpha"] = 0.0

    def limp(self):
        with self.w_torque.buf() as b:
            b["enable"][:] = np.zeros(self.dof, dtype=np.bool_)
        self.cmd = None

    def j0_at(self, frac):
        return float(self.home[J0] + np.clip(frac, 0.0, 1.0) * (self.j0_bottom - self.home[J0]))

    def grip_at(self, frac):
        return self.grip_open + float(np.clip(frac, 0.0, 1.0)) * (self.grip_closed - self.grip_open)

    def pose(self, j0_frac, swing_deg, grip_frac):
        """Home posture with J0 at j0_frac, J2 swung by swing_deg (+ = out), gripper set."""
        u = self.cfg.q2urdf(self.home.copy())
        u[J2] += self.out_dir * np.deg2rad(swing_deg)
        q = self.cfg.urdf2q(u)
        q[J0] = self.j0_at(j0_frac)
        q[GRIPPER_IDX] = self.grip_at(grip_frac)
        return q


class Rig:
    def __init__(self):
        sys.path.insert(0, "/home/bracketbot/bbapps/quest_teleop")
        from scripts.homing import staged_home_arms
        self._home_fn = staged_home_arms
        self.arms = (Arm("left"), Arm("right"))
        self._paced = [w for a in self.arms for w in (a.w_ctrl, a.w_torque)]
        self._saved = [w._keeptime for w in self._paced]
        for w in self._paced:
            w._keeptime = False
        self.homed = False
        self.abort = False
        self.swing = 0.0     # current J2 swing (deg, + = out)
        self.j0_frac = 0.0
        self.grip = 0.0

    def home(self):
        for w, kt in zip(self._paced, self._saved):
            w._keeptime = kt
        try:
            self._home_fn([a.spec() for a in self.arms])
        finally:
            for w in self._paced:
                w._keeptime = False
        for a in self.arms:
            a.cmd = a.live()
        self.swing, self.j0_frac, self.grip = 0.0, 0.0, 0.0
        self.homed = True

    def ramp(self, targets, secs):
        starts = [a.cmd if a.cmd is not None else a.live() for a in self.arms]
        dt = 1.0 / RATE_HZ
        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            f = smoothstep(t / max(secs, 1e-3))
            for a, p0, p1 in zip(self.arms, starts, targets):
                a.write(p0 + f * (p1 - p0))
            if t >= secs or self.abort:
                return not self.abort
            time.sleep(dt)

    def move(self, p, j0_frac=None, swing=None, grip=None, secs=None):
        """Ramp both arms to (j0_frac, swing, grip); None keeps the current value."""
        j0_frac = self.j0_frac if j0_frac is None else j0_frac
        swing = self.swing if swing is None else swing
        grip = self.grip if grip is None else grip
        if secs is None:
            dj0 = max(abs(a.j0_at(j0_frac) - a.j0_at(self.j0_frac)) for a in self.arms)
            secs = max(p["move_secs"], dj0 / max(p["j0_speed"], 1e-3))
        targets = [a.pose(j0_frac, swing, grip) for a in self.arms]
        ok = self.ramp(targets, secs)
        if ok:
            self.j0_frac, self.swing, self.grip = j0_frac, swing, grip
        return ok

    def reassert(self):
        for a in self.arms:
            if a.cmd is not None:
                a.write(a.cmd)

    def state(self):
        return {a.side: np.round(a.peek(), 3).tolist() for a in self.arms}

    def limp(self):
        for a in self.arms:
            a.limp()
        self.homed = False

    def shutdown(self):
        self.limp()
        for w, kt in zip(self._paced, self._saved):
            w._keeptime = kt


# ---------------------------------------------------------------------------
# job runner
# ---------------------------------------------------------------------------

class Body:
    """One motion thread: runs queued jobs, re-asserts the held pose when idle."""

    def __init__(self, mock):
        self.mock = mock
        self.rig = None if mock else Rig()
        self.params = dict(DEFAULTS)
        self.job = None
        self.state = {}
        self.log_lines, self.log_q, self.jobs = [], queue.Queue(), queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def log(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self.log_lines.append(line)
        del self.log_lines[:-200]
        self.log_q.put(line)

    def status(self):
        r = self.rig
        return {"type": "status", "mock": self.mock, "job": self.job,
                "homed": bool(r and r.homed), "arms": self.state, "params": self.params,
                "pose": None if not r else {"j0_frac": r.j0_frac, "swing": r.swing, "grip": r.grip}}

    def submit(self, name, **kw):
        if name == "stop":
            if self.rig:
                self.rig.abort = True
            self.log("[web] stop — arms hold where they are")
            return {"ok": True}
        if self.job is not None:
            return {"ok": False, "error": f"busy: {self.job}"}
        self.jobs.put((name, kw))
        return {"ok": True}

    def set_params(self, upd):
        for k, v in upd.items():
            if k in self.params:
                self.params[k] = float(v)
        self.log(f"[web] params: " + ", ".join(f"{k}={self.params[k]:g}" for k in upd if k in self.params))

    def _loop(self):
        dt = 1.0 / HOLD_HZ
        while not _stop:
            try:
                name, kw = self.jobs.get(timeout=dt)
            except queue.Empty:
                if self.rig:
                    if self.rig.homed:
                        self.rig.reassert()
                    self.state = self.rig.state()
                continue
            self.job = name
            if self.rig:
                self.rig.abort = False
            try:
                self._run(name, kw)
            except Exception as e:
                self.log(f"[web] {name} failed: {type(e).__name__}: {e}")
            finally:
                self.job = None

    def _ready(self):
        if self.mock:
            return False
        if not self.rig.homed:
            self.log("[web] arms not homed — click Home first")
            return False
        return True

    def _phase(self, name):
        p = self.params
        self.log(f"[web] phase {name}")
        if self.mock:
            return True
        if name == "down":
            return self.rig.move(p, j0_frac=p["down_frac"], grip=p["grip_open"])
        if name == "out":
            return self.rig.move(p, swing=p["out_sign"] * p["out_deg"], grip=p["grip_open"])
        if name == "pinch":
            return self.rig.move(p, swing=-p["out_sign"] * p["pinch_deg"], grip=p["grip_close"])
        if name == "raise":
            return self.rig.move(p, j0_frac=p["raise_frac"])
        raise ValueError(name)

    def _run(self, name, kw):
        p = self.params
        if name == "home":
            if self.mock:
                return self.log("[mock] home")
            self.log("[web] homing arms (staged)")
            self.rig.home()
            self.log("[web] homed — holding")
        elif name in PHASES:
            if self._ready():
                self._phase(name)
        elif name == "pickup":
            self.log("[web] PICKUP: down -> out -> pinch -> raise")
            if not self._ready() and not self.mock:
                return
            for ph in PHASES:
                if _stop or (self.rig and self.rig.abort) or not self._phase(ph):
                    return self.log("[web] pickup interrupted")
                time.sleep(p["settle_secs"])
            self.log("[web] pickup complete — holding box at shoulder level")
        elif name == "grip":
            if self._ready():
                self.rig.move(p, grip=float(kw.get("frac", 0.0)), secs=0.8)
        elif name == "neutral":
            if self._ready():
                self.rig.move(p, swing=0.0)
        elif name == "release":
            if self.rig:
                self.rig.limp()
            self.log("[web] released — torque off, arms limp")
        else:
            self.log(f"[web] unknown command {name}")


# ---------------------------------------------------------------------------
# web
# ---------------------------------------------------------------------------

HTML = """
<!doctype html><meta charset=utf-8>
<title>[bot] Pickup</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Departure+Mono&display=swap">
<style>
:root {
  --text-primary:#000; --text-muted:#5e5e5e; --text-faint:#9ca3af; --border:#e5e5e5; --border-medium:#d8d8d8;
  --bg:#fff; --surface:#fbfbfb; --primary:#222; --brand-orange:#dc6100; --brand-blue:#2563eb; --danger:#b91c1c; --radius:3px;
  --font-sans:"Helvetica Neue",Helvetica,Arial,sans-serif; --font-mono:"Departure Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text-primary);font-family:var(--font-sans);-webkit-font-smoothing:antialiased}
.topbar{height:48px;display:flex;align-items:center;gap:10px;padding:0 20px;border-bottom:1px solid var(--border);background:var(--surface)}
.logo{font-family:var(--font-mono);font-size:20px;font-weight:500}
.page-title{font-size:14px;font-weight:500;color:var(--text-muted)}
.layout{display:grid;grid-template-columns:minmax(360px,1fr) 340px;gap:20px;height:calc(100vh - 48px);padding:16px 20px}
.col{display:flex;flex-direction:column;gap:14px;min-height:0}
#feed{width:100%;max-height:360px;object-fit:contain;border:1px solid var(--border);border-radius:var(--radius);background:var(--surface)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:12px 14px;display:flex;flex-direction:column;gap:10px}
.card h3{margin:0;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:var(--text-muted)}
.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
button{font-family:var(--font-mono);font-size:12px;padding:8px 12px;background:var(--surface);border:1px solid var(--border-medium);border-radius:var(--radius);box-shadow:0 1px 0 0 var(--border-medium);color:var(--text-primary);cursor:pointer}
button:hover{background:#f0f0f0} button:disabled{opacity:.45;cursor:default}
button.primary{background:var(--primary);color:#fff;border-color:var(--primary);box-shadow:none}
button.auto{background:var(--brand-orange);color:#fff;border-color:var(--brand-orange);box-shadow:none;font-weight:600}
button.danger{color:var(--danger);border-color:var(--danger)}
.params{display:grid;grid-template-columns:1fr 90px;gap:6px 10px;align-items:center;font-size:12px;color:var(--text-muted)}
.params input{width:100%;font-family:var(--font-mono);font-size:12px;padding:4px 6px;border:1px solid var(--border-medium);border-radius:var(--radius)}
.stat{display:flex;gap:8px;font-size:12px;color:var(--text-muted)} .stat b{color:var(--text-primary);font-weight:500;font-family:var(--font-mono)}
#log{flex:1 1 auto;min-height:100px;overflow-y:auto;font-family:var(--font-mono);font-size:11px;white-space:pre-wrap;color:var(--text-muted);background:#fff;border:1px solid var(--border);border-radius:var(--radius);padding:8px}
.conn-dot{width:8px;height:8px;border-radius:50%;background:var(--text-faint);display:inline-block;margin-right:6px} .conn-dot.live{background:#16a34a}
.badge{font-family:var(--font-mono);font-size:11px;padding:2px 6px;border-radius:var(--radius);border:1px solid var(--border-medium);color:var(--text-muted)}
.badge.busy{color:var(--brand-orange);border-color:var(--brand-orange)} .badge.mock{color:var(--brand-blue);border-color:var(--brand-blue)}
.hint{font-size:11px;color:var(--text-faint)}
</style>
<div class="topbar">
  <span class="logo">[bot]</span><span class="page-title">Pickup</span>
  <span id="mock-badge" class="badge mock" style="display:none">MOCK</span>
  <span id="job-badge" class="badge">idle</span>
  <span style="flex:1"></span>
  <span class="hint"><span id="conn-dot" class="conn-dot"></span><span id="conn-text">connecting…</span></span>
</div>
<div class="layout">
  <div class="col">
    <img id="feed" alt="Camera Feed">
    <div class="card">
      <h3>Control</h3>
      <div class="row">
        <button id="b-home" class="primary" onclick="cmd('home')">Home arms</button>
        <button id="b-pickup" class="auto" onclick="cmd('pickup')">Run pickup (1→4)</button>
        <span style="flex:1"></span>
        <button class="danger" onclick="cmd('stop')">Stop</button>
        <button class="danger" onclick="cmd('release')">Release (limp)</button>
      </div>
      <h3>Phases</h3>
      <div class="row">
        <button class="step" onclick="cmd('down')">1 Down</button>
        <button class="step" onclick="cmd('out')">2 Out</button>
        <button class="step" onclick="cmd('pinch')">3 Pinch</button>
        <button class="step" onclick="cmd('raise')">4 Raise</button>
        <span style="flex:1"></span>
        <button class="step" onclick="cmd('neutral')">Arms neutral</button>
        <button class="step" onclick="cmd('grip',{frac:0})">Grip open</button>
        <button class="step" onclick="cmd('grip',{frac:1})">Grip close</button>
      </div>
      <div class="stat">Pose <b id="pose">—</b></div>
      <div class="stat">Left <b id="arm-l">—</b></div>
      <div class="stat">Right <b id="arm-r">—</b></div>
    </div>
    <div id="log"></div>
  </div>
  <div class="col" style="overflow-y:auto">
    <div class="card">
      <h3>Behaviour</h3>
      <div class="params">
        <label>1 Down: J0 travel (0–1)</label><input type="number" step="0.05" min="0" max="1" data-k="down_frac">
        <label>J0 speed (turns/s)</label><input type="number" step="0.05" min="0.05" data-k="j0_speed">
        <label>2 Out: swing (deg)</label><input type="number" step="1" data-k="out_deg">
        <label>3 Pinch: past-centre (deg)</label><input type="number" step="0.5" data-k="pinch_deg">
        <label>Out direction (+1 / -1)</label><input type="number" step="2" min="-1" max="1" data-k="out_sign">
        <label>Grip while open (0–1)</label><input type="number" step="0.05" min="0" max="1" data-k="grip_open">
        <label>Grip at pinch (0–1)</label><input type="number" step="0.05" min="0" max="1" data-k="grip_close">
        <label>4 Raise: J0 travel (0=top)</label><input type="number" step="0.05" min="0" max="1" data-k="raise_frac">
        <label>Swing time (s)</label><input type="number" step="0.1" min="0.2" data-k="move_secs">
        <label>Pause between phases (s)</label><input type="number" step="0.1" min="0" data-k="settle_secs">
      </div>
      <div class="row">
        <button class="primary" onclick="applyParams()">Apply</button>
        <button onclick="resetParams()">Reset defaults</button>
        <span id="params-note" class="hint"></span>
      </div>
      <div class="hint">Changes apply to the next phase you trigger. If "2 Out" moves the arms toward each other, set Out direction to -1.</div>
    </div>
  </div>
</div>
<script>
const feedEl=document.getElementById("feed"); let prevUrl=null, paramsLoaded=false;
const ws=new WebSocket((location.protocol==="https:"?"wss://":"ws://")+location.host+"/ws"); ws.binaryType="arraybuffer";
ws.onopen=()=>{document.getElementById("conn-dot").classList.add("live");document.getElementById("conn-text").textContent="live";};
ws.onclose=()=>{document.getElementById("conn-dot").classList.remove("live");document.getElementById("conn-text").textContent="disconnected";};
ws.onmessage=(e)=>{
  if(e.data instanceof ArrayBuffer){ if(prevUrl)URL.revokeObjectURL(prevUrl); prevUrl=URL.createObjectURL(new Blob([e.data],{type:"image/jpeg"})); feedEl.src=prevUrl; return; }
  let m; try{m=JSON.parse(e.data);}catch(_){return;}
  if(m.type==="log") appendLog(m.line); else if(m.type==="status") renderStatus(m);
};
function appendLog(l){const el=document.getElementById("log"); el.textContent+=l+"\\n"; el.scrollTop=el.scrollHeight;}
const fmt3=v=>v.map(x=>(x>=0?" ":"")+x.toFixed(3)).join(" ");
function renderStatus(s){
  const jb=document.getElementById("job-badge");
  jb.textContent=s.job?"running: "+s.job:(s.homed?"idle · homed":"idle · not homed"); jb.classList.toggle("busy",!!s.job);
  document.getElementById("mock-badge").style.display=s.mock?"":"none";
  document.querySelectorAll("#b-home,#b-pickup").forEach(b=>b.disabled=!!s.job);
  document.querySelectorAll(".step").forEach(b=>b.disabled=!!s.job||(!s.homed&&!s.mock));
  document.getElementById("pose").textContent=s.pose?`J0 ${(s.pose.j0_frac*100).toFixed(0)}% down · swing ${s.pose.swing.toFixed(1)}° · grip ${(s.pose.grip*100).toFixed(0)}%`:"—";
  document.getElementById("arm-l").textContent=s.arms.left?fmt3(s.arms.left):"—";
  document.getElementById("arm-r").textContent=s.arms.right?fmt3(s.arms.right):"—";
  if(!paramsLoaded){fillParams(s.params);paramsLoaded=true;}
}
function fillParams(p){document.querySelectorAll("[data-k]").forEach(i=>i.value=p[i.dataset.k]);}
function readParams(){const o={};document.querySelectorAll("[data-k]").forEach(i=>{const v=parseFloat(i.value);if(!isNaN(v))o[i.dataset.k]=v;});return o;}
async function post(path,body){const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body||{})});return r.json();}
async function cmd(name,args){const r=await post("/api/cmd",{name,args:args||{}}); if(!r.ok)appendLog("!! "+r.error);}
async function applyParams(){const r=await post("/api/params",readParams()); const n=document.getElementById("params-note"); n.textContent=r.ok?"applied":"error: "+r.error; setTimeout(()=>n.textContent="",1500);}
async function resetParams(){fillParams(await (await fetch("/api/defaults")).json()); applyParams();}
</script>
"""


def make_app(body, mock):
    app = FastAPI()
    jpeg_q = queue.Queue(maxsize=2)

    def camera_loop():
        from bbos import Reader
        with Reader("camera.head.jpeg") as r:
            while not _stop:
                if not r.ready():
                    time.sleep(0.005)
                    continue
                jpeg = bytes(r.data["jpeg"][:int(r.data["jpeg_len"])])
                try:
                    jpeg_q.put_nowait(jpeg)
                except queue.Full:
                    try:
                        jpeg_q.get_nowait()
                        jpeg_q.put_nowait(jpeg)
                    except queue.Empty:
                        pass

    if not mock:
        threading.Thread(target=camera_loop, daemon=True).start()

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return HTMLResponse(HTML)

    @app.get("/api/status")
    async def status():
        return body.status()

    @app.get("/api/defaults")
    async def defaults():
        return DEFAULTS

    @app.post("/api/params")
    async def set_params(upd: dict):
        try:
            body.set_params(upd)
        except (TypeError, ValueError) as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return {"ok": True, "params": body.params}

    @app.post("/api/cmd")
    async def cmd(req: dict):
        return body.submit(req.get("name", ""), **(req.get("args") or {}))

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        for line in body.log_lines[-50:]:
            await websocket.send_text(json.dumps({"type": "log", "line": line}))
        last_status = 0.0
        try:
            while True:
                sent = False
                try:
                    await websocket.send_bytes(jpeg_q.get_nowait())
                    sent = True
                except queue.Empty:
                    pass
                while not body.log_q.empty():
                    await websocket.send_text(json.dumps({"type": "log", "line": body.log_q.get_nowait()}))
                    sent = True
                now = time.monotonic()
                if now - last_status > 0.2:
                    await websocket.send_text(json.dumps(body.status()))
                    last_status, sent = now, True
                if not sent:
                    await asyncio.sleep(0.02)
        except WebSocketDisconnect:
            pass

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="UI only, no hardware")
    ap.add_argument("--port", type=int, default=PORT)
    a = ap.parse_args()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    try:
        body = Body(mock=a.mock)
    except RuntimeError as e:
        sys.exit(f"[pickup] could not claim arm writers ({e}); stop mc_skills / quest_teleop / mimic first")
    body.log(f"[pickup] http://0.0.0.0:{a.port} mock={a.mock} — arms idle until you click Home")
    server = uvicorn.Server(uvicorn.Config(make_app(body, a.mock), host="0.0.0.0", port=a.port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    try:
        while not _stop:
            time.sleep(0.1)
    finally:
        if body.rig:
            body.rig.shutdown()
        body.log("[pickup] arms limp, exiting")
        server.should_exit = True


if __name__ == "__main__":
    main()
