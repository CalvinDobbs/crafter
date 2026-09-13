"""Manual tool bench behind the panel's debug screen.

Every call here is one the reasoning loop makes: observe, find_build_sites, check_build_site, and
submit/status/monitor/cancel/stop on the same providers, wrapped in the same ActionRequest envelope
Agent builds. Nothing is relaxed for the operator's convenience — a tool that refuses by hand
refuses identically inside a build, which is the only reason driving it by hand is worth anything.

There is no simulator here, deliberately. A screen whose entire job is to say what the robot can
actually see would be worse than useless if some of what it showed were invented; a plausible
fake reads exactly like a real reading and you would tune against it. Off the robot this console
reports that it cannot open its providers, which is the truth.

The two halves open separately, and that separation is the point. Perception is readers only, so
it opens as soon as you look at anything. The action provider opens `arm_*.ctrl` and `drive.ctrl`
the moment it is constructed — this panel becomes the designated hardware owner — so it waits for
a deliberate arm. Disarming does not close it again: releasing those writers cuts arm torque, and
a held box would fall.
"""

from __future__ import annotations

import base64
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict

from agent_types import MOTION_OPS, ActionRequest, BuildRequirements, CancellationReceipt, Step
from contracts import BOX_SIZE

# The agent's own choice vocabulary. Five are submitted to the action provider; observe and
# select_site are perception calls the controller makes itself, and done/stop are decisions the
# loop takes rather than motions, so here they report the evidence those decisions would rest on.
OPERATIONS = ("observe", "look_around", "select_site", "approach_box", "pickup",
              "move_to_build", "place", "done", "stop")
ARGUMENTS = {"observe": ("site_id",), "look_around": ("search",), "select_site": ("site_id",),
             "approach_box": ("box_id", "site_id", "cell"), "pickup": ("box_id",),
             "move_to_build": ("box_id", "site_id", "cell"), "place": ("box_id", "site_id", "cell"),
             "done": (), "stop": ()}
DEFAULT_VOXEL = (BOX_SIZE, BOX_SIZE, BOX_SIZE)
LOG_LIMIT = 80
POLL_S = 0.05


def _jsonable(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def snapshot_to_dict(snapshot, now):
    """The whole snapshot, minus image bytes: those are served separately so polling stays light."""
    data = _jsonable(asdict(snapshot))
    data.pop("world_model_json", None)
    data["world_model"] = _jsonable(snapshot.world_model)
    data["age_s"] = round(max(0.0, now-snapshot.captured_at), 3)
    data["images"] = [dict({key: item for key, item in _jsonable(asdict(image)).items() if key != "data_url"},
                           age_s=round(max(0.0, now-image.captured_at), 3),
                           source=f"/api/debug/image?view={image.view}")
                      for image in snapshot.images]
    return data


def decode_image(image):
    """SceneImage data URL -> (media type, bytes). The dataclass already validated both."""
    header, _, encoded = image.data_url.partition(",")
    return header[len("data:"):-len(";base64")], base64.b64decode(encoded)


class DebugConsole:
    """Serializes hand-driven tool calls against the real providers and records what came back.

    State mutation happens under the lock; provider calls do not, so a slow sensor read cannot
    stall the page's polling. Concurrency is refused rather than queued: the action interface
    admits one motion at a time, and pretending otherwise would hide that.
    """

    def __init__(self, open_observations, open_actions=None, *, kind="robot",
                 voxel_size=DEFAULT_VOXEL, perception_url=None, fresh_s=2.0,
                 action_deadline=120.0, clock=time.time):
        self.open_observations, self.open_actions = open_observations, open_actions
        self.kind, self.perception_url = kind, perception_url
        self.voxel_size = tuple(voxel_size)
        self.fresh_s, self.action_deadline, self.clock = fresh_s, action_deadline, clock
        self.lock = threading.RLock()
        self.job_id = "debug-" + uuid.uuid4().hex[:8]
        self.requirements = BuildRequirements(((0, 0, 0),), self.voxel_size, (1, 1, 1))
        self.requirements_source = "single cell at the site origin"
        self.armed = False
        self.observations = None
        self.actions = None
        self.closers = []
        self.capabilities = {"observations": None, "actions": None}
        self.snapshot = None
        self.executor = None
        self.images = {}
        self.sites = ()
        self.site = None
        self.active = None
        self.outcome = None
        self.monitoring = None
        self.worker = None
        self.error = None
        self.revision = 0
        self.log = deque(maxlen=LOG_LIMIT)
        self.closed = False

    # ---- plumbing ------------------------------------------------------------------------

    def _perception(self):
        """Readers only. Opening this claims nothing and can run beside any hardware owner."""
        with self.lock:
            if self.closed:
                raise ValueError("the panel is shutting down")
            if self.observations is not None:
                return self.observations
        try:
            provider, close = self.open_observations()
            capabilities = _jsonable(asdict(provider.capabilities()))
        except Exception as exc:
            self._fail("perception could not be opened", exc)
        with self.lock:
            if self.observations is None:
                self.observations = provider
                self.capabilities["observations"] = capabilities
                if close is not None:
                    self.closers.append(close)
                self._record("session", f"Perception opened ({self.kind})", **capabilities)
            return self.observations

    def _motion(self):
        with self.lock:
            if self.actions is None:
                raise ValueError("arm actions before submitting a motion" if self.open_actions
                                 else "no action provider is configured; this panel owns no motors")
            return self.actions

    def _record(self, kind, title, **data):
        self.revision += 1
        self.log.appendleft({"id": self.revision, "at": self.clock(), "kind": kind,
                             "title": title, "data": _jsonable(data)})

    def _fail(self, title, exc):
        with self.lock:
            self.error = f"{title}: {type(exc).__name__}: {exc}"
            self._record("error", title, error=type(exc).__name__, detail=str(exc)[:400])
        raise ValueError(self.error)

    def _store(self, snapshot):
        with self.lock:
            self.snapshot = snapshot
            self.images = {image.view: decode_image(image) for image in snapshot.images}
            self.revision += 1
            return snapshot

    # ---- perception ----------------------------------------------------------------------

    def observe(self, site_id=None):
        observations = self._perception()
        site_id = site_id or (self.site.id if self.site else None)
        try:
            snapshot = observations.observe(site_id)
        except Exception as exc:
            self._fail("observe failed", exc)
        executor = self._read_executor()
        with self.lock:
            self.error = None
            self._record("perception", "observe()",
                         site_id=site_id, revision=snapshot.revision, valid=snapshot.valid,
                         pose_valid=snapshot.pose_valid, epoch=snapshot.epoch,
                         boxes=len(snapshot.boxes), images=len(snapshot.images),
                         occupancy=len(snapshot.occupancy),
                         occupancy_complete=snapshot.occupancy_complete,
                         holding=_jsonable(asdict(snapshot.holding)) if snapshot.holding else None,
                         warnings=list(snapshot.warnings))
        return self._store(snapshot)

    def _read_executor(self):
        """Only when the motors are already owned: reading must never open the writers."""
        with self.lock:
            actions = self.actions
        if actions is None:
            return None
        try:
            executor = actions.state()
        except Exception as exc:
            with self.lock:
                self._record("error", "state() failed", error=type(exc).__name__, detail=str(exc)[:400])
            return None
        with self.lock:
            self.executor = executor
            return executor

    def find_sites(self):
        observations = self._perception()
        try:
            sites = tuple(observations.find_build_sites(self.requirements))
        except Exception as exc:
            self._fail("find_build_sites failed", exc)
        with self.lock:
            self.sites, self.error = sites, None
            self._record("perception", "find_build_sites()", count=len(sites),
                         requirements=_jsonable(asdict(self.requirements)),
                         sites=[_jsonable(asdict(site)) for site in sites])
        return sites

    def select_site(self, site_id):
        if not isinstance(site_id, str) or not site_id:
            raise ValueError("choose a candidate site first")
        observations = self._perception()
        try:
            site = observations.check_build_site(site_id, self.requirements)
        except Exception as exc:
            self._fail("check_build_site failed", exc)
        with self.lock:
            self.site, self.error = site, None
            self._record("perception", f"check_build_site({site_id})",
                         selected=site is not None, site=_jsonable(asdict(site)) if site else None)
        return site

    def verify(self):
        """What FINAL_VERIFY would be looking at. Reports measured cells; confirms nothing."""
        snapshot = self.observe()
        with self.lock:
            measured = {tuple(item.cell): item for item in snapshot.occupancy}
            cells = [{"cell": list(cell),
                      "status": measured[cell].status if cell in measured else "unmeasured",
                      "box_id": measured[cell].box_id if cell in measured else None}
                     for cell in self.requirements.cells]
            occupied = sum(item["status"] == "occupied" for item in cells)
            self._record("verify", "done — target occupancy as measured now",
                         site_id=snapshot.site_id, complete=snapshot.occupancy_complete,
                         occupied=occupied, required=len(cells), cells=cells,
                         verdict=("the agent would accept this build as complete"
                                  if snapshot.occupancy_complete and occupied == len(cells)
                                  else "the agent would NOT accept done here"))
            return cells

    # ---- actions -------------------------------------------------------------------------

    def submit(self, operation, *, box_id=None, site_id=None, cell=None, search=None, reason=""):
        if operation not in MOTION_OPS:
            raise ValueError(f"{operation} is not a motion the action provider executes")
        with self.lock:
            if not self.armed:
                raise ValueError("arm actions before submitting a motion")
            if self.active is not None:
                raise ValueError("an action is still running; wait for it or stop the robot")
            if self.snapshot is None:
                raise ValueError("observe first: an action needs fresh geometry to reference")
        actions = self._motion()
        executor = self._read_executor()
        if executor is None:
            raise ValueError("the executor did not report its state; nothing will be submitted")
        with self.lock:
            snapshot, site = self.snapshot, self.site
            if site_id and (site is None or site.id != site_id):
                raise ValueError("select that site before referencing it in an action")
            box = next((b for b in snapshot.boxes if b.id == box_id), None) if box_id is not None else None
            if box_id is not None and box is None:
                raise ValueError("that box is not in the latest observation; observe again")
            step = Step(operation, reason or f"manual {operation} from the debug console",
                        box_id=box_id, site_id=site_id, cell=tuple(cell) if cell else None,
                        search=search)
            times = [snapshot.captured_at, executor.ts]
            if site is not None:
                times.append(site.ts)
            if box is not None and operation in {"approach_box", "pickup"}:
                times.append(box.last_seen)
            request = ActionRequest(uuid.uuid4().hex, self.job_id, step, snapshot.revision,
                                    snapshot.epoch, snapshot.frame_id, self.clock(),
                                    requirements=self.requirements, site=site, box=box,
                                    expires_at=min(times)+self.fresh_s)
            self._record("action", f"submit {operation}", request_id=request.request_id,
                         step=_jsonable(asdict(step)), expires_in_s=round(request.expires_at-self.clock(), 3))
        try:
            receipt = actions.submit(request)
        except Exception as exc:
            self._fail(f"{operation} was refused at admission", exc)
        with self.lock:
            self.active = {"request": request, "receipt": receipt}
            self.outcome, self.monitoring, self.error = None, None, None
            self._record("action", f"{operation} admitted", request_id=request.request_id,
                         action_id=receipt.action_id)
            self.worker = threading.Thread(target=self._follow, args=(request, receipt),
                                           name="debug-action", daemon=True)
            self.worker.start()
            return receipt

    def _follow(self, request, receipt):
        """Poll status and monitor exactly as the agent would, recording each transition."""
        actions, observations = self.actions, self.observations
        deadline = time.monotonic()+self.action_deadline
        phase, safe = None, object()
        while time.monotonic() < deadline:
            try:
                outcome = actions.status(receipt.action_id)
            except Exception as exc:
                with self.lock:
                    self._record("error", "status() failed", error=type(exc).__name__, detail=str(exc)[:400])
                self._stop_quietly("status became unreadable")
                break
            with self.lock:
                self.outcome = outcome
                if outcome.phase != phase:
                    phase = outcome.phase
                    self._record("action", f"{request.step.operation} phase {outcome.phase}",
                                 status=outcome.status, motion=outcome.motion,
                                 effects_started=outcome.effects_started,
                                 holding=_jsonable(asdict(outcome.holding)) if outcome.holding else None)
            if outcome.status != "running":
                with self.lock:
                    self._record("action", f"{request.step.operation} {outcome.status}",
                                 outcome=_jsonable(asdict(outcome)))
                break
            try:
                report = observations.monitor(request, outcome)
            except Exception as exc:
                with self.lock:
                    self._record("error", "monitor() failed", error=type(exc).__name__, detail=str(exc)[:400])
                time.sleep(POLL_S)
                continue
            with self.lock:
                self.monitoring = report
                if report.safe is not safe:
                    safe = report.safe
                    self._record("monitor", f"monitor: safe={report.safe}", phase=report.phase,
                                 reason=report.reason,
                                 revision=report.snapshot.revision if report.snapshot else None)
                if report.snapshot is not None:
                    self._store(report.snapshot)
            time.sleep(POLL_S)
        else:
            with self.lock:
                self._record("error", "action deadline exceeded without a terminal status",
                             request_id=request.request_id)
            self._stop_quietly("action deadline exceeded")
        # Refresh before clearing `active`. The other order lets the page report idle while still
        # showing the snapshot from before the motion, which is the one moment it most misleads.
        try:
            self.observe()
        except ValueError:
            pass
        with self.lock:
            self.active = None

    def _stop_quietly(self, why):
        """A motion nobody can read the status of is one that has to be stopped, not left running."""
        try:
            reply = self.actions.stop()
        except Exception as exc:
            with self.lock:
                self._record("error", f"stop after {why} failed", error=type(exc).__name__)
            return
        with self.lock:
            self._record("action", f"stop() requested: {why}",
                         acknowledged=reply.acknowledged, stopped=reply.stopped)

    def cancel(self):
        actions = self._motion()
        with self.lock:
            active = self.active
        try:
            reply = (actions.cancel(active["receipt"].action_id) if active else actions.stop())
        except Exception as exc:
            self._fail("cancel failed", exc)
        with self.lock:
            self._record("action", "cancel()", acknowledged=reply.acknowledged, stopped=reply.stopped,
                         note="acknowledgement is not proof that motion stopped or a load is safe")
        return reply

    def stop(self):
        """Never gated by the arm switch: this is the control that has to work while moving.

        With nothing armed there are no writers to quiet, and saying so beats a reassuring noop.
        """
        with self.lock:
            actions = self.actions
        if actions is None:
            with self.lock:
                self._record("action", "stop(): this panel owns no motors",
                             note="nothing was armed here, so nothing here can be stopped")
            return CancellationReceipt(None, False)
        try:
            reply = actions.stop()
        except Exception as exc:
            self._fail("stop failed", exc)
        executor = self._read_executor()
        with self.lock:
            self._record("action", "stop()", acknowledged=reply.acknowledged, stopped=reply.stopped,
                         executor=_jsonable(asdict(executor)) if executor else None)
        return reply

    # ---- operator settings ---------------------------------------------------------------

    def arm(self, armed):
        """Arming opens the hardware writers. Disarming refuses new motions but keeps them.

        Closing the action provider would release `arm_*.ctrl`, and the arm daemon cuts torque the
        moment its writer disappears — a held box would drop. Disarm is a policy switch, not a
        teardown.
        """
        if not armed:
            with self.lock:
                if self.active is not None:
                    raise ValueError("stop the running action before disarming")
                self.armed = False
                self._record("session", "actions disarmed",
                             note="the hardware writers stay open; releasing them would drop a held box")
                return False
        if self.open_actions is None:
            raise ValueError("no action provider is configured for this panel")
        with self.lock:
            if self.closed:
                raise ValueError("the panel is shutting down")
            already = self.actions is not None
        if not already:
            self._perception()      # geometry and possession evidence come from the sensing half
            try:
                provider, close = self.open_actions(self.observations)
                capabilities = _jsonable(asdict(provider.capabilities()))
            except Exception as exc:
                self._fail("the action provider could not open the hardware writers", exc)
            with self.lock:
                self.actions = provider
                self.capabilities["actions"] = capabilities
                if close is not None:
                    self.closers.append(close)
                self._record("session", "action provider opened; this panel now owns the motors",
                             **capabilities)
        with self.lock:
            self.armed = True
            self._record("session", "actions armed")
            return True

    def set_requirements(self, requirements, source):
        with self.lock:
            if self.active is not None:
                raise ValueError("an action is running against the current requirements")
            self.requirements = requirements
            self.requirements_source = source
            self.sites, self.site = (), None
            self._record("session", f"requirements from {source}",
                         requirements=_jsonable(asdict(requirements)))
            return requirements

    def image(self, view):
        with self.lock:
            return self.images.get(view)

    def state(self):
        with self.lock:
            now = self.clock()
            return {"revision": self.revision, "kind": self.kind, "armed": self.armed,
                    "perception_open": self.observations is not None,
                    "actions_open": self.actions is not None,
                    "actions_available": self.open_actions is not None,
                    "now": now, "job_id": self.job_id, "error": self.error,
                    "perception_url": self.perception_url,
                    "voxel_size": list(self.voxel_size),
                    "operations": list(OPERATIONS), "arguments": _jsonable(ARGUMENTS),
                    "motion_operations": sorted(MOTION_OPS),
                    "capabilities": dict(self.capabilities),
                    "requirements": dict(_jsonable(asdict(self.requirements)),
                                         source=self.requirements_source,
                                         dimensions=list(self.requirements.dimensions)),
                    "snapshot": snapshot_to_dict(self.snapshot, now) if self.snapshot else None,
                    "executor": _jsonable(asdict(self.executor)) if self.executor else None,
                    "sites": [_jsonable(asdict(site)) for site in self.sites],
                    "site": _jsonable(asdict(self.site)) if self.site else None,
                    "active": ({"operation": self.active["request"].step.operation,
                                "request_id": self.active["request"].request_id,
                                "action_id": self.active["receipt"].action_id,
                                "step": _jsonable(asdict(self.active["request"].step))}
                               if self.active else None),
                    "outcome": _jsonable(asdict(self.outcome)) if self.outcome else None,
                    "monitor": ({"safe": self.monitoring.safe, "phase": self.monitoring.phase,
                                 "reason": self.monitoring.reason}
                                if self.monitoring else None),
                    "log": list(self.log)}

    def close(self):
        """Transports and sensors only. Provider close must never park or untorque a loaded robot."""
        with self.lock:
            self.closed = True
            closers, self.closers = list(self.closers), []
            self.observations = self.actions = None
            worker = self.worker
        if worker is not None:
            worker.join(timeout=1.0)
        for close in closers:
            try:
                close()
            except Exception:
                pass


# ---- provider sources -------------------------------------------------------------------
# Each returns (open_observations, open_actions). open_observations() -> (provider, close);
# open_actions(observations) -> (provider, close). Both are called at most once, lazily, so
# merely opening the debug screen neither starts a detector nor claims a motor.


def robot_sources(voxel_size, **session_kwargs):
    """The deployed shape: one in-process PerceptionSession, with the arms behind the arm switch."""
    def observations():
        import importlib.util
        if importlib.util.find_spec("bbos") is None:
            # Left to itself this surfaces later as a bare ModuleNotFoundError from inside the
            # poll thread, which says nothing about what is actually wrong.
            raise RuntimeError("this panel is not running on the robot: bbos is unavailable, so "
                               "there are no cameras or arms here to open. Run the panel on the bot "
                               "and forward port 8005")
        from observations import RobotObservations
        provider = RobotObservations(voxel_size, **session_kwargs).start()
        return provider, provider.close

    def actions(observations_provider):
        from actions.provider import build_providers
        pair = build_providers(observations=observations_provider, voxel_size=tuple(voxel_size))
        return pair.actions, pair.close

    return observations, actions


def provider_sources(spec):
    """--debug-provider module:factory. One AgentProviders, handed out a half at a time."""
    pair = {}

    def load():
        if "value" not in pair:
            from agent_adapters import load_providers
            pair["value"] = load_providers(spec)
        return pair["value"]

    return (lambda: (load().observations, load().close)), (lambda _: (load().actions, None))


def perception_sources(url, fresh_s=2.0):
    """--debug-perception URL: a perception server over HTTP, with no motion owner behind it."""
    def observations():
        from agent_adapters import PerceptionObservations
        provider = PerceptionObservations(url, fresh_s=fresh_s).start()
        return provider, provider.close

    return observations, None
