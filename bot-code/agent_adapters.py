from __future__ import annotations

import base64
import copy
import importlib
import json
import math
import threading
import time
from dataclasses import dataclass, replace
from http.client import HTTPConnection, HTTPSConnection
from types import SimpleNamespace
from typing import Callable
from urllib.parse import urlsplit

from agent import CapabilityError
from agent_types import (MAX_IMAGE_BYTES, MAX_WORLD_MODEL_BYTES, ActionOutcome, ActionProvider,
                         ActionReceipt, BoxObservation, CancellationReceipt, Capabilities,
                         ObservationProvider, ObservationSnapshot, PerceptionCapabilities,
                         SceneImage, vector_valid)


@dataclass
class AgentProviders:
    """Replaceable component pair. close releases transports/sensors, never a held load."""

    actions: ActionProvider
    observations: ObservationProvider
    close: Callable[[], None] = lambda: None


@dataclass(frozen=True)
class FunctionResult:
    """Motion result only; possession/readiness come from the separate read_state callback."""

    success: bool
    phase: str = "completed"
    error_code: str | None = None
    effects_started: str = "unknown"


class FunctionActions:
    """Adapt request-taking functions; state/stop callbacks must be independently bounded.

    This bridge never invents possession and cannot forcibly interrupt Python or
    hardware. The supplied stop callback must revoke/interrupt the motion routine.
    """

    def __init__(self, functions, read_state, stop, *, max_box_size, max_height,
                 action_timeout=30.0, clock=time.time):
        self.functions, self.read_state, self._stop = dict(functions), read_state, stop
        self.clock = clock
        self._capabilities = Capabilities(
            operations=frozenset(functions), status=True, cancellation=stop is not None,
            idempotency=True, possession=True, carrying="move_to_build" in functions,
            max_box_size=tuple(max_box_size), max_height=max_height, action_timeout=action_timeout,
            emergency_stop=stop is not None)
        self._lock = threading.Lock()
        self._active = None
        self._outcomes, self._receipts, self._requests = {}, {}, {}

    def capabilities(self):
        return self._capabilities

    def submit(self, request):
        with self._lock:
            if request.request_id in self._receipts:
                if request != self._requests[request.request_id]:
                    raise ValueError("request ID reused with a different payload")
                return self._receipts[request.request_id]
            request.validate_admission(self.clock())
            if self._active is not None:
                raise RuntimeError("another action function is still running")
            if request.step.operation not in self.functions:
                raise CapabilityError("action function is unavailable")
            receipt = ActionReceipt(request.request_id, request.request_id)
            self._receipts[request.request_id] = receipt
            self._requests[request.request_id] = request
            self._active = receipt.action_id
            self._outcomes[receipt.action_id] = ActionOutcome(
                request.request_id, receipt.action_id, "running", self.clock(), motion="running",
                observed_at=self.clock())
            threading.Thread(target=self._execute, args=(request,), name="agent-action", daemon=True).start()
            return receipt

    def lookup(self, request_id):
        with self._lock:
            return self._receipts.get(request_id)

    def _execute(self, request):
        try:
            result = self.functions[request.step.operation](request)
            if not isinstance(result, FunctionResult) or type(result.success) is not bool:
                raise TypeError("action functions must return FunctionResult")
            executor = self.read_state()
            outcome = ActionOutcome(
                request.request_id, request.request_id, "succeeded" if result.success else "failed",
                self.clock(), result.phase, result.effects_started, executor.motion,
                result.error_code, executor.holding, observed_at=self.clock())
        except Exception as exc:
            outcome = ActionOutcome(request.request_id, request.request_id, "unknown", self.clock(),
                                    error_code=type(exc).__name__, observed_at=self.clock())
        with self._lock:
            self._outcomes[request.request_id] = outcome
            self._active = None

    def status(self, action_id):
        with self._lock:
            outcome = self._outcomes[action_id]
        if outcome.status == "running":
            executor = self.read_state()
            with self._lock:
                outcome = self._outcomes[action_id]
                if outcome.status == "running":
                    phase = executor.phase if executor.phase != "unknown" else outcome.phase
                    outcome = replace(outcome, phase=phase, holding=executor.holding,
                                      ts=self.clock() if phase != outcome.phase else outcome.ts)
                    self._outcomes[action_id] = outcome
        return replace(outcome, observed_at=self.clock())

    def state(self):
        executor = self.read_state()
        with self._lock:
            if self._active is not None:
                return replace(executor, ready=False, motion="running", active_action=self._active)
        return executor

    def cancel(self, action_id):
        with self._lock:
            outcome = self._outcomes[action_id]
            if self._active not in {None, action_id}:
                return CancellationReceipt(action_id, False)
            if self._active is None and outcome.status in {"succeeded", "failed", "cancelled"} and outcome.motion == "stopped":
                return CancellationReceipt(action_id, True, True)
        return replace(self.stop(), action_id=action_id)

    def stop(self):
        with self._lock:
            action_id = self._active
        if self._stop is None:
            return CancellationReceipt(action_id, False)
        self._stop()
        executor = self.state()
        return CancellationReceipt(action_id, True, executor.motion == "stopped" and executor.active_action is None)


def load_providers(spec):
    if not spec:
        raise CapabilityError(
            "live provider is not configured; supply --provider module:factory implementing "
            "semantic actions, status/cancellation, possession, material eligibility, "
            "candidate build sites and occupancy verification. Use --mock for offline reasoning.")
    module, separator, name = spec.partition(":")
    if not separator or not module or not name.isidentifier():
        raise ValueError("provider must be module:factory")
    providers = getattr(importlib.import_module(module), name)()
    if not isinstance(providers, AgentProviders):
        raise TypeError("provider factory must return AgentProviders(actions, observations, close)")
    return providers


def normalize_skill_result(payload):
    if not isinstance(payload, dict):
        return "unknown"
    if payload.get("ok") is False:
        return "failed"
    if payload.get("ok") is not True:
        return "unknown"
    result = payload.get("result")
    if type(result) is bool:
        return "succeeded" if result else "failed"
    if isinstance(result, (list, tuple)) and len(result) == 2 and type(result[0]) is bool:
        return "succeeded" if result[0] else "failed"
    return "unknown"


def normalize_scan(scan, received_at, *, eligibility=None, revision=None):
    pose = scan.pose
    valid = bool(pose.valid)
    tracks = list(getattr(scan, "tracks", ()))
    if not tracks:
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        for box in list(getattr(scan, "boxes", ())) + list(getattr(scan, "protected", ())) + list(getattr(scan, "unknown", ())):
            x, y, z = box.pos
            tracks.append({"id": box.id, "world": [pose.x+c*x-s*y, pose.y+s*x+c*y, z],
                           "last_seen": scan.ts, "current": True, "size": box.size,
                           "classification": "unknown"})
    boxes = []
    for track in tracks:
        mid, position = track.get("id"), track.get("world")
        if type(mid) is not int or mid < 0 or not vector_valid(position):
            continue
        size = track.get("size")
        if type(size) in (float, int) and math.isfinite(size) and size > 0:
            size = (size, size, size)
        elif vector_valid(size) and min(size) > 0:
            size = tuple(size)
        else:
            size = None
        eligible = eligibility(dict(track)) if eligibility is not None else None
        eligible = eligible if type(eligible) is bool else None
        boxes.append(BoxObservation(mid, tuple(position), size, track.get("last_seen", 0.0),
                                    current=track.get("current") is True, eligible=eligible, valid=valid))
    return ObservationSnapshot(revision or str(scan.ts), scan.ts, received_at, pose.epoch,
                               frame_id="session-world", valid=valid, pose_valid=valid,
                               boxes=tuple(boxes), warnings=tuple(getattr(scan, "warnings", ())),
                               base_position=(float(pose.x), float(pose.y), 0.0), base_yaw=float(pose.yaw))


def normalize_perception(payload, received_at, *, fresh_s=2.0):
    if (not isinstance(payload, dict) or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 2 or payload.get("mock") is not False
            or payload.get("frame") != "base_at_capture" or type(payload.get("stale")) is not bool):
        raise ValueError("expected a live perception /scan schema v2 in base_at_capture")
    pose, ts = payload.get("pose"), payload.get("ts")
    if (not isinstance(pose, dict) or type(pose.get("epoch")) is not int or pose["epoch"] < 0
            or type(pose.get("valid")) is not bool
            or not vector_valid((pose.get("x"), pose.get("y"), pose.get("yaw")))
            or not vector_valid((ts, pose.get("ts"), received_at))):
        raise ValueError("perception requires finite capture times and a versioned pose")
    settings = payload.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("perception settings must be an object")
    sensor_fresh = settings.get("fresh_s", .8)
    anchor_ttl = settings.get("anchor_ttl", 15.0)
    if not vector_valid((fresh_s, sensor_fresh, anchor_ttl)) or min(fresh_s, sensor_fresh, anchor_ttl) <= 0:
        raise ValueError("perception freshness limits must be finite and positive")
    max_age = min(fresh_s, sensor_fresh)

    def fresh(value, age=max_age):
        return type(value) in (float, int) and math.isfinite(value) and -.05 <= received_at-value <= age

    valid = (pose["valid"] and not payload["stale"] and fresh(ts) and fresh(pose["ts"])
             and abs(pose["ts"]-ts) <= .06)
    fields = ("schema_version", "frame", "pose", "build", "anchor_seen", "tracks", "objects",
              "boxes", "protected", "unknown", "surface_cells", "settings", "diagnostics",
              "streams", "telemetry", "detector", "detector_status", "warnings", "map_semantics")
    world = copy.deepcopy({key: payload[key] for key in fields if key in payload})
    world.update(captured_at=ts, received_at=received_at, frame_id="session-world", mock=False,
                 stale=payload["stale"] or not fresh(ts), pose_valid=valid, site_feasibility="unknown")
    world["map_semantics"] = "observed surfaces only; blank, omitted and unobserved cells UNKNOWN, not free; not a navigation map"
    warnings = world.get("warnings", [])
    if not isinstance(warnings, list) or any(not isinstance(w, str) for w in warnings):
        raise ValueError("perception warnings must be strings")
    warnings = [w[:512] for w in warnings[:16]]
    if not valid:
        warnings.append("perception pose or scan is invalid/stale; geometry cannot authorize motion")
    world["warnings"] = warnings
    collections = ("tracks", "objects", "surface_cells", "boxes", "protected", "unknown")
    for name in collections:
        rows = world.setdefault(name, [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"perception {name} must contain objects")
    for name in ("tracks", "objects"):
        for item in world[name]:
            seen = item.get("last_seen")
            item["observed_current"] = not world["stale"] and item.get("current") is True and fresh(seen)
            item["current"] = valid and item["observed_current"]
            item["age_s" if name == "objects" else "age"] = (
                max(0.0, received_at-seen) if type(seen) in (int, float) and math.isfinite(seen) else None)
            item["pick_candidate"] = item.get("pick_candidate") is True and item["current"]
            if name == "objects":
                item["pick_candidate"] = False
                if item.get("pose_epoch") != pose["epoch"]:
                    item.update(current=False, world_position_m=None, position_base_m=None,
                                identity_status="pose_epoch_changed")
    build = world.get("build")
    if build is not None:
        if not isinstance(build, dict):
            raise ValueError("perception build registration must be an object")
        cells = build.pop("cells", [])
        build["cell_count"] = len(cells)
        build["valid"] = valid and build.get("valid") is True and fresh(build.get("ts"), anchor_ttl)
        if type(build.get("ts")) in (int, float):
            build["age"] = max(0.0, received_at-build["ts"])
        if all(vector_valid(build.get(key)) for key in ("origin", "col", "row", "marker")):
            c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
            registered = dict(build, frame_id="session-world")
            for key in ("origin", "col", "row", "marker"):
                x, y, z = build[key]
                dx, dy = (pose["x"], pose["y"]) if key in {"origin", "marker"} else (0.0, 0.0)
                registered[key] = [dx+c*x-s*y, dy+s*x+c*y, z]
            world["build_world"] = registered
    for item in world["surface_cells"]:
        age = item.get("age")
        seen = ts-age if type(age) in (int, float) and math.isfinite(age) and age >= 0 else None
        item.update(last_seen=seen, age=None if seen is None else max(0.0, received_at-seen),
                    current=valid and fresh(seen))
    surfaces = world["surface_cells"]
    heights = [s["z_max"] for s in surfaces if type(s.get("z_max")) in (int, float) and math.isfinite(s["z_max"])]
    positions = [s["world"] for s in surfaces if vector_valid(s.get("world"))]
    world["surface_summary"] = {
        "count": len(surfaces), "current": sum(s["current"] for s in surfaces),
        "max_height": max(heights) if heights else None,
        "bounds_world": [[min(p[i] for p in positions) for i in range(3)],
                         [max(p[i] for p in positions) for i in range(3)]] if positions else None}
    source = SimpleNamespace(ts=ts, pose=SimpleNamespace(**pose), warnings=warnings,
                             tracks=[t for t in world["tracks"] if not t.get("position_kind")])
    snapshot = normalize_scan(source, received_at)
    boxes = tuple(replace(box, valid=valid, current=valid and box.current and fresh(box.last_seen))
                  for box in snapshot.boxes)

    def priority(item):
        position = item.get("world", item.get("world_position_m"))
        distance = math.hypot(position[0]-pose["x"], position[1]-pose["y"]) if vector_valid(position) else math.inf
        if vector_valid(position) and world.get("build_world"):
            origin = world["build_world"]["origin"]
            distance = min(distance, math.hypot(position[0]-origin[0], position[1]-origin[1]))
        return not item.get("current", False), distance

    totals = {name: len(world[name]) for name in collections}
    for name in collections:
        if len(world[name]) > 128:
            world[name] = sorted(world[name], key=priority)[:128]
    while True:
        world["coverage"] = {name: {"total": totals[name], "included": len(world[name]),
                                    "omitted": totals[name]-len(world[name])} for name in collections}
        encoded = json.dumps(world, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) <= MAX_WORLD_MODEL_BYTES:
            break
        largest = max(collections, key=lambda name: len(json.dumps(world[name], separators=(",", ":"))))
        if not world[largest]:
            raise ValueError("perception metadata exceeds the world-model budget")
        world[largest] = sorted(world[largest], key=priority)[:len(world[largest])//2]
    return replace(snapshot, valid=valid, pose_valid=valid, boxes=boxes, world_model_json=encoded)


class ObservationWorker:
    def __init__(self, session_factory, *, eligibility=None, interval=.005,
                 timeout=4.0, clock=time.time, retry_errors=False, close_timeout=None):
        close_timeout = timeout if close_timeout is None else close_timeout
        if not all(math.isfinite(v) and v > 0 for v in (interval, timeout, close_timeout)):
            raise ValueError("observation intervals must be finite and positive")
        self.factory, self.eligibility = session_factory, eligibility
        self.interval, self.timeout, self.clock = interval, timeout, clock
        self.retry_errors, self.close_timeout = retry_errors, close_timeout
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._latest = None
        self._error = None
        self._revision = 0

    def capabilities(self):
        return PerceptionCapabilities(inventory=True)

    def start(self):
        with self._condition:
            if self._stop.is_set():
                raise RuntimeError("observation worker is closed")
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="agent-observations", daemon=True)
                self._thread.start()
        return self

    def _run(self):
        session = None
        try:
            session = self.factory()
            while not self._stop.is_set():
                try:
                    scan = session.poll()
                    if scan is not None:
                        self._revision += 1
                        latest = (replace(scan, revision=str(self._revision)) if isinstance(scan, ObservationSnapshot)
                                  else normalize_scan(scan, self.clock(), eligibility=self.eligibility,
                                                      revision=str(self._revision)))
                        with self._condition:
                            self._latest, self._error = latest, None
                            self._condition.notify_all()
                except Exception as exc:
                    if not self.retry_errors:
                        raise
                    with self._condition:
                        self._error = type(exc).__name__
                        self._condition.notify_all()
                self._stop.wait(self.interval)
        except Exception as exc:
            with self._condition:
                self._error = type(exc).__name__
                self._condition.notify_all()
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception as exc:
                    with self._condition:
                        self._error = type(exc).__name__
                        self._condition.notify_all()

    def observe(self, site_id=None):
        self.start()
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._latest is not None or self._error is not None or self._stop.is_set(),
                timeout=self.timeout)
            if self._error:
                raise RuntimeError(f"observation source failed: {self._error}")
            if self._stop.is_set():
                raise RuntimeError("observation worker is closed")
            if not ready:
                raise TimeoutError("no observation before deadline")
            return self._latest

    def monitor(self, request, outcome):
        raise CapabilityError("inventory-only worker lacks phase-aware action monitoring")

    def find_build_sites(self, requirements):
        raise CapabilityError("perception adapter lacks candidate sites; fixed anchor is not a site finder")

    def check_build_site(self, site_id, requirements):
        raise CapabilityError("perception adapter lacks selected-site validation")

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=self.close_timeout)
            if self._thread.is_alive():
                raise TimeoutError("sensor provider did not return from poll; worker could not close")

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()


class _PerceptionHTTPSource:
    def __init__(self, address, timeout, fresh_s, clock):
        self.address, self.timeout, self.fresh_s, self.clock = address, timeout, fresh_s, clock

    def _get(self, path, limit):
        connection_type = HTTPSConnection if self.address.scheme == "https" else HTTPConnection
        connection = connection_type(self.address.hostname, self.address.port, timeout=self.timeout)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            if path == "/frame?view=rect" and response.status in {204, 503}:
                return None, response.headers
            if response.status != 200:
                raise ConnectionError(f"perception returned HTTP {response.status}")
            if int(response.getheader("Content-Length", "0")) > limit:
                raise ValueError("perception response exceeds its byte budget")
            data = response.read(limit+1)
            if len(data) > limit:
                raise ValueError("perception response exceeds its byte budget")
            return data, response.headers
        finally:
            connection.close()

    def _scan(self):
        data, headers = self._get("/scan", 2*1024*1024)
        if headers.get_content_type() != "application/json":
            raise ValueError("perception scan must be JSON")
        payload = json.loads(data)
        return payload, normalize_perception(payload, self.clock(), fresh_s=self.fresh_s)

    def poll(self):
        before, snapshot = self._scan()
        data, headers = self._get("/frame?view=rect", MAX_IMAGE_BYTES)
        if data is not None:
            after, snapshot = self._scan()
            ts = float(headers.get("X-Frame-Timestamp", "nan"))
            aligned = (before["pose"]["epoch"] == snapshot.epoch
                       and before["ts"]-.001 <= ts <= after["ts"]+.001
                       and -.05 <= self.clock()-ts <= min(self.fresh_s, after.get("settings", {}).get("fresh_s", .8)))
            if aligned and headers.get_content_type() == "image/jpeg":
                image = SceneImage("rect", "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"),
                                   ts, snapshot.epoch, frame_id=snapshot.frame_id,
                                   description="Live rectified head camera with marker overlays; not motion authority.")
                return replace(snapshot, images=(image,))
        return replace(snapshot, warnings=snapshot.warnings + ("live image unavailable, stale or not aligned to the map epoch",))

    def close(self):
        pass


class PerceptionObservations(ObservationWorker):
    def __init__(self, url="http://127.0.0.1:8007", *, interval=.1, timeout=.2, fresh_s=2.0, clock=time.time):
        if not isinstance(url, str) or any(ord(c) <= 32 for c in url):
            raise ValueError("perception URL must be an HTTP(S) origin")
        address = urlsplit(url)
        if (address.scheme not in {"http", "https"} or not address.hostname
                or address.username is not None or address.password is not None
                or address.path not in {"", "/"} or address.query or address.fragment):
            raise ValueError("perception URL must be an HTTP(S) origin without credentials, path, query or fragment")
        if address.port == 0 or not math.isfinite(fresh_s) or fresh_s <= 0:
            raise ValueError("perception port and freshness must be positive")
        super().__init__(lambda: _PerceptionHTTPSource(address, timeout, fresh_s, clock),
                         interval=interval, timeout=min(timeout, .2), clock=clock, retry_errors=True,
                         close_timeout=3*timeout+.5)

    def capabilities(self):
        return PerceptionCapabilities(inventory=True, images=True)
