from __future__ import annotations

import importlib
import math
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable

from agent import CapabilityError
from agent_types import (ActionOutcome, ActionProvider, ActionReceipt, BoxObservation,
                         CancellationReceipt, Capabilities, ObservationProvider,
                         ObservationSnapshot, PerceptionCapabilities, vector_valid)


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


class ObservationWorker:
    def __init__(self, session_factory, *, eligibility=None, interval=.005,
                 timeout=4.0, clock=time.time):
        if not all(math.isfinite(v) and v > 0 for v in (interval, timeout)):
            raise ValueError("observation intervals must be finite and positive")
        self.factory, self.eligibility = session_factory, eligibility
        self.interval, self.timeout, self.clock = interval, timeout, clock
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
                scan = session.poll()
                if scan is not None:
                    self._revision += 1
                    latest = normalize_scan(scan, self.clock(), eligibility=self.eligibility,
                                            revision=str(self._revision))
                    with self._condition:
                        self._latest = latest
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
            self._thread.join(timeout=self.timeout)
            if self._thread.is_alive():
                raise TimeoutError("sensor provider did not return from poll; worker could not close")

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
