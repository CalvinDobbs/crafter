from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import asdict, replace

from agent_backend import validate_step
from agent_types import (INTERFACE_VERSION, MAX_SCENE_IMAGES, MOTION_OPS, ActionReceipt,
                         ActionRequest, AgentConfig, BuildRequirements, BuildState, Holding,
                         JobSpec, MotionObservation, ObservationSnapshot, RunResult, SceneImage,
                         Step, cell_valid, vector_valid)


class BusyError(RuntimeError):
    pass


class CapabilityError(RuntimeError):
    pass


class EvidenceError(RuntimeError):
    pass


class JobCancelled(RuntimeError):
    pass


def validate_job(structure, job_id, config):
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("job_id must be a nonempty string")
    originals = tuple((b.x, b.y, b.z) for b in structure.blocks)
    if not 0 < len(originals) <= config.max_blocks:
        raise ValueError("structure is empty or exceeds the block limit")
    if any(not cell_valid(c) or c[1] < 0 for c in originals):
        raise ValueError("cells require integer coordinates and nonnegative layers")
    if len(set(originals)) != len(originals):
        raise ValueError("duplicate target cells")
    dx, dz = min(c[0] for c in originals), min(c[2] for c in originals)
    pairs = sorted((((x-dx, y, z-dz), (x, y, z)) for x, y, z in originals),
                   key=lambda pair: (pair[0][1], pair[0][0], pair[0][2]))
    cells = tuple(pair[0] for pair in pairs)
    extents = tuple(max(c[i] for c in cells)+1 for i in range(3))
    if max(extents) > config.max_extent:
        raise ValueError("structure exceeds the extent limit")
    if any(y > 0 and (x, y-1, z) not in cells for x, y, z in cells):
        raise ValueError("unsupported schematic: overhangs are not supported")
    return JobSpec(job_id, BuildRequirements(cells, tuple(config.voxel_size), extents),
                   tuple(pair[1] for pair in pairs))


class JobManager:
    def __init__(self, config=None):
        self.config = config or AgentConfig()
        self._lock = threading.Lock()
        self.active = None

    def submit(self, structure, job_id=None):
        with self._lock:
            if self.active is not None:
                raise BusyError(f"busy: {self.active.job.job_id}")
            job = validate_job(structure, job_id or uuid.uuid4().hex, self.config)
            self.active = BuildState(job)
            return self.active

    def release(self, state):
        with self._lock:
            if self.active is not state:
                raise BusyError("job ownership changed")
            if state.phase not in {"COMPLETED", "FAILED", "CANCELLED"}:
                raise BusyError("job has unresolved obligations")
            self.active = None


def preflight(actions, observations, requirements):
    a, p = actions.capabilities(), observations.capabilities()
    missing = sorted(MOTION_OPS - a.operations)
    missing += [name for name in ("status", "cancellation", "carrying", "idempotency", "emergency_stop")
                if not getattr(a, name, False)]
    missing += [name for name in ("inventory", "sites", "occupancy", "monitoring", "images")
                if not getattr(p, name, False)]
    if not (a.possession or p.possession):
        missing.append("possession evidence")
    if a.api_version != INTERFACE_VERSION or getattr(p, "api_version", None) != INTERFACE_VERSION:
        missing.append(f"action/perception protocol v{INTERFACE_VERSION}")
    for provider, methods in ((actions, ("submit", "lookup", "status", "state", "cancel", "stop")),
                              (observations, ("observe", "monitor", "find_build_sites", "check_build_site"))):
        missing += [name + " method" for name in methods if not callable(getattr(provider, name, None))]
    if missing:
        raise CapabilityError("missing capabilities: " + ", ".join(missing))
    if (not vector_valid(a.max_box_size) or not math.isfinite(a.max_height)
            or any(v > limit for v, limit in zip(requirements.voxel_size, a.max_box_size))
            or requirements.dimensions[1] > a.max_height):
        raise CapabilityError("structure or box dimensions exceed action handling limits")
    if not math.isfinite(a.action_timeout) or a.action_timeout <= 0:
        raise CapabilityError("action provider must supply a finite positive timeout")
    return a


class Agent:
    def __init__(self, actions, observations, *, config=None, reasoner=None,
                 backend="auto", manager=None, clock=time.time, sleep=time.sleep,
                 monotonic=None, event_sink=None):
        self.config = config or AgentConfig()
        self.manager = manager or JobManager(self.config)
        if self.manager.config != self.config:
            raise ValueError("manager and agent configuration must match")
        if backend not in {"auto", "llm", "deterministic"}:
            raise ValueError("unknown reasoning backend")
        self.actions, self.observations = actions, observations
        self.reasoner, self.backend = reasoner, backend
        self.clock, self.sleep = clock, sleep
        self.monotonic = monotonic or (time.monotonic if clock is time.time else clock)
        self.event_sink = event_sink
        self._cancel = threading.Event()
        self._running = threading.Lock()
        self.state = None
        self.capabilities = None

    def _event(self, event, **fields):
        item = {"event": event, **fields}
        self.state.history.append(item)
        del self.state.history[:-self.config.history_limit]
        if self.event_sink:
            try:
                self.event_sink(dict(item))
            except Exception:
                pass

    def _fresh(self, ts):
        return type(ts) in (int, float) and math.isfinite(ts) and -0.05 <= self.clock()-ts <= self.config.fresh_s

    def _holding(self, *evidence, not_before=0.0):
        known = []
        for h in evidence:
            if h is None or not self._fresh(h.ts) or h.ts < not_before or not h.source:
                continue
            if h.status == "empty" and h.box_id is None:
                known.append(h)
            elif h.status == "holding" and type(h.box_id) is int and h.box_id >= 0:
                known.append(h)
        if not known or len({(h.status, h.box_id) for h in known}) != 1:
            return Holding(ts=self.clock(), source="reconciliation")
        return max(known, key=lambda h: h.ts)

    def _snapshot_valid(self, snapshot):
        return (isinstance(snapshot, ObservationSnapshot) and snapshot.valid and snapshot.pose_valid
                and isinstance(snapshot.revision, str) and 0 < len(snapshot.revision) <= 128
                and isinstance(snapshot.frame_id, str) and 0 < len(snapshot.frame_id) <= 128
                and type(snapshot.epoch) is int and snapshot.epoch >= 0
                and self._fresh(snapshot.captured_at) and self._fresh(snapshot.received_at)
                and snapshot.captured_at <= snapshot.received_at+.05
                and vector_valid(snapshot.base_position)
                and type(snapshot.base_yaw) in (int, float) and math.isfinite(snapshot.base_yaw)
                and 0 < len(snapshot.images) <= MAX_SCENE_IMAGES
                and all(isinstance(image, SceneImage) and self._fresh(image.captured_at)
                        and (image.epoch, image.frame_id) == (snapshot.epoch, snapshot.frame_id)
                        for image in snapshot.images))

    def _refresh(self, after=0.0):
        s = self.state
        deadline = self.monotonic()+self.config.observation_timeout
        attempt = 0
        while True:
            try:
                snapshot = self.observations.observe(s.site.id if s.site else None)
                executor = self.actions.state()
                valid = (self._snapshot_valid(snapshot) and snapshot.captured_at > after
                         and executor.ts >= after and self._fresh(executor.ts))
                if valid and after:
                    valid = self._holding(executor.holding, snapshot.holding, not_before=after).status != "unknown"
                    if s.active_request and s.active_request.step.operation == "place":
                        target = next((item for item in snapshot.occupancy if item.cell == s.target_cell), None)
                        valid = valid and target is not None and target.ts > after
            except Exception as exc:
                valid = False
                self._event("observation_error", error=type(exc).__name__)
            if valid:
                break
            attempt += 1
            if self.monotonic() >= deadline or (not after and attempt >= self.config.observation_attempts):
                raise EvidenceError("fresh observations, images and executor state unavailable")
            self.sleep(min(self.config.poll_s, max(0.0, deadline-self.monotonic())))
        previous = s.snapshot
        if previous and (previous.epoch, previous.frame_id) != (snapshot.epoch, snapshot.frame_id):
            if s.confirmed or s.target_box is not None or s.active_request:
                raise EvidenceError("pose epoch/frame changed; re-registration required")
            s.inventory.clear()
            s.site = None
            self._event("frame_reset")
        s.snapshot, s.executor = snapshot, executor
        s.holding = self._holding(executor.holding, snapshot.holding, not_before=after)
        if s.holding.status == "unknown":
            raise EvidenceError("possession is unknown or contradictory")
        if s.target_box is None and s.holding.status != "empty":
            raise EvidenceError("unassigned held box requires reconciliation")
        if s.holding.status == "holding" and s.holding.box_id != s.target_box:
            raise EvidenceError("held identity does not match the reserved box")
        grasping = s.active_request is not None and s.active_request.step.operation == "pickup"
        if s.stage in {"approach_box", "pickup"} and s.holding.status != "empty" and not grasping:
            raise EvidenceError("unexpected possession before pickup")
        releasing = s.active_request is not None and s.active_request.step.operation == "place"
        if s.stage in {"move_to_build", "place"} and s.holding.status != "holding" and not releasing:
            raise EvidenceError("reserved carried box is no longer confirmed held")
        inventory = {mid: replace(box, current=False) for mid, box in s.inventory.items()}
        counts = {}
        for box in snapshot.boxes:
            if type(box.id) is int and box.id >= 0:
                counts[box.id] = counts.get(box.id, 0)+1
        for box in snapshot.boxes:
            if (type(box.id) is not int or box.id < 0 or counts[box.id] != 1
                    or not box.valid or not vector_valid(box.position)):
                if type(box.id) is int:
                    inventory.pop(box.id, None)
                continue
            inventory[box.id] = box
        s.inventory = inventory
        if s.site:
            occupancy = self._occupancy()
            for cell, mid in s.confirmed.items():
                observed = occupancy.get(cell)
                if observed is None or observed.status != "occupied" or observed.box_id != mid:
                    raise EvidenceError("confirmed construction disagrees with observation")
        return snapshot

    def _occupancy(self):
        s = self.state
        snapshot = s.snapshot
        if snapshot.site_id != s.site.id or not snapshot.occupancy_complete:
            raise EvidenceError("complete occupancy for the selected site is unavailable")
        result = {}
        for item in snapshot.occupancy:
            if (not cell_valid(item.cell) or item.cell in result or not self._fresh(item.ts)
                    or item.status not in {"empty", "occupied", "unknown"}
                    or (item.box_id is not None and (type(item.box_id) is not int or item.box_id < 0))
                    or (item.status == "empty" and item.box_id is not None)):
                raise EvidenceError("invalid, duplicate or stale occupancy evidence")
            result[item.cell] = item
        ext = s.job.requirements.extents
        envelope = {(x, y, z) for x in range(ext[0]) for y in range(ext[1]) for z in range(ext[2])}
        if not envelope.issubset(result) or any(result[c].status == "unknown" for c in envelope):
            raise EvidenceError("target envelope contains unobserved cells")
        allowed = set(s.confirmed)
        if s.stage == "place" and s.active_request:
            allowed.add(s.target_cell)
        if any(o.status == "occupied" and c not in allowed for c, o in result.items()):
            raise EvidenceError("unexpected object obstructs the selected build site")
        return result

    def _site_valid(self, site):
        s = self.state
        if site is None or not isinstance(site.id, str) or not site.id:
            return False
        if not all((site.valid, site.floor_valid, site.clearance_valid, site.feasible)):
            return False
        if (not self._fresh(site.ts) or (site.epoch, site.frame_id) != (s.snapshot.epoch, s.snapshot.frame_id)
                or not all(vector_valid(v) for v in (site.origin, site.col, site.row, site.dimensions))
                or not math.isfinite(site.cost) or site.cost < 0):
            return False
        dot = sum(a*b for a, b in zip(site.col, site.row))
        normal_z = site.col[0]*site.row[1]-site.col[1]*site.row[0]
        axes_ok = abs(dot) < .01 and normal_z > .99 and all(
            abs(sum(v*v for v in axis)-1) < .01 for axis in (site.col, site.row))
        return axes_ok and all(a >= b for a, b in zip(site.dimensions, s.job.requirements.dimensions))

    def _usable(self, box, current=False):
        return (box.valid and box.eligible is True and vector_valid(box.size)
                and all(abs(a-b) <= .01*b for a, b in zip(box.size, self.config.voxel_size))
                and (not current or (box.current and self._fresh(box.last_seen))))

    def _idle(self):
        e = self.state.executor
        return e.ready and e.motion == "stopped" and e.active_action is None

    def choices(self):
        s = self.state
        if not self._idle() or s.active_request:
            raise EvidenceError("executor is not settled or an action is unresolved")
        extra = (Step("observe", "Refresh observations"), Step("stop", "Stop this job"))
        if s.site is None:
            remaining = [b for mid, b in s.inventory.items() if self._usable(b) and mid not in s.quarantined]
            if len(remaining) < len(s.job.requirements.cells):
                s.phase = "INVENTORY"
                primary = self._search("materials")
            else:
                s.phase = "SELECT_SITE"
                sites = self.observations.find_build_sites(s.job.requirements)
                ids = [site.id for site in sites]
                s.candidate_sites = tuple(site for site in sorted(sites, key=lambda site: (site.cost, site.id))
                                          if ids.count(site.id) == 1 and self._site_valid(site))[:32]
                primary = tuple(Step("select_site", "Use a validated feasible footprint", site_id=site.id)
                                for site in s.candidate_sites)
                if not primary:
                    primary = self._search("sites")
            return primary[:32] + extra
        checked = self.observations.check_build_site(s.site.id, s.job.requirements)
        if not self._site_valid(checked) or checked.id != s.site.id:
            raise EvidenceError("selected site is stale, obstructed or infeasible")
        if (checked.origin, checked.col, checked.row) != (s.site.origin, s.site.col, s.site.row):
            raise EvidenceError("selected site registration changed")
        s.site = checked
        occupancy = self._occupancy()
        if len(s.confirmed) == len(s.job.requirements.cells):
            s.phase = "FINAL_VERIFY"
            return (Step("done", "All target cells are verified"),) + extra
        s.phase = "BUILD"
        if s.target_box is not None:
            if s.stage in {"approach_box", "pickup"}:
                box = s.inventory.get(s.target_box)
                if box is None or not self._usable(box, current=True):
                    self._event("target_lost", box_id=s.target_box)
                    s.target_box, s.target_cell, s.stage = None, None, "approach_box"
                elif s.stage == "pickup":
                    return (Step("pickup", "Pick the approached box", box_id=s.target_box),) + extra
                else:
                    return (Step("approach_box", "Retry approach to fresh material", box_id=s.target_box,
                                 site_id=s.site.id, cell=s.target_cell),) + extra
            else:
                return (Step(s.stage, "Carry or place the reserved box", box_id=s.target_box,
                             site_id=s.site.id, cell=s.target_cell),) + extra
        used = set(s.confirmed.values()) | s.quarantined
        boxes = [b for mid, b in s.inventory.items() if mid not in used and self._usable(b, current=True)]
        boxes.sort(key=lambda b: (sum(v*v for v in b.position[:2]), b.id))
        cells = [c for c in s.job.requirements.cells if c not in s.confirmed
                 and occupancy[c].status == "empty"
                 and (c[1] == 0 or (c[0], c[1]-1, c[2]) in s.confirmed)]
        if not cells:
            raise EvidenceError("no supported empty target cell")
        primary = tuple(Step("approach_box", "Fetch material for a supported cell", box_id=b.id,
                             site_id=s.site.id, cell=c) for c in cells for b in boxes)
        return (primary[:32] or self._search("materials")) + extra

    def _search(self, kind):
        s = self.state
        if s.searches >= self.config.max_searches or s.snapshot.search_exhausted:
            return (Step("stop", f"Insufficient observed {kind}; search exhausted"),)
        return (Step("look_around", f"Search for {kind}", search=kind),)

    def _context(self, choices):
        s = self.state
        return {"job_id": s.job.job_id, "phase": s.phase,
                "goal": asdict(s.job.requirements), "holding": asdict(s.holding),
                "inventory": [asdict(b) for b in s.inventory.values()],
                "confirmed": [{"cell": c, "box_id": mid} for c, mid in sorted(s.confirmed.items())],
                "site": asdict(s.site) if s.site else None,
                "candidate_sites": [asdict(site) for site in s.candidate_sites
                                    if self._site_valid(site)],
                "enabled_operations": sorted(self.capabilities.operations),
                "observation_revision": s.snapshot.revision,
                "observation_time": s.snapshot.captured_at,
                "frame_id": s.snapshot.frame_id, "epoch": s.snapshot.epoch,
                "base_position": s.snapshot.base_position, "base_yaw": s.snapshot.base_yaw,
                "images": [asdict(image) for image in s.snapshot.images],
                "occupancy": [asdict(item) for item in s.snapshot.occupancy
                              if item.cell in s.job.requirements.cells or item.status != "empty"],
                "warnings": list(s.snapshot.warnings),
                "last_outcome": asdict(s.last_outcome) if s.last_outcome else None,
                "history": list(s.history), "allowed_choices": [asdict(c) for c in choices]}

    def _decide(self, choices):
        if self.backend == "deterministic" or (self.backend == "auto" and self.reasoner is None):
            return choices[0]
        if self.reasoner is None:
            raise CapabilityError("llm backend requires a configured reasoner")
        for _ in range(2):
            try:
                step = validate_step(self.reasoner.decide(self._context(choices), choices))
                if step.key() not in {c.key() for c in choices}:
                    raise ValueError("model selected an ineligible decision")
                return step
            except Exception as exc:
                self._event("model_rejected", error=type(exc).__name__)
        if self.backend == "llm":
            raise CapabilityError("LLM failed to produce a valid decision")
        self._event("model_fallback", backend="deterministic")
        return choices[0]

    def cancel(self):
        self._cancel.set()

    def _cancel_active(self):
        s = self.state
        if (not isinstance(s.receipt, ActionReceipt) or s.receipt.request_id != s.active_request.request_id
                or not s.receipt.action_id):
            s.receipt = None
            try:
                receipt = self.actions.lookup(s.active_request.request_id)
                if (isinstance(receipt, ActionReceipt) and receipt.request_id == s.active_request.request_id
                        and receipt.action_id):
                    s.receipt = receipt
            except Exception as exc:
                self._event("lookup_error", error=type(exc).__name__)
        try:
            reply = self.actions.cancel(s.receipt.action_id) if s.receipt else self.actions.stop()
            self._event("cancel_requested", acknowledged=reply.acknowledged, stopped=reply.stopped)
            if reply.acknowledged is not True:
                reply = self.actions.stop()
                self._event("stop_requested", acknowledged=reply.acknowledged, stopped=reply.stopped)
        except Exception as exc:
            self._event("cancel_error", error=type(exc).__name__)
            try:
                reply = self.actions.stop()
                self._event("stop_requested", acknowledged=reply.acknowledged, stopped=reply.stopped)
            except Exception as stop_error:
                self._event("stop_error", error=type(stop_error).__name__)

    def _resolve_cancellation(self, release=True):
        s = self.state
        self._cancel_active()
        deadline = self.monotonic()+self.capabilities.action_timeout
        while self.monotonic() < deadline:
            try:
                executor = self.actions.state()
                if s.receipt is None:
                    s.executor = executor
                    s.holding = self._holding(executor.holding, not_before=s.active_request.submitted_at)
                    return False
                outcome = self.actions.status(s.receipt.action_id)
                try:
                    snapshot = self.observations.observe(s.site.id if s.site else None)
                    observed_holding = snapshot.holding if snapshot.valid and self._fresh(snapshot.captured_at) else None
                except Exception:
                    observed_holding = None
                holding = self._holding(executor.holding, observed_holding, outcome.holding,
                                        not_before=s.active_request.submitted_at)
                stopped = (outcome.action_id == s.receipt.action_id
                           and outcome.request_id == s.active_request.request_id
                           and outcome.status in {"cancelled", "failed", "succeeded"}
                           and outcome.motion == executor.motion == "stopped"
                           and executor.active_action is None and self._fresh(executor.ts)
                           and self._fresh(outcome.observed_at)
                           and s.active_request.submitted_at <= outcome.ts <= outcome.observed_at+.05)
                if stopped:
                    s.last_outcome, s.executor, s.holding = outcome, executor, holding
                    if holding.status == "empty" and release:
                        s.active_request, s.receipt = None, None
                        return True
                    return False
            except Exception as exc:
                self._event("cancel_reconciliation_error", error=type(exc).__name__)
            self.sleep(self.config.poll_s)
        return False

    def _monitor_action(self, outcome):
        s = self.state
        report = self.observations.monitor(s.active_request, outcome)
        if (not isinstance(report, MotionObservation) or report.request_id != s.active_request.request_id
                or report.action_id != s.receipt.action_id or report.phase != outcome.phase
                or not self._snapshot_valid(report.snapshot)
                or (report.snapshot.epoch, report.snapshot.frame_id)
                != (s.active_request.epoch, s.active_request.frame_id)):
            raise EvidenceError("action monitoring identity, images or localization is stale/invalid")
        previous = s.last_monitor
        s.last_monitor = report
        if previous is None or (previous.action_id, previous.phase, previous.safe) != (report.action_id, report.phase, report.safe):
            self._event("action_observed", operation=s.active_request.step.operation, phase=report.phase,
                        observation_revision=report.snapshot.revision, safe=report.safe)
        if report.safe is not True:
            raise EvidenceError("action monitor requires a stop: " + str(report.reason)[:256])
        executor = self.actions.state()
        if not self._fresh(executor.ts) or executor.active_action not in {None, s.receipt.action_id}:
            raise EvidenceError("executor state is stale or another action is active during monitoring")
        holding = self._holding(executor.holding, report.snapshot.holding)
        op = s.active_request.step.operation
        if holding.status == "unknown":
            raise EvidenceError("possession became unknown during motion")
        if op in {"approach_box", "look_around"} and holding.status != "empty":
            raise EvidenceError("unexpected possession during empty-gripper motion")
        if op == "move_to_build" and (holding.status != "holding" or holding.box_id != s.target_box):
            raise EvidenceError("carried load was lost or changed during motion")
        if holding.status == "holding" and holding.box_id != s.target_box:
            raise EvidenceError("unexpected held identity during motion")

    def _poll(self):
        s = self.state
        deadline = self.monotonic()+self.capabilities.action_timeout
        while self.monotonic() < deadline:
            if self._cancel.is_set():
                if self._resolve_cancellation():
                    raise JobCancelled("cancelled with verified stop and empty grippers")
                raise EvidenceError("cancellation requires stopped/possession reconciliation")
            try:
                result = self.actions.status(s.receipt.action_id)
            except Exception as exc:
                raise EvidenceError(f"action status unavailable ({type(exc).__name__})") from exc
            if (result.request_id != s.active_request.request_id or result.action_id != s.receipt.action_id
                    or not self._fresh(result.observed_at)
                    or not s.active_request.submitted_at <= result.ts <= result.observed_at+.05):
                raise EvidenceError("action result identity or timestamp is invalid")
            if result.status != "running":
                return result
            self._monitor_action(result)
            self.sleep(self.config.poll_s)
        if self._resolve_cancellation():
            raise JobCancelled("action deadline exceeded; cancelled with verified stop and empty grippers")
        raise EvidenceError("action deadline exceeded; stopped state is not established")

    def _execute(self, step):
        s = self.state
        validate_step(step)
        if self._cancel.is_set():
            raise JobCancelled("cancelled before dispatch")
        if not self._fresh(s.snapshot.captured_at):
            raise EvidenceError("observation expired during final validation")
        if s.actions >= self.config.max_actions:
            raise EvidenceError("action budget exhausted")
        if step.operation == "approach_box":
            s.target_box, s.target_cell = step.box_id, step.cell
        box = s.inventory.get(step.box_id) if step.box_id is not None else None
        evidence_times = [s.snapshot.captured_at, s.executor.ts]
        if s.site:
            evidence_times.append(s.site.ts)
        if box and step.operation in {"approach_box", "pickup"}:
            evidence_times.append(box.last_seen)
        request = ActionRequest(uuid.uuid4().hex, s.job.job_id, step, s.snapshot.revision,
                                s.snapshot.epoch, s.snapshot.frame_id, self.clock(),
                                requirements=s.job.requirements, site=s.site, box=box,
                                expires_at=min(evidence_times)+self.config.fresh_s)
        s.active_request = request
        s.actions += 1
        self._event("action_submitted", operation=step.operation, request_id=request.request_id)
        try:
            s.receipt = self.actions.submit(request)
        except Exception as exc:
            raise EvidenceError(f"ambiguous dispatch ({type(exc).__name__}); request will not be replayed") from exc
        if s.receipt.request_id != request.request_id or not s.receipt.action_id:
            raise EvidenceError("invalid action receipt")
        outcome = self._poll()
        s.last_outcome = outcome
        if outcome.status not in {"succeeded", "failed", "cancelled"} or outcome.motion != "stopped":
            raise EvidenceError("action outcome or stopped state is unknown")
        self._refresh(after=outcome.ts)
        if not self._idle():
            raise EvidenceError("executor reports unsettled state after action result")
        holding = self._holding(s.holding, outcome.holding, not_before=request.submitted_at)
        if holding.status == "unknown":
            raise EvidenceError("action and observation possession evidence disagree")
        s.holding = holding
        op = step.operation
        if op == "pickup" and outcome.status == "succeeded":
            if holding.status != "holding" or holding.box_id != s.target_box:
                raise EvidenceError("pickup finished without confirmed possession")
            s.stage = "move_to_build"
        elif op == "place" and holding.status == "empty":
            occupied = self._occupancy().get(s.target_cell)
            if (outcome.phase not in {"released", "retreated", "completed"}
                    or occupied is None or occupied.status != "occupied" or occupied.box_id != s.target_box
                    or occupied.ts <= outcome.ts):
                raise EvidenceError("released box placement is not verified")
            s.confirmed[s.target_cell] = s.target_box
            self._event("placement_confirmed", box_id=s.target_box, cell=s.target_cell)
            s.target_box, s.target_cell, s.stage = None, None, "approach_box"
        elif outcome.status == "succeeded":
            if op == "place":
                raise EvidenceError("place completed but box remains held")
            if op == "approach_box":
                s.stage = "pickup"
            elif op == "move_to_build":
                s.stage = "place"
            elif op == "look_around":
                s.searches += 1
        else:
            safe_empty = (holding.status == "empty" and op in {"approach_box", "pickup", "look_around"}
                          and (outcome.effects_started == "no" or outcome.error_code in {
                              "target_lost_before_pick", "pickup_rejected_before_grasp", "blocked_motion"}))
            safe_held = (holding.status == "holding" and op in {"move_to_build", "place"}
                         and outcome.error_code in {"blocked_motion", "placement_rejected_before_release"})
            if not (safe_empty or safe_held):
                raise EvidenceError("failure is not a known recoverable precondition")
            key = (op, s.target_box, s.target_cell)
            s.retries[key] = s.retries.get(key, 0)+1
            self._event("action_retry", operation=op, count=s.retries[key])
            if op == "look_around":
                s.searches += 1
            if s.retries[key] > self.config.max_retries:
                if safe_held:
                    raise EvidenceError("holding a box after recovery budget exhausted")
                if s.target_box is not None:
                    s.quarantined.add(s.target_box)
                elif op == "look_around":
                    s.searches = self.config.max_searches
                s.target_box, s.target_cell, s.stage = None, None, "approach_box"
        s.active_request, s.receipt = None, None

    def _finish(self, requested, reason):
        s = self.state
        if s.active_request is not None:
            self._resolve_cancellation(release=False)
        try:
            executor = self.actions.state()
            safe = (self._fresh(executor.ts) and executor.motion == "stopped"
                    and executor.active_action is None and self._holding(executor.holding).status == "empty"
                    and s.active_request is None
                    and (s.snapshot is None or s.holding.status == "empty"))
        except Exception:
            safe = False
        s.phase = requested if safe else "NEEDS_OPERATOR"
        s.reason = reason
        self._event("job_finished", status=s.phase, reason=reason)
        result = RunResult(s.job.job_id, s.phase, reason, len(s.confirmed),
                           len(s.job.requirements.cells), s.actions, s.steps, tuple(s.history))
        if safe:
            self.manager.release(s)
        return result

    def run(self, structure=None, *, job_id=None, state=None):
        if not self._running.acquire(blocking=False):
            raise BusyError("agent runner is already active")
        try:
            self.state = state if state is not None else self.manager.submit(structure, job_id)
            if self.manager.active is not self.state:
                raise BusyError("job is not admitted by this manager")
            s = self.state
            self._cancel.clear()
            try:
                self.capabilities = preflight(self.actions, self.observations, s.job.requirements)
                if self.backend == "llm" and self.reasoner is None:
                    raise CapabilityError("llm backend requires a configured reasoner")
                for _ in range(self.config.max_steps):
                    s.steps += 1
                    if self._cancel.is_set():
                        return self._finish("CANCELLED", "cancelled before next action")
                    before = (len(s.inventory), len(s.confirmed), s.site.id if s.site else None, s.stage)
                    self._refresh()
                    choices = self.choices()
                    chosen = self._decide(choices)
                    self._event("decision", operation=chosen.operation, box_id=chosen.box_id,
                                site_id=chosen.site_id, cell=chosen.cell, reason=chosen.reason)
                    if self._cancel.is_set():
                        raise JobCancelled("cancelled during reasoning, before dispatch")
                    revision = (s.snapshot.epoch, s.snapshot.frame_id)
                    self._refresh()
                    if revision != (s.snapshot.epoch, s.snapshot.frame_id):
                        self._event("decision_invalidated", reason="frame changed")
                        continue
                    fresh_choices = self.choices()
                    if chosen.key() not in {c.key() for c in fresh_choices}:
                        self._event("decision_invalidated", reason="preconditions changed")
                    elif chosen.operation == "stop":
                        return self._finish("FAILED", chosen.reason or "reasoner stopped")
                    elif chosen.operation == "done":
                        return self._finish("COMPLETED", "target occupancy verified")
                    elif chosen.operation == "select_site":
                        site = self.observations.check_build_site(chosen.site_id, s.job.requirements)
                        if self._site_valid(site) and site.id == chosen.site_id:
                            s.site = site
                            self._event("site_selected", site_id=site.id)
                    elif chosen.operation in MOTION_OPS:
                        self._execute(chosen)
                    after = (len(s.inventory), len(s.confirmed), s.site.id if s.site else None, s.stage)
                    s.no_progress = s.no_progress+1 if before == after else 0
                    if s.no_progress >= self.config.max_no_progress:
                        return self._finish("FAILED", "no-progress budget exhausted")
                return self._finish("FAILED", "decision budget exhausted")
            except JobCancelled as exc:
                return self._finish("CANCELLED", str(exc))
            except KeyboardInterrupt:
                return self._finish("CANCELLED", "operator interrupted the agent")
            except (CapabilityError, EvidenceError) as exc:
                self._event("blocked", reason=str(exc))
                return self._finish("FAILED", str(exc))
            except Exception as exc:
                self._event("provider_error", error=type(exc).__name__)
                return self._finish("FAILED", f"provider error: {type(exc).__name__}")
        finally:
            self._running.release()
