from __future__ import annotations

from dataclasses import replace

from agent_types import (MOTION_OPS, ActionOutcome, ActionReceipt, BoxObservation,
                         BuildSite, CancellationReceipt, Capabilities, CellObservation,
                         ExecutorState, Holding, ObservationSnapshot, PerceptionCapabilities)


class MockAgentWorld:
    def __init__(self, box_count=8, *, voxel_size=(0.3, 0.3, 0.3), faults=None, no_sites=False):
        self.now = 1000.0
        self.epoch = 0
        self.revision = 0
        self.voxel_size = tuple(voxel_size)
        self.positions = {i: (-1.0-i*.4, (-1)**i*.6, voxel_size[1]/2) for i in range(box_count)}
        self.all_visible = False
        self.no_sites = no_sites
        self.stale = False
        self.requirements = None
        self.selected_site = None
        self.placed = {}
        self.holding = Holding("empty", ts=self.now, source="mock-gripper")
        self.requests = []
        self.cancellations = []
        self.cancel_stops = True
        self.faults = {op: list(sequence) for op, sequence in (faults or {}).items()}
        self.outcomes = {}
        self.receipts = {}
        self.active = None
        self.approached = None
        self.at_build = False

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def capabilities(self):
        return MockCapabilities()

    def state(self):
        return ExecutorState(replace(self.holding, ts=self.now), self.now, ready=self.active is None,
                             motion="running" if self.active else "stopped", active_action=self.active)

    def observe(self, site_id=None):
        self.now += .001
        self.revision += 1
        visible = set(self.positions) if self.all_visible else {min(self.positions, default=-1)}
        boxes = tuple(BoxObservation(mid, self.positions[mid], self.voxel_size, self.now,
                                     current=True, eligible=True)
                      for mid in sorted(visible & self.positions.keys()) if mid not in self.placed.values()
                      and not (self.holding.status == "holding" and self.holding.box_id == mid))
        occupancy = []
        if site_id and self.requirements:
            ext = self.requirements.extents
            for x in range(ext[0]):
                for y in range(ext[1]):
                    for z in range(ext[2]):
                        cell = (x, y, z)
                        mid = self.placed.get(cell)
                        occupancy.append(CellObservation(cell, "occupied" if mid is not None else "empty",
                                                         self.now, mid))
            for cell, mid in self.placed.items():
                if not all(0 <= n < limit for n, limit in zip(cell, ext)):
                    occupancy.append(CellObservation(cell, "occupied", self.now, mid))
        return ObservationSnapshot(str(self.revision), self.now-100 if self.stale else self.now,
                                   self.now, self.epoch, valid=True, pose_valid=True, boxes=boxes,
                                   holding=replace(self.holding, ts=self.now), site_id=site_id,
                                   occupancy=tuple(occupancy), occupancy_complete=bool(site_id),
                                   search_exhausted=self.all_visible)

    def find_build_sites(self, requirements):
        self.requirements = requirements
        if self.no_sites or not self.all_visible:
            return ()
        return (self._site("obstructed", valid=False), self._site("floor-a"))

    def _site(self, site_id, valid=True):
        return BuildSite(site_id, (2.0, -1.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0),
                         (10.0, 10.0, 10.0), self.now, self.epoch,
                         valid=valid, floor_valid=valid, clearance_valid=valid,
                         feasible=valid, cost=1.0 if valid else 0.0)

    def check_build_site(self, site_id, requirements):
        self.requirements = requirements
        if self.no_sites or site_id != "floor-a" or not self.all_visible:
            return None
        self.selected_site = site_id
        return self._site(site_id)

    def submit(self, request):
        if request.request_id in self.receipts:
            return self.receipts[request.request_id]
        if self.active:
            raise RuntimeError("mock received concurrent physical actions")
        self.requests.append(request)
        action_id = f"mock-{len(self.requests)}"
        receipt = ActionReceipt(request.request_id, action_id)
        self.receipts[request.request_id] = receipt
        step = request.step
        sequence = self.faults.get(step.operation, [])
        fault = sequence.pop(0) if sequence else None
        if fault == "running_forever":
            self.active = action_id
            self.outcomes[action_id] = ActionOutcome(request.request_id, action_id, "running", self.now,
                                                     phase="approaching", motion="running")
            return receipt
        status, phase, effects, error = "succeeded", "completed", "yes", None
        if fault in {"pregrasp", "prerelease", "blocked"}:
            status, effects = "failed", "no"
            error = {"pregrasp": "pickup_rejected_before_grasp",
                     "prerelease": "placement_rejected_before_release",
                     "blocked": "blocked_motion"}[fault]
            phase = "before_grasp" if fault == "pregrasp" else "before_release"
        else:
            if step.operation == "look_around":
                self.all_visible = True
            elif step.operation == "approach_box":
                if self.holding.status != "empty" or step.box_id not in self.positions:
                    raise AssertionError("invalid approach")
                self.approached = step.box_id
                self.at_build = False
            elif step.operation == "pickup":
                if self.holding.status != "empty" or self.approached != step.box_id:
                    raise AssertionError("pickup without empty grippers and approach")
                self.holding = Holding("holding", step.box_id, self.now, "mock-gripper")
                self.at_build = False
                phase = "lifted"
            elif step.operation == "move_to_build":
                if self.holding.status != "holding" or self.holding.box_id != step.box_id:
                    raise AssertionError("carry without held box")
                self.at_build = True
            elif step.operation == "place":
                if not self.at_build or self.holding.box_id != step.box_id:
                    raise AssertionError("place without arrival and possession")
                if step.cell in self.placed:
                    raise AssertionError("duplicate placement")
                if step.cell[1] and (step.cell[0], step.cell[1]-1, step.cell[2]) not in self.placed:
                    raise AssertionError("unsupported placement")
                if fault != "bad_placement":
                    self.placed[step.cell] = step.box_id
                self.holding = Holding("empty", ts=self.now, source="mock-gripper")
                self.approached = None
                phase = "released"
            if fault == "unknown_holding":
                self.holding = Holding(ts=self.now, source="mock-gripper")
            if fault == "postrelease":
                status, error = "failed", "post_release_failure"
        self.now += .01
        self.outcomes[action_id] = ActionOutcome(request.request_id, action_id, status, self.now,
                                                 phase, effects, "stopped", error,
                                                 replace(self.holding, ts=self.now))
        if fault == "submit_timeout":
            raise TimeoutError("simulated lost reply after execution")
        return receipt

    def status(self, action_id):
        return replace(self.outcomes[action_id], ts=self.now)

    def cancel(self, action_id):
        self.cancellations.append(action_id)
        if self.cancel_stops:
            self.active = None
            self.outcomes[action_id] = replace(self.outcomes[action_id], status="cancelled",
                                               motion="stopped", ts=self.now,
                                               holding=replace(self.holding, ts=self.now))
        return CancellationReceipt(action_id, True, self.cancel_stops)


class MockCapabilities(Capabilities):
    def __init__(self):
        super().__init__(operations=MOTION_OPS, status=True, cancellation=True,
                         idempotency=True, possession=True, carrying=True,
                         max_height=10.0, max_box_size=(1.0, 1.0, 1.0), action_timeout=.05)

    inventory = True
    sites = True
    occupancy = True
