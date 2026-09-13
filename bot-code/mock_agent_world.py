"""Temporary v2 component simulator: virtual time, phased actions and synthetic images.

This exercises agent contracts, not real perception, collision avoidance or physics.
Replace world.actions/world.observations independently in offline component tests.
"""

from __future__ import annotations

import base64
import math
import struct
import zlib
from dataclasses import replace

from agent_types import (MOTION_OPS, ActionOutcome, ActionReceipt, BoxObservation,
                         BuildSite, CancellationReceipt, Capabilities, CellObservation,
                         ExecutorState, Holding, MotionObservation, ObservationSnapshot,
                         PerceptionCapabilities, SceneImage)

PHASES = {
    "look_around": ("surveying", "settling"),
    "approach_box": ("approaching", "settling"),
    "pickup": ("grasping", "lifting"),
    "move_to_build": ("carrying", "settling"),
    "place": ("lowering", "releasing", "retreating"),
}
FAULTS = frozenset({"pregrasp", "prerelease", "blocked", "unknown_holding", "bad_placement",
                    "postrelease", "submit_timeout", "running_forever", "obstacle", "pose_loss",
                    "stale_observation", "lost_load"})


class MockActions:
    """ActionProvider backed by a MockAgentWorld, with no sensing capabilities mixed in."""

    def __init__(self, world):
        self.world = world

    def capabilities(self):
        return Capabilities(operations=MOTION_OPS, status=True, cancellation=True,
                            idempotency=True, possession=True, carrying=True, emergency_stop=True,
                            max_height=10.0, max_box_size=(1.0, 1.0, 1.0),
                            action_timeout=self.world.action_timeout)

    def submit(self, request):
        return self.world.submit(request)

    def lookup(self, request_id):
        return self.world.lookup(request_id)

    def status(self, action_id):
        return self.world.status(action_id)

    def state(self):
        return self.world.state()

    def cancel(self, action_id):
        return self.world.cancel(action_id)

    def stop(self):
        return self.world.stop()


class MockPerception:
    """Read-only ObservationProvider; motion only occurs through MockActions."""

    def __init__(self, world):
        self.world = world

    def capabilities(self):
        return PerceptionCapabilities(inventory=True, sites=True, occupancy=True,
                                      possession=True, monitoring=True, images=True)

    def observe(self, site_id=None):
        return self.world.observe(site_id)

    def monitor(self, request, outcome):
        return self.world.monitor(request, outcome)

    def find_build_sites(self, requirements):
        return self.world.find_build_sites(requirements)

    def check_build_site(self, site_id, requirements):
        return self.world.check_build_site(site_id, requirements)


class MockAgentWorld:
    def __init__(self, box_count=8, *, voxel_size=(0.3, 0.3, 0.3), faults=None, no_sites=False,
                 action_duration=.04, action_timeout=.5):
        if not all(math.isfinite(v) and v > 0 for v in (action_duration, action_timeout)):
            raise ValueError("mock action durations and deadlines must be finite and positive")
        self.now = 1000.0
        self.epoch = 0
        self.revision = 0
        self.voxel_size = tuple(voxel_size)
        self.positions = {i: (-1.0-i*.4, (-1)**i*.6, voxel_size[1]/2) for i in range(box_count)}
        self.base_position, self.base_yaw = (0.0, 0.0, 0.0), 0.0
        self.pose_valid = True
        self.all_visible = False
        self.no_sites = no_sites
        self.stale = False
        self.requirements = None
        self.selected_site = None
        self.placed = {}
        self.holding = Holding("empty", ts=self.now, source="mock-gripper")
        self.requests = []
        self.monitor_calls = []
        self.cancellations = []
        self.stop_calls = 0
        self.cancel_stops = True
        self.faults = {op: list(sequence) for op, sequence in (faults or {}).items()}
        if any(op not in MOTION_OPS or any(f not in FAULTS and f is not None for f in sequence)
               for op, sequence in self.faults.items()):
            raise ValueError("unknown mock operation or fault")
        self.action_duration, self.action_timeout = action_duration, action_timeout
        self.outcomes, self.receipts, self._pending = {}, {}, {}
        self.active = None
        self.approached = None
        self.at_build = False
        self.actions, self.observations = MockActions(self), MockPerception(self)

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def close(self):
        """There are no transports to close; never change simulated possession here."""

    def state(self):
        phase = self.outcomes[self.active].phase if self.active else "idle"
        return ExecutorState(replace(self.holding, ts=self.now), self.now, ready=self.active is None,
                             motion="running" if self.active else "stopped", active_action=self.active,
                             phase=phase)

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
        captured = self.now-100 if self.stale else self.now
        snapshot = ObservationSnapshot(str(self.revision), captured, self.now, self.epoch,
                                       valid=True, pose_valid=self.pose_valid, boxes=boxes,
                                       holding=replace(self.holding, ts=self.now), site_id=site_id,
                                       occupancy=tuple(occupancy), occupancy_complete=bool(site_id),
                                       search_exhausted=self.all_visible, base_position=self.base_position,
                                       base_yaw=self.base_yaw,
                                       warnings=("SIMULATED scene; not camera or physical verification",))
        return replace(snapshot, images=(self._image(snapshot),))

    def _image(self, snapshot):
        width = height = 128
        pixels = bytearray(bytes((242, 244, 247)) * width * height)
        points = [self.base_position] + [b.position for b in snapshot.boxes]
        if self.requirements and snapshot.site_id:
            site = self._site(snapshot.site_id)
            points += [site.cell_center(c, self.voxel_size) for c in self.requirements.cells]
        left, right = min(p[0] for p in points)-.5, max(p[0] for p in points)+.5
        bottom, top = min(p[1] for p in points)-.5, max(p[1] for p in points)+.5

        def square(position, color, radius):
            x = round((position[0]-left)/(right-left)*(width-1))
            y = round((top-position[1])/(top-bottom)*(height-1))
            for row in range(max(0, y-radius), min(height, y+radius+1)):
                for col in range(max(0, x-radius), min(width, x+radius+1)):
                    offset = (row*width+col)*3
                    pixels[offset:offset+3] = bytes(color)

        if self.requirements and snapshot.site_id:
            for cell in self.requirements.cells:
                square(site.cell_center(cell, self.voxel_size), (170, 175, 185), 4)
            for cell in self.placed:
                square(site.cell_center(cell, self.voxel_size), (45, 110, 215), 3)
        for box in snapshot.boxes:
            square(box.position, (45, 165, 85), 3)
        square(self.base_position, (225, 160, 20), 4)
        if self.holding.status == "holding":
            square(self.base_position, (200, 55, 60), 2)

        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind+data))

        rows = b"".join(b"\x00" + pixels[row*width*3:(row+1)*width*3] for row in range(height))
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
        return SceneImage("overview", "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
                          snapshot.captured_at, self.epoch, simulated=True,
                          description="SIMULATED top-down world, +X right, +Y up. Green: loose boxes; "
                                      "blue: placed; gray: target cells; yellow: robot; red: held load.")

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
            original = next(r for r in self.requests if r.request_id == request.request_id)
            if original != request:
                raise ValueError("request ID reused with a different payload")
            return self.receipts[request.request_id]
        request.validate_admission(self.now)
        if request.epoch != self.epoch or request.frame_id != "world":
            raise ValueError("mock action uses stale localization")
        if self.active:
            raise RuntimeError("mock received concurrent physical actions")
        self.requirements = request.requirements
        self.requests.append(request)
        action_id = f"mock-{len(self.requests)}"
        receipt = ActionReceipt(request.request_id, action_id)
        self.receipts[request.request_id] = receipt
        sequence = self.faults.get(request.step.operation, [])
        fault = sequence.pop(0) if sequence else None
        self._pending[action_id] = {"request": request, "fault": fault, "started": self.now,
                                    "origin": self.base_position, "yaw": self.base_yaw, "effect": False}
        self.active = action_id
        self.outcomes[action_id] = ActionOutcome(request.request_id, action_id, "running", self.now,
                                                 phase=PHASES[request.step.operation][0],
                                                 effects_started="no", motion="running", observed_at=self.now)
        if fault == "submit_timeout":
            raise TimeoutError("simulated lost acknowledgement after admission")
        return receipt

    def lookup(self, request_id):
        return self.receipts.get(request_id)

    def _apply_effect(self, pending):
        if pending["effect"]:
            return
        request, fault = pending["request"], pending["fault"]
        step = request.step
        if step.operation == "pickup":
            if self.holding.status != "empty" or self.approached != step.box_id:
                raise AssertionError("pickup without empty grippers and approach")
            self.holding = Holding("holding", step.box_id, self.now, "mock-gripper")
            self.at_build = False
        elif step.operation == "place":
            if not self.at_build or self.holding.box_id != step.box_id:
                raise AssertionError("place without arrival and possession")
            if step.cell in self.placed:
                raise AssertionError("duplicate placement")
            if step.cell[1] and (step.cell[0], step.cell[1]-1, step.cell[2]) not in self.placed:
                raise AssertionError("unsupported placement")
            if fault != "bad_placement":
                self.placed[step.cell] = step.box_id
                self.positions[step.box_id] = request.site.cell_center(step.cell, self.voxel_size)
            self.holding = Holding("empty", ts=self.now, source="mock-gripper")
            self.approached = None
        if fault == "unknown_holding":
            self.holding = Holding(ts=self.now, source="mock-gripper")
        pending["effect"] = True

    def _advance(self, action_id):
        if self.active != action_id:
            return
        pending = self._pending[action_id]
        request, fault = pending["request"], pending["fault"]
        step = request.step
        progress = min(1.0, (self.now-pending["started"])/self.action_duration)
        if fault == "running_forever":
            progress = min(progress, .25)
        rejected = fault in {"pregrasp", "prerelease", "blocked"}
        phase = PHASES[step.operation][min(int(progress*len(PHASES[step.operation])), len(PHASES[step.operation])-1)]
        outcome = self.outcomes[action_id]
        if phase != outcome.phase:
            outcome = replace(outcome, phase=phase, ts=self.now)
        if not rejected:
            if step.operation == "look_around":
                self.base_yaw = pending["yaw"] + progress*math.pi/4
            elif step.operation in {"approach_box", "move_to_build"}:
                target = request.box.position if step.operation == "approach_box" else request.site.cell_center(
                    step.cell, self.voxel_size)
                offset = .45 if step.operation == "approach_box" else -.6
                destination = (target[0]+offset, target[1], 0.0)
                self.base_position = tuple(a+(b-a)*progress for a, b in zip(pending["origin"], destination))
                self.base_yaw = math.atan2(target[1]-self.base_position[1], target[0]-self.base_position[0])
            if progress >= .5 and step.operation in {"pickup", "place"}:
                self._apply_effect(pending)
        if progress < 1:
            self.outcomes[action_id] = replace(outcome, effects_started="no" if rejected else "yes")
            return
        status, error, effects = "succeeded", None, "yes"
        if rejected:
            status, effects = "failed", "no"
            error = {"pregrasp": "pickup_rejected_before_grasp",
                     "prerelease": "placement_rejected_before_release", "blocked": "blocked_motion"}[fault]
            phase = "before_grasp" if fault == "pregrasp" else "before_release"
        else:
            if step.operation == "look_around":
                self.all_visible = True
            elif step.operation == "approach_box":
                if self.holding.status != "empty" or step.box_id not in self.positions:
                    raise AssertionError("invalid approach")
                self.approached, self.at_build = step.box_id, False
            elif step.operation == "move_to_build":
                if self.holding.status != "holding" or self.holding.box_id != step.box_id:
                    raise AssertionError("carry without held box")
                self.at_build = True
            phase = {"pickup": "lifted", "place": "retreated"}.get(step.operation, "completed")
            if fault == "postrelease":
                status, error = "failed", "post_release_failure"
        self.outcomes[action_id] = ActionOutcome(request.request_id, action_id, status, self.now,
                                                 phase, effects, "stopped", error,
                                                 replace(self.holding, ts=self.now), observed_at=self.now)
        self.active = None

    def status(self, action_id):
        self._advance(action_id)
        return replace(self.outcomes[action_id], observed_at=self.now)

    def monitor(self, request, outcome):
        pending = self._pending[outcome.action_id]
        fault = pending["fault"]
        if fault == "pose_loss":
            self.pose_valid = False
        if fault == "lost_load":
            self.holding = Holding("empty", ts=self.now, source="mock-gripper")
        snapshot = self.observe(request.site.id if request.site else None)
        if fault == "stale_observation":
            snapshot = replace(snapshot, captured_at=self.now-100)
        safe = fault not in {"obstacle", "pose_loss", "stale_observation", "lost_load"}
        if self.holding.status == "unknown":
            safe = None
        self.monitor_calls.append((request.request_id, outcome.action_id, outcome.phase, snapshot.revision, safe))
        return MotionObservation(request.request_id, outcome.action_id, outcome.phase, snapshot, safe,
                                 "" if safe else f"simulated {fault or 'unknown possession'}")

    def cancel(self, action_id):
        self.cancellations.append(action_id)
        outcome = self.outcomes[action_id]
        if self.cancel_stops and self.active == action_id:
            self.active = None
            self.outcomes[action_id] = replace(outcome, status="cancelled", motion="stopped", ts=self.now,
                                               observed_at=self.now, holding=replace(self.holding, ts=self.now))
        return CancellationReceipt(action_id, True, self.active != action_id and self.cancel_stops)

    def stop(self):
        self.stop_calls += 1
        if self.active is not None:
            return self.cancel(self.active)
        return CancellationReceipt(None, True, True)
