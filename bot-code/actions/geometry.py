"""World-frame targets resolved into the live base frame, against a fresh observation.

The action provider reasons in motor turns and base-frame metres; the agent hands it
session-world geometry. This is the seam between them, and it is deliberately the only
place that conversion happens.

Its whole job is to refuse. A pose that has moved on, an epoch that has rolled, evidence
that has aged out: each one turns a confident move into a confident move to the wrong
place. So every resolve re-reads the snapshot at the moment of the effect rather than
trusting anything carried in the request, and raises instead of approximating.
"""
from __future__ import annotations

import math


class StaleGeometry(RuntimeError):
    """The live pose cannot authorize this request's geometry."""


def world_to_base(point, base_position, base_yaw):
    """Session-world xyz -> base-frame xyz. Planar: the base frame is assumed level.

    Inverse of the rotate-then-translate that normalize_perception applies.
    """
    dx = point[0] - base_position[0]
    dy = point[1] - base_position[1]
    c, s = math.cos(-base_yaw), math.sin(-base_yaw)
    return (c * dx - s * dy, s * dx + c * dy, point[2] - base_position[2])


class Geometry:
    """Resolves an ActionRequest's target against a freshly observed pose.

    ``carry_height`` reports the current height of the cradle in the base frame, so a
    descent can be measured rather than assumed. ``max_age`` bounds how old the
    supporting observation may be at the moment the effect starts.
    """

    def __init__(self, observations, carry_height, max_age=2.0, clock=None):
        import time
        self.observations = observations
        self.carry_height = carry_height
        self.max_age = max_age
        self.clock = clock or time.time

    def _fresh_snapshot(self, request):
        snapshot = self.observations.observe(request.step.site_id)
        if snapshot is None or not snapshot.valid or not snapshot.pose_valid:
            raise StaleGeometry("perception pose is invalid; geometry cannot authorize motion")
        if (snapshot.epoch, snapshot.frame_id) != (request.epoch, request.frame_id):
            raise StaleGeometry(
                f"geometry spans epochs: request {(request.epoch, request.frame_id)} "
                f"vs live {(snapshot.epoch, snapshot.frame_id)}")
        age = self.clock() - snapshot.captured_at
        if not -0.05 <= age <= self.max_age:
            raise StaleGeometry(f"supporting observation is {age:.2f}s old")
        if snapshot.base_position is None or snapshot.base_yaw is None:
            raise StaleGeometry("snapshot carries no base pose")
        return snapshot

    def cell_in_base(self, request, snapshot=None):
        """Base-frame centre of the request's target cell, right now."""
        snapshot = snapshot or self._fresh_snapshot(request)
        if request.site is None or request.step.cell is None:
            raise StaleGeometry("request carries no site or cell to resolve")
        voxel = request.requirements.voxel_size
        world = request.site.cell_center(request.step.cell, voxel)
        return world_to_base(world, snapshot.base_position, snapshot.base_yaw)

    def box_in_base(self, request, snapshot=None):
        """Base-frame position of the request's target box, as currently observed.

        The carried box's remembered pre-grasp position is never used: the box must still be
        current in this snapshot, or there is nothing to approach.
        """
        snapshot = snapshot or self._fresh_snapshot(request)
        if request.step.box_id is None:
            raise StaleGeometry("request carries no box to resolve")
        for box in snapshot.boxes:
            if box.id == request.step.box_id:
                if not box.current:
                    raise StaleGeometry(f"box {box.id} is remembered, not currently observed")
                return world_to_base(box.position, snapshot.base_position, snapshot.base_yaw)
        raise StaleGeometry(f"box {request.step.box_id} is not in the current snapshot")

    def confirm_graspable(self, request):
        """Re-check the box immediately before closing on it, against the higher bar.

        Selection is reversible and grasping is not, so this asks again with evidence gathered
        after the approach -- from closer, and after the robot has stopped moving -- rather than
        trusting the score that justified setting off.
        """
        from observations import SCORE_PICKUP
        snapshot = self._fresh_snapshot(request)
        world = snapshot.world_model
        for item in world.get("objects", ()) or ():
            if item.get("id") != request.step.box_id:
                continue
            score = item.get("score")
            if not isinstance(score, (int, float)) or score < SCORE_PICKUP:
                raise StaleGeometry(
                    f"box {request.step.box_id} is only {score} confident at the grasp, "
                    f"below the {SCORE_PICKUP} required to close on it")
            return {"box_id": request.step.box_id, "score": score}
        raise StaleGeometry(f"box {request.step.box_id} is not visible at the grasp")

    def approach_box(self, request):
        """A target callable the drive loop re-reads every cycle, never a remembered pose."""
        self._fresh_snapshot(request)       # refuse up front if the request cannot be resolved at all
        return {"target_fn": lambda: self.box_in_base(request), "loaded": False}

    def move_to_build(self, request):
        self._fresh_snapshot(request)
        return {"target_fn": lambda: self.cell_in_base(request), "loaded": True}

    def place(self, request):
        """How far the cradled box must descend, measured rather than assumed.

        The target is the cell's bottom face; the box's own bottom is half a voxel below
        the cradle. A negative drop means the target is above the load, which is a
        reachability failure, not something to silently clamp to zero.
        """
        snapshot = self._fresh_snapshot(request)
        target = self.cell_in_base(request, snapshot)
        voxel = request.requirements.voxel_size
        cell_bottom_z = target[2] - voxel[1] / 2.0
        drop = self.carry_height() - cell_bottom_z
        if not math.isfinite(drop):
            raise StaleGeometry("carry height is not measurable")
        if drop < 0:
            raise StaleGeometry(f"target cell sits {-drop:.3f}m above the load; not reachable by lowering")
        return {"z_drop": drop, "target": target}
