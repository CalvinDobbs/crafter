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

from agent_types import vector_valid


# How stale a sighting may be and still serve as a heading. Measured on hardware: an approach
# turned 56 degrees to face its box, lost it for 3.2 s while re-acquiring, and gave up just over
# the old 3.0 s bound with the box centred and 0.63 m away.
#
# The first bound was picked on the wrong intuition -- that a heading decays as the robot drives.
# It does not: perception tracks the box in world coordinates and re-projects it through the
# CURRENT pose, so translation is already accounted for. What the bound actually has to cover is
# the box being moved or the odometry drifting under it, and for a static box on a floor that is
# comfortably a matter of seconds. This detector drops a box for that long routinely while the
# robot turns, so 3 s was a bound on the detector rather than on the evidence.
STEER_MAX_AGE = 8.0


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

    def box_in_base(self, request, snapshot=None, max_age=None):
        """Base-frame position of the request's target box.

        ``max_age`` bounds how stale a sighting may be. Steering allows a few seconds: this
        detector runs near 9 fps on marginal scores and drops a box for a frame or two
        constantly, so demanding a current sighting every cycle makes any drive fail within
        centimetres. A sighting seconds old is still a good heading, because perception tracks
        the box in world coordinates and re-projects it through the current pose.

        That tolerance stops at the grasp. confirm_graspable asks for a current sighting and a
        higher score before anything closes on it, so continuing toward a remembered box never
        becomes grasping a remembered one.
        """
        snapshot = snapshot or self._fresh_snapshot(request)
        if request.step.box_id is None:
            raise StaleGeometry("request carries no box to resolve")
        limit = STEER_MAX_AGE if max_age is None else max_age
        for box in snapshot.boxes:
            if box.id != request.step.box_id:
                continue
            age = self.clock() - box.last_seen
            if not box.current and age > limit:
                raise StaleGeometry(f"box {box.id} last seen {age:.1f}s ago, beyond the "
                                    f"{limit:.1f}s a heading may be trusted")
            return world_to_base(box.position, snapshot.base_position, snapshot.base_yaw)

        # The track is gone, not merely stale. This detector's tracker mints a NEW id whenever a
        # box blinks out and back -- ids climbed 1000, 1003, 1008, 1041 across one session on a
        # single box -- so the id the controller selected can simply cease to exist mid-approach,
        # which stopped one drive dead after two centimetres.
        #
        # Fall back to the position the request itself carries. That position was measured and
        # validated at admission, and it is in the same world frame this snapshot uses, which the
        # epoch check above has already confirmed. Steering toward it is a heading, not a claim
        # about identity: nothing here asserts that whatever is now in front IS that box.
        # confirm_graspable still demands a live, identified, confident sighting before anything
        # closes on it, so a lost track can lead the robot to the right place and still refuse the
        # grasp -- which is the correct outcome.
        if request.box is not None and vector_valid(request.box.position):
            return world_to_base(request.box.position, snapshot.base_position, snapshot.base_yaw)
        raise StaleGeometry(f"box {request.step.box_id} is not in the snapshot and the request "
                            f"carries no position to fall back on")

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

    def strongest_bearing(self, _request=None):
        """Bearing to the most confident box proposal currently in view, or None.

        Deliberately ignores the eligibility gates. Turning to look at something is free and
        reversible, so it warrants far weaker evidence than picking it up: a box out at the edge
        of a wide-angle frame is small and distorted, scores poorly for exactly that reason, and
        can never improve while nothing ever points the camera at it.
        """
        snapshot = self.observations.observe()
        if snapshot is None or not snapshot.valid:
            return None
        scan = getattr(self.observations, "_scan", lambda: None)()
        # Remembered sightings count here, and that is the whole point: everything seen during a
        # sweep is remembered by the time the sweep ends, so requiring a current detection means
        # never facing anything the survey found. perception re-projects a remembered object
        # through the CURRENT pose, so position_base_m is a valid heading even after rotating.
        # A current sighting is still preferred where one exists.
        best, best_key = None, None
        for item in (getattr(scan, "objects", ()) or ()) if scan is not None else ():
            position, score = item.get("position_base_m"), item.get("score")
            if position is None or score is None:
                continue
            key = (bool(item.get("current")), float(score))
            if best_key is None or key > best_key:
                best, best_key = position, key
        if best is None:
            return None
        best_score, seen_now = best_key[1], best_key[0]
        return {"bearing": math.atan2(best[1], best[0]), "score": best_score,
                "range": math.hypot(best[0], best[1]), "current": seen_now}

    def look_around(self, request):
        """A survey needs no target, but it can finish pointed at what it found."""
        return {"face_best": self.strongest_bearing}

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
