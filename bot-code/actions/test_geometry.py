import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # agent_types
sys.path.insert(0, str(Path(__file__).resolve().parent))          # geometry

from agent_types import BuildRequirements, BuildSite, ObservationSnapshot, Step
from geometry import Geometry, StaleGeometry, world_to_base

VOXEL = (0.2, 0.2, 0.2)


def snapshot(epoch=3, captured_at=100.0, valid=True, pose=(0.0, 0.0, 0.0), yaw=0.0):
    return ObservationSnapshot(
        revision="r1", captured_at=captured_at, received_at=captured_at, epoch=epoch,
        frame_id="session-world", valid=valid, pose_valid=valid,
        base_position=pose, base_yaw=yaw)


def request(epoch=3, cell=(0, 0, 0), site=True, frame_id="session-world"):
    build = BuildSite(id="s1", origin=(1.0, 0.0, 0.0), col=(1.0, 0.0, 0.0), row=(0.0, 1.0, 0.0),
                      dimensions=(1.0, 1.0, 1.0), ts=100.0, epoch=epoch, frame_id=frame_id)
    from agent_types import BoxObservation
    return SimpleNamespace(
        epoch=epoch, frame_id=frame_id,
        step=Step("place", site_id="s1", cell=cell, box_id=7),
        site=build if site else None,
        box=BoxObservation(id=7, position=(1.0, 0.0, 0.1), size=None, last_seen=100.0),
        requirements=BuildRequirements(cells=(cell,), voxel_size=VOXEL, extents=(1, 1, 1)))


class Observations:
    def __init__(self, snap):
        self.snap = snap
        self.site_ids = []

    def observe(self, site_id=None):
        self.site_ids.append(site_id)
        return self.snap


def geometry(snap, carry=1.0, now=100.5):
    return Geometry(Observations(snap), carry_height=lambda: carry, clock=lambda: now)


class WorldToBaseTests(unittest.TestCase):
    def test_identity_when_base_is_at_the_origin(self):
        self.assertEqual(world_to_base((1.0, 2.0, 3.0), (0.0, 0.0, 0.0), 0.0), (1.0, 2.0, 3.0))

    def test_inverts_the_adapters_rotate_then_translate(self):
        # normalize_perception maps base->world as dx + c*x - s*y, dy + s*x + c*y
        base_point, pose, yaw = (0.4, 0.1, 0.2), (1.0, 2.0, 0.0), math.pi / 2
        c, s = math.cos(yaw), math.sin(yaw)
        world = (pose[0] + c * base_point[0] - s * base_point[1],
                 pose[1] + s * base_point[0] + c * base_point[1], base_point[2])
        back = world_to_base(world, pose, yaw)
        for got, want in zip(back, base_point):
            self.assertAlmostEqual(got, want, places=9)


class SteeringStalenessTests(unittest.TestCase):
    """A drive must survive the detector dropping a box for a frame; a grasp must not."""

    def snapshot_with(self, box_id=7, current=True, age=0.0, now=100.5):
        from agent_types import BoxObservation
        snap = snapshot()
        box = BoxObservation(id=box_id, position=(1.0, 0.0, 0.1), size=None,
                             last_seen=now - age, current=current)
        return SimpleNamespace(**{**snap.__dict__, "boxes": (box,)})

    def test_a_current_box_steers(self):
        g = geometry(snapshot())
        pos = g.box_in_base(request(), snapshot=self.snapshot_with(current=True))
        self.assertEqual(len(pos), 3)

    def test_a_briefly_lost_box_still_steers(self):
        g = geometry(snapshot())
        pos = g.box_in_base(request(), snapshot=self.snapshot_with(current=False, age=1.0))
        self.assertEqual(len(pos), 3)

    def test_a_long_lost_box_stops_the_drive(self):
        g = geometry(snapshot())
        stale = self.snapshot_with(current=False, age=geometry(snapshot()).__class__ and 99.0)
        with self.assertRaises(StaleGeometry) as caught:
            g.box_in_base(request(), snapshot=stale)
        self.assertIn("beyond", str(caught.exception))

    def test_a_lost_track_falls_back_to_the_validated_position(self):
        """The tracker mints a new id when a box blinks, so the selected id can vanish.

        Steering to the position the request carries is a heading, not a claim that whatever is
        there is that box. The grasp still demands a live identified sighting.
        """
        g = geometry(snapshot())
        empty = SimpleNamespace(**{**snapshot().__dict__, "boxes": ()})
        position = g.box_in_base(request(), snapshot=empty)
        self.assertEqual(len(position), 3)

    def test_a_lost_track_with_no_fallback_position_stops_the_drive(self):
        g = geometry(snapshot())
        empty = SimpleNamespace(**{**snapshot().__dict__, "boxes": ()})
        bare = SimpleNamespace(**{**request().__dict__, "box": None})
        with self.assertRaises(StaleGeometry):
            g.box_in_base(bare, snapshot=empty)

    def test_a_cross_epoch_request_is_still_refused_despite_the_fallback(self):
        # the fallback position is only meaningful in the same world frame
        g = geometry(snapshot(epoch=4))
        with self.assertRaises(StaleGeometry):
            g.box_in_base(request(epoch=3))


class ResolveTests(unittest.TestCase):
    def test_place_measures_the_drop_from_the_carry_height(self):
        # cell (0,0,0) centre is origin + up*0.5*voxel -> z = 0.1; its bottom face is z = 0.0
        result = geometry(snapshot(), carry=0.85).place(request())
        self.assertAlmostEqual(result["z_drop"], 0.85, places=9)

    def test_site_id_is_passed_through_so_occupancy_can_be_scoped(self):
        observations = Observations(snapshot())
        Geometry(observations, carry_height=lambda: 1.0, clock=lambda: 100.5).place(request())
        self.assertEqual(observations.site_ids, ["s1"])

    def test_rejects_a_cross_epoch_request(self):
        with self.assertRaises(StaleGeometry) as caught:
            geometry(snapshot(epoch=4)).place(request(epoch=3))
        self.assertIn("epoch", str(caught.exception))

    def test_rejects_an_invalid_pose(self):
        with self.assertRaises(StaleGeometry):
            geometry(snapshot(valid=False)).place(request())

    def test_rejects_an_observation_that_has_aged_out(self):
        with self.assertRaises(StaleGeometry) as caught:
            geometry(snapshot(captured_at=100.0), now=105.0).place(request())
        self.assertIn("old", str(caught.exception))

    def test_rejects_a_target_above_the_load_rather_than_clamping(self):
        # carry height below the cell bottom: lowering cannot reach it
        with self.assertRaises(StaleGeometry) as caught:
            geometry(snapshot(), carry=-0.5).place(request())
        self.assertIn("not reachable", str(caught.exception))

    def test_rejects_a_request_with_no_site(self):
        with self.assertRaises(StaleGeometry):
            geometry(snapshot()).place(request(site=False))

    def test_base_pose_offset_shifts_the_target(self):
        near = geometry(snapshot(pose=(0.0, 0.0, 0.0)), carry=1.0).place(request())
        far = geometry(snapshot(pose=(0.5, 0.0, 0.0)), carry=1.0).place(request())
        self.assertAlmostEqual(near["target"][0] - far["target"][0], 0.5, places=9)
        # a purely horizontal move must not change how far the box has to descend
        self.assertAlmostEqual(near["z_drop"], far["z_drop"], places=9)


if __name__ == "__main__":
    unittest.main()
