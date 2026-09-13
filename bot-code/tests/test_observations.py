import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import observations
import perception
from agent_types import BuildRequirements

VOXEL = (0.2, 0.2, 0.2)
SETTINGS = perception.Settings(box_size=0.2, cell=0.2)


def build_frame(origin=(1.0, 0.0, 0.0)):
    return perception.BuildFrame(list(origin), [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                 list(origin), time.time())


def pose(valid=True, epoch=3):
    return SimpleNamespace(x=0.0, y=0.0, yaw=0.0, ts=100.0, valid=valid, epoch=epoch, warning="")


def scan(points=None, build=None, surface=(), boxes=(), ts=None, valid=True):
    # column_heights gates on real wall-clock freshness, so fixtures must be current
    ts = time.time() if ts is None else ts
    return SimpleNamespace(
        ts=ts, pose=pose(valid), build=build, points=np.array(points if points is not None else []),
        surface_cells=list(surface), boxes=list(boxes), protected=[], unknown=[],
        tracks=[], warnings=[])


def column(x, y, z, n=40):
    """A dense cluster of depth points at a single height."""
    return [[x + 0.001 * i, y + 0.001 * i, z] for i in range(n)]


class OccupancyTests(unittest.TestCase):
    def cells(self, s, cells=((0, 0, 0),)):
        return observations.classify_cells(s, cells, SETTINGS)

    def test_no_build_frame_is_unknown_not_empty(self):
        result, complete = self.cells(scan(build=None))
        self.assertEqual([c.status for c in result], ["unknown"])
        self.assertFalse(complete)

    def test_a_cell_with_no_depth_return_is_unknown_not_empty(self):
        result, complete = self.cells(scan(points=[], build=build_frame()))
        self.assertEqual([c.status for c in result], ["unknown"])
        self.assertFalse(complete, "an unmeasured envelope is never complete")

    def test_a_surface_at_the_cell_bottom_reads_empty(self):
        frame = build_frame()
        bottom, top, centre = observations._cell_bounds((0, 0, 0), frame, SETTINGS)
        s = scan(points=column(centre[0], centre[1], bottom), build=frame)
        result, complete = self.cells(s)
        self.assertEqual([c.status for c in result], ["empty"])
        self.assertTrue(complete)

    def test_a_surface_at_the_cell_top_reads_occupied(self):
        frame = build_frame()
        bottom, top, centre = observations._cell_bounds((0, 0, 0), frame, SETTINGS)
        s = scan(points=column(centre[0], centre[1], top), build=frame)
        result, _ = self.cells(s)
        self.assertEqual([c.status for c in result], ["occupied"])

    def test_a_height_matching_neither_face_is_unknown(self):
        frame = build_frame()
        bottom, top, centre = observations._cell_bounds((0, 0, 0), frame, SETTINGS)
        s = scan(points=column(centre[0], centre[1], (bottom + top) / 2.0), build=frame)
        result, complete = self.cells(s)
        self.assertEqual([c.status for c in result], ["unknown"])
        self.assertFalse(complete)

    def test_a_stale_scan_yields_unknown(self):
        frame = build_frame()
        _, _, centre = observations._cell_bounds((0, 0, 0), frame, SETTINGS)
        s = scan(points=column(centre[0], centre[1], 0.0), build=frame, ts=time.time() - 100.0)
        result, _ = observations.classify_cells(s, [(0, 0, 0)], SETTINGS)
        self.assertEqual([c.status for c in result], ["unknown"])

    def test_an_occupied_cell_without_a_visible_marker_has_no_box_id(self):
        frame = build_frame()
        _, top, centre = observations._cell_bounds((0, 0, 0), frame, SETTINGS)
        result, _ = self.cells(scan(points=column(centre[0], centre[1], top), build=frame))
        self.assertIsNone(result[0].box_id, "an occluded marker attributes the cell to nobody")


class SiteTests(unittest.TestCase):
    def requirements(self, extents=(1, 1, 1)):
        return BuildRequirements(cells=((0, 0, 0),), voxel_size=VOXEL, extents=extents)

    def floor(self, centre=(1.0, 0.0), span=0.6, z=0.0, top=None):
        cells = []
        n = int(span / SETTINGS.resolution)
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                cells.append({"world": [centre[0] + i * SETTINGS.resolution,
                                        centre[1] + j * SETTINGS.resolution, z],
                              "z_min": z, "z_max": z if top is None else top})
        return cells

    def find(self, s, extents=(1, 1, 1)):
        return observations.find_clear_site(s, self.requirements(extents), SETTINGS,
                                            (0.0, 0.0, 0.0), 0.0, clock=lambda: 100.0)

    def test_unobserved_space_yields_no_site(self):
        self.assertIsNone(self.find(scan(surface=())),
                          "blank space is unknown, not free; it must never become a site")

    def test_a_flat_observed_patch_becomes_a_feasible_site(self):
        site = self.find(scan(surface=self.floor()))
        self.assertIsNotNone(site)
        self.assertTrue(site.valid and site.floor_valid and site.clearance_valid and site.feasible)
        self.assertEqual(site.frame_id, "session-world")

    def test_something_standing_on_the_patch_disqualifies_it(self):
        site = self.find(scan(surface=self.floor(top=0.30)))
        self.assertIsNone(site, "a raised surface is an obstruction, not a floor")

    def test_a_loose_box_in_the_footprint_disqualifies_it(self):
        # the only observed floor is one small patch, and a loose box is sitting on it
        floor = self.floor(centre=(0.6, 0.0), span=0.2)
        box = SimpleNamespace(id=7, pos=[0.6, 0.0, 0.1], size=0.2, color=None)
        self.assertIsNotNone(self.find(scan(surface=floor)), "the bare patch is sitable")
        self.assertIsNone(self.find(scan(surface=floor, boxes=[box])),
                          "the build must not be sited on top of its own materials")

    def test_an_invalid_pose_yields_no_site(self):
        self.assertIsNone(self.find(scan(surface=self.floor(), valid=False)))

    def test_site_axes_are_orthonormal_and_face_up(self):
        site = self.find(scan(surface=self.floor()))
        up = np.cross(site.col, site.row)
        self.assertAlmostEqual(float(np.linalg.norm(site.col)), 1.0, places=9)
        self.assertAlmostEqual(float(np.dot(site.col, site.row)), 0.0, places=9)
        # the agent requires the normal's z above .99, tighter than perception's own .9
        self.assertGreater(float(up[2]), 0.99)

    def test_a_build_too_large_for_the_observed_floor_is_not_sited(self):
        self.assertIsNone(self.find(scan(surface=self.floor(span=0.2)), extents=(8, 1, 8)))


class EligibilityTests(unittest.TestCase):
    def test_only_current_loose_correctly_sized_boxes_are_eligible(self):
        good = {"current": True, "classification": "loose", "size": 0.2}
        self.assertTrue(observations.eligible_box(good, VOXEL))
        for bad in ({**good, "current": False}, {**good, "classification": "protected"},
                    {**good, "classification": "unknown"}, {**good, "size": 0.5},
                    {**good, "size": None}):
            self.assertFalse(observations.eligible_box(bad, VOXEL), bad)


if __name__ == "__main__":
    unittest.main()


class CarryVolumeTests(unittest.TestCase):
    """Possession must answer unknown readily and empty only when it truly looked."""

    def source(self, s, centre=(0.4, 0.0, 0.3)):
        holder = SimpleNamespace(_scan=lambda: s, settings=SETTINGS)
        rig = SimpleNamespace(carry_centre=lambda: centre)
        return observations.CarryVolume(holder, rig, VOXEL, settings=SETTINGS)

    def box(self, pos, mid=7):
        return SimpleNamespace(id=mid, pos=list(pos), size=0.2, color=None)

    def test_a_box_in_the_volume_is_possession_with_its_identity(self):
        s = scan(boxes=[self.box((0.4, 0.0, 0.3))])
        held = self.source(s).holding()
        self.assertEqual((held.status, held.box_id), ("holding", 7))
        self.assertEqual(held.source, "carry-volume-detector")

    def test_a_box_outside_the_volume_is_not_possession(self):
        s = scan(boxes=[self.box((1.4, 0.0, 0.3))])
        self.assertNotEqual(self.source(s).holding().status, "holding")

    def test_no_scan_is_unknown(self):
        self.assertEqual(self.source(None).holding().status, "unknown")

    def test_no_carry_centre_is_unknown(self):
        self.assertEqual(self.source(scan(), centre=None).holding().status, "unknown")

    def test_an_invalid_pose_is_unknown(self):
        self.assertEqual(self.source(scan(valid=False)).holding().status, "unknown")

    def test_a_stale_scan_is_unknown_not_empty(self):
        s = scan(ts=time.time() - 100.0)
        held = self.source(s).holding()
        self.assertEqual(held.status, "unknown")
        self.assertEqual(held.source, "carry-volume-stale")

    def test_an_unseen_volume_is_unknown_not_empty(self):
        # no depth returns between the forearms: a blind spot, not an absence
        held = self.source(scan(points=[])).holding()
        self.assertEqual(held.status, "unknown")
        self.assertEqual(held.source, "carry-volume-occluded")

    def test_a_seen_but_boxless_volume_is_empty(self):
        seen = [[0.4 + 0.001 * i, 0.0, 0.3] for i in range(20)]
        held = self.source(scan(points=seen)).holding()
        self.assertEqual(held.status, "empty")
        self.assertEqual(held.source, "carry-volume-detector")

    def test_evidence_carries_the_measurement_time_not_the_read_time(self):
        s = scan(boxes=[self.box((0.4, 0.0, 0.3))])
        self.assertEqual(self.source(s).holding().ts, s.ts)
