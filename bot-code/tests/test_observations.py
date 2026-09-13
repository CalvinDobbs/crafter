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


def site(origin=(1.0, 0.0, 0.0)):
    from agent_types import BuildSite
    return BuildSite(id="s1", origin=tuple(float(v) for v in origin), col=(1.0, 0.0, 0.0),
                     row=(0.0, 1.0, 0.0), dimensions=(0.2, 0.2, 0.2), ts=time.time(), epoch=3,
                     frame_id="session-world", valid=True, floor_valid=True,
                     clearance_valid=True, feasible=True)


def pose(valid=True, epoch=3):
    return SimpleNamespace(x=0.0, y=0.0, yaw=0.0, ts=100.0, valid=valid, epoch=epoch, warning="")


def scan(points=None, build=None, surface=(), boxes=(), objects=(), ts=None, valid=True):
    # column_heights gates on real wall-clock freshness, so fixtures must be current
    ts = time.time() if ts is None else ts
    return SimpleNamespace(
        ts=ts, pose=pose(valid), build=build, points=np.array(points if points is not None else []),
        surface_cells=list(surface), boxes=list(boxes), protected=[], unknown=[],
        tracks=[], objects=list(objects), warnings=[])


def column(x, y, z, n=40):
    """A dense cluster of depth points at a single height, inside _height_at's sampling disk."""
    return [[x + 0.001 * i, y + 0.001 * (i % 7), z] for i in range(n)]


class OccupancyTests(unittest.TestCase):
    def cells(self, s, cells=((0, 0, 0),), selected=None):
        chosen = site() if selected is None else (selected or None)
        return observations.classify_cells(s, chosen, cells, VOXEL, SETTINGS)

    def test_no_selected_site_is_unknown_not_empty(self):
        result, complete = self.cells(scan(build=None), selected=False)
        self.assertEqual([c.status for c in result], ["unknown"])
        self.assertFalse(complete)

    def test_a_cell_with_no_depth_return_is_unknown_not_empty(self):
        result, complete = self.cells(scan(points=[], build=build_frame()))
        self.assertEqual([c.status for c in result], ["unknown"])
        self.assertFalse(complete, "an unmeasured envelope is never complete")

    def test_a_surface_at_the_cell_bottom_reads_empty(self):
        frame = build_frame()
        bottom, top, centre = observations._cell_bounds(site(), (0, 0, 0), VOXEL, pose())
        s = scan(points=column(centre[0], centre[1], bottom), build=frame)
        result, complete = self.cells(s)
        self.assertEqual([c.status for c in result], ["empty"])
        self.assertTrue(complete)

    def test_a_surface_at_the_cell_top_reads_occupied(self):
        frame = build_frame()
        bottom, top, centre = observations._cell_bounds(site(), (0, 0, 0), VOXEL, pose())
        s = scan(points=column(centre[0], centre[1], top), build=frame)
        result, _ = self.cells(s)
        self.assertEqual([c.status for c in result], ["occupied"])

    def test_a_height_matching_neither_face_is_unknown(self):
        frame = build_frame()
        bottom, top, centre = observations._cell_bounds(site(), (0, 0, 0), VOXEL, pose())
        s = scan(points=column(centre[0], centre[1], (bottom + top) / 2.0), build=frame)
        result, complete = self.cells(s)
        self.assertEqual([c.status for c in result], ["unknown"])
        self.assertFalse(complete)

    def test_a_stale_scan_yields_unknown(self):
        frame = build_frame()
        _, _, centre = observations._cell_bounds(site(), (0, 0, 0), VOXEL, pose())
        s = scan(points=column(centre[0], centre[1], 0.0), build=frame, ts=time.time() - 100.0)
        result, _ = observations.classify_cells(s, site(), [(0, 0, 0)], VOXEL, SETTINGS)
        self.assertEqual([c.status for c in result], ["unknown"])

    def test_an_occupied_cell_without_a_visible_marker_has_no_box_id(self):
        frame = build_frame()
        _, top, centre = observations._cell_bounds(site(), (0, 0, 0), VOXEL, pose())
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

    def test_a_detected_box_in_the_footprint_disqualifies_it(self):
        # the only observed floor is one small patch, and the detector sees a box on it
        floor = self.floor(centre=(0.6, 0.0), span=0.2)
        obj = {"position_base_m": [0.6, 0.0, 0.1], "current": True, "score": 0.7}
        self.assertIsNotNone(self.find(scan(surface=floor)), "the bare patch is sitable")
        self.assertIsNone(self.find(scan(surface=floor, objects=[obj])),
                          "the build must not be sited on top of its own materials")

    def test_a_low_confidence_detection_still_blocks_the_footprint(self):
        # too uncertain to PICK is not too uncertain to be in the way
        floor = self.floor(centre=(0.6, 0.0), span=0.2)
        faint = {"position_base_m": [0.6, 0.0, 0.1], "current": True, "score": 0.11}
        self.assertIsNone(self.find(scan(surface=floor, objects=[faint])))

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


def proposal(**over):
    """A detector proposal that passes every gate, so each test can spoil exactly one."""
    base = {"source": "box_detector", "label": "cardboard_box", "current": True,
            "classification": "loose", "identity_status": "tracked", "partial_view": False,
            "depth_status": "surface_supported", "score": 0.62, "size": None}
    base.update(over)
    return base


class EligibilityTests(unittest.TestCase):
    """There are no markers in this world, so every box is a detector proposal and every gate
    here is a confidence policy over evidence rather than a measurement."""

    def test_a_confident_supported_proposal_is_eligible(self):
        self.assertTrue(observations.eligible_box(proposal(), VOXEL))

    def test_an_unmeasured_size_does_not_disqualify_it(self):
        # the detector cannot measure a box; requiring a size would mean fabricating one
        self.assertTrue(observations.eligible_box(proposal(size=None), VOXEL))

    def test_low_confidence_is_rejected(self):
        self.assertFalse(observations.eligible_box(proposal(score=0.19), VOXEL))
        self.assertFalse(observations.eligible_box(proposal(score=None), VOXEL))

    def test_neither_threshold_drops_below_the_handoff_minimum(self):
        self.assertGreaterEqual(observations.SCORE_SELECT, 0.20)
        self.assertGreaterEqual(observations.SCORE_PICKUP, 0.20)

    def test_grasping_demands_more_confidence_than_selecting(self):
        # selecting a box can be abandoned after a closer look; closing two arms on one cannot
        self.assertGreater(observations.SCORE_PICKUP, observations.SCORE_SELECT)

    def test_a_box_good_enough_to_approach_may_not_be_good_enough_to_grasp(self):
        between = (observations.SCORE_SELECT + observations.SCORE_PICKUP) / 2
        self.assertTrue(observations.eligible_box(proposal(score=between), VOXEL),
                        "it is worth approaching for a closer look")
        self.assertLess(between, observations.SCORE_PICKUP,
                        "but the grasp must ask again with fresher evidence")

    def test_a_proposal_without_a_support_plane_is_rejected(self):
        # flat floor or wall reads as box-shaped; the support plane is what separates them
        for status in ("missing", "background_or_flat_surface", "weak_no_support_plane",
                       "inconsistent_depth"):
            self.assertFalse(observations.eligible_box(proposal(depth_status=status), VOXEL), status)

    def test_a_clipped_proposal_is_rejected(self):
        # a box half out of frame has a centroid that is not its visible face
        self.assertFalse(observations.eligible_box(proposal(partial_view=True), VOXEL))

    def test_ambiguous_or_stale_identity_is_rejected(self):
        for status in ("ambiguous", "pose_epoch_changed"):
            self.assertFalse(observations.eligible_box(proposal(identity_status=status), VOXEL), status)

    def test_a_remembered_proposal_is_rejected(self):
        self.assertFalse(observations.eligible_box(proposal(current=False), VOXEL))

    def test_a_box_inside_the_build_footprint_is_protected_not_material(self):
        self.assertFalse(observations.eligible_box(proposal(classification="protected"), VOXEL))

    def test_a_non_detector_track_is_rejected(self):
        self.assertFalse(observations.eligible_box(proposal(source="aruco"), VOXEL))

    def test_rejection_names_the_gate_that_stopped_it(self):
        # "nothing was eligible" is useless when a build will not start; this layer knows why
        cases = {
            "confidence": proposal(score=0.19),
            "depth": proposal(depth_status="background_or_flat_surface"),
            "clipped by the frame": proposal(partial_view=True),
            "inside the build footprint": proposal(classification="protected"),
            "identity": proposal(identity_status="ambiguous"),
            "remembered": proposal(current=False),
        }
        for expected, track in cases.items():
            reason = observations.box_rejection(track, VOXEL)
            self.assertIsNotNone(reason)
            self.assertIn(expected, reason)

    def test_an_eligible_proposal_has_no_rejection_reason(self):
        self.assertIsNone(observations.box_rejection(proposal(), VOXEL))


if __name__ == "__main__":
    unittest.main()


class CarryVolumeTests(unittest.TestCase):
    """Possession must answer unknown readily and empty only when it truly looked."""

    def source(self, s, centre=(0.4, 0.0, 0.3)):
        holder = SimpleNamespace(_scan=lambda: s, settings=SETTINGS)
        rig = SimpleNamespace(carry_centre=lambda: centre)
        return observations.CarryVolume(holder, rig, VOXEL, settings=SETTINGS)

    def box(self, pos, mid=7):
        """A detector proposal sitting at pos."""
        return {"id": mid, "position_base_m": list(pos), "current": True, "score": 0.7}

    def test_a_box_in_the_volume_is_possession_with_its_identity(self):
        s = scan(objects=[self.box((0.4, 0.0, 0.3))])
        held = self.source(s).holding()
        self.assertEqual((held.status, held.box_id), ("holding", 7))
        self.assertEqual(held.source, "carry-volume-detector")

    def test_a_box_outside_the_volume_is_not_possession(self):
        s = scan(objects=[self.box((1.4, 0.0, 0.3))])
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

    def test_seeing_through_the_volume_to_something_beyond_is_empty(self):
        # returns landing further out on the same lines of sight: nothing was in the way
        beyond = [[1.2, 0.0 + 0.001 * i, 0.3] for i in range(20)]
        held = self.source(scan(points=beyond)).holding()
        self.assertEqual(held.status, "empty")
        self.assertEqual(held.source, "carry-volume-see-through")

    def test_returns_inside_the_volume_alone_do_not_prove_it_is_empty(self):
        # an empty volume has no returns IN it; points there mean something is there
        inside = [[0.4, 0.0 + 0.001 * i, 0.3] for i in range(20)]
        self.assertEqual(self.source(scan(points=inside)).holding().status, "unknown")

    def test_evidence_carries_the_measurement_time_not_the_read_time(self):
        s = scan(objects=[self.box((0.4, 0.0, 0.3))])
        self.assertEqual(self.source(s).holding().ts, s.ts)
