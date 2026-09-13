"""In-process perception for the interface-v2 agent: sites, occupancy and motion monitoring.

The reasoning, perception and action code all run on the robot, in one process, around ONE
PerceptionSession. That matters for more than latency. Two sessions would each hold their own
Odometry, each starting at Pose(0,0,0) with its own epoch counter, so their world origins would
disagree and geometry from one would be meaningless in the other. It also keeps Scan.points in
reach: occupancy is measured from the depth cloud, and the cloud never crosses a wire.

ObservationWorker already supports this shape. Its poll loop normalizes anything that is not
already an ObservationSnapshot, and PerceptionSession.poll is exactly the session it expects; the
HTTP adapter is the remote development path, not the deployed one.

What this refuses to do is as important as what it provides. An unmeasured cell is unknown, never
empty. A footprint that was never observed is not clear. A monitor that cannot see is not safe.
Each of those would be an invented certificate, and the agent is built to act on them.
"""
from __future__ import annotations

import base64
import math
import struct
import zlib
import threading
import time

import numpy as np
from dataclasses import replace

from agent_adapters import ObservationWorker, normalize_scan
from agent_types import (MAX_IMAGE_BYTES, BuildSite, CellObservation, MotionObservation,
                         PerceptionCapabilities, SceneImage)

import perception

JPEG_MAGIC = bytes.fromhex("ffd8ff")      # SOI marker; a JPEG must start with it
PNG_MAGIC = bytes.fromhex("89504e470d0a1a0a")

# A measured column height counts as a surface when it lands within this fraction of a box edge
# of where that surface should be. Wider than this and the cell is unknown, not a near miss.
HEIGHT_TOL_FRAC = 0.25
FLOOR_TOL = 0.03          # m of z spread inside which a patch of floor counts as flat
SITE_SEARCH_RADIUS = 1.5  # m from the robot to consider for a build site
SITE_STEP = 0.10          # m between candidate site origins; matches the surface grid pitch
BOX_CLEARANCE = 0.15      # m a loose box must be clear of the footprint for it to count as free
SITE_COVERAGE = 0.8       # fraction of the footprint that must actually have been observed


def _cell_bounds(cell, build, settings):
    """(bottom, top) z of a cell in the build frame, in metres."""
    centre = perception.cell_center(cell[0], cell[1], cell[2], build, settings)
    half = settings.box_size / 2.0
    return centre[2] - half, centre[2] + half, centre


def classify_cells(scan, cells, settings):
    """Measured occupancy per cell: occupied, empty, or unknown.

    A column height near the cell's top face means something fills it. A height near its bottom
    face means the supporting surface is bare, which is the only positive evidence of empty there
    is. Anything else -- no depth return, too few points, an occluded view, a height that matches
    neither face -- is unknown. Absence of measurement is never emptiness.
    """
    if scan.build is None or not scan.pose.valid:
        return tuple(CellObservation(tuple(c), "unknown", scan.ts) for c in cells), False
    heights = perception.column_heights(list(cells), snapshot=scan, settings=settings)
    tolerance = settings.box_size * HEIGHT_TOL_FRAC
    marked = _markers_by_cell(scan, settings)
    observations, complete = [], True
    for cell in cells:
        key = tuple(cell)
        height = heights.get(key)
        bottom, top, _ = _cell_bounds(key, scan.build, settings)
        if height is None:
            observations.append(CellObservation(key, "unknown", scan.ts))
            complete = False
        elif abs(height - top) <= tolerance:
            observations.append(CellObservation(key, "occupied", scan.ts, marked.get(key)))
        elif abs(height - bottom) <= tolerance:
            observations.append(CellObservation(key, "empty", scan.ts))
        else:
            # A height that matches neither face: a partial stack, a leaning box, or a bad return.
            observations.append(CellObservation(key, "unknown", scan.ts))
            complete = False
    return tuple(observations), complete


def _markers_by_cell(scan, settings):
    """Marker id per cell, for the boxes whose markers are actually visible.

    An occluded marker leaves its cell attributed to nobody. That is honest, and it is also why
    a stack can report occupied with no box_id.
    """
    found = {}
    if scan.build is None:
        return found
    for detection in list(getattr(scan, "protected", ())) + list(getattr(scan, "boxes", ())):
        try:
            u, v = perception._grid_uv(detection.pos, scan.build, settings)
        except Exception:
            continue
        origin_z = scan.build.origin[2]
        layer = int(round((detection.pos[2] - origin_z - settings.box_size / 2.0) / settings.box_size))
        if layer < 0:
            continue
        found[(int(round(u)), layer, int(round(v)))] = detection.id
    return found


def eligible_box(track, voxel_size):
    """Only a currently-seen loose box of about the right size may be picked.

    Protected boxes are part of the build, unknown ones have no anchor to judge them against, and
    a remembered box is a memory rather than a measurement.
    """
    if not track.get("current") or track.get("classification") != "loose":
        return False
    size = track.get("size")
    if isinstance(size, (int, float)) and math.isfinite(size):
        size = (size, size, size)
    if not size or len(size) != 3:
        return False
    return all(abs(float(s) - v) <= 0.01 * v for s, v in zip(size, voxel_size))


def _floor_patch(surface_cells, centre, half_extent, radius=None):
    """Surface cells whose horizontal position falls inside a square patch."""
    inside = []
    for cell in surface_cells:
        world = cell.get("world")
        if not world or len(world) < 2:
            continue
        if (abs(world[0] - centre[0]) <= half_extent[0]
                and abs(world[1] - centre[1]) <= half_extent[1]):
            inside.append(cell)
    return inside


def find_clear_site(scan, requirements, settings, base_position, base_yaw, clock=time.time):
    """Pick a measured patch of clear floor near the robot big enough for the whole build.

    There is no marker involved. The site is defined by what the depth sensor actually saw: a
    patch whose observed surface is flat, sits at one height, has nothing standing on it, and has
    no loose box inside it. Axes come from the robot's own heading, so the build faces the robot.

    A candidate is only returned when the footprint was genuinely observed. Blank space in the
    surface grid is unknown, not free, so an unobserved patch is never offered as a site --
    which means this correctly finds nothing at all until the robot has looked around.
    """
    cells = list(getattr(scan, "surface_cells", ()) or ())
    if not cells or not scan.pose.valid:
        return None

    width, height, depth = requirements.dimensions
    half = (width / 2.0 + settings.resolution, depth / 2.0 + settings.resolution)
    # The footprint has to be substantially observed, not merely clipped at its edge. Counting
    # against the patch's own area is what stops a candidate qualifying on a sliver of floor that
    # happens to overlap its boundary while the middle was never seen at all.
    spans = (2 * half[0] / settings.resolution, 2 * half[1] / settings.resolution)
    needed = max(4, int(spans[0] * spans[1] * SITE_COVERAGE))

    boxes = [d.pos for d in list(getattr(scan, "boxes", ())) + list(getattr(scan, "unknown", ()))]
    best = None
    steps = int(SITE_SEARCH_RADIUS / SITE_STEP)
    for ix in range(-steps, steps + 1):
        for iy in range(-steps, steps + 1):
            centre = (base_position[0] + ix * SITE_STEP, base_position[1] + iy * SITE_STEP)
            distance = math.hypot(centre[0] - base_position[0], centre[1] - base_position[1])
            if distance > SITE_SEARCH_RADIUS or distance < max(half):
                continue        # too far to reach, or so close the robot stands in its own site
            patch = _floor_patch(cells, centre, half)
            if len(patch) < needed:
                continue        # the footprint was not observed; unknown is not clear
            lows = [c["z_min"] for c in patch if c.get("z_min") is not None]
            highs = [c["z_max"] for c in patch if c.get("z_max") is not None]
            if not lows or not highs:
                continue
            floor = min(lows)
            if max(highs) - floor > FLOOR_TOL:
                continue        # something stands on it, or it is not flat
            if any(abs(p[0] - centre[0]) <= half[0] + BOX_CLEARANCE
                   and abs(p[1] - centre[1]) <= half[1] + BOX_CLEARANCE for p in boxes):
                continue        # a loose box is sitting where the build would go
            if best is None or distance < best[0]:
                best = (distance, centre, floor)

    if best is None:
        return None
    distance, centre, floor = best
    c, s = math.cos(base_yaw), math.sin(base_yaw)
    col = (c, s, 0.0)                 # build +x runs along the robot's heading
    row = (-s, c, 0.0)                # build +z runs to its left; col x row is +z up
    origin = (centre[0] - col[0] * width / 2.0 - row[0] * depth / 2.0,
              centre[1] - col[1] * width / 2.0 - row[1] * depth / 2.0,
              floor)
    return BuildSite(
        id=f"floor@{centre[0]:.2f},{centre[1]:.2f}", origin=origin, col=col, row=row,
        dimensions=(width, height, depth), ts=scan.ts, epoch=scan.pose.epoch,
        frame_id="session-world", valid=True, floor_valid=True, clearance_valid=True,
        feasible=True, cost=distance)


class _Selection:
    """What the controller has chosen, shared with the polling thread under a lock.

    Sites are cached byte-for-byte once selected. The agent compares a re-checked site's
    origin/col/row by exact equality, and the frame is re-derived from a live pose on every poll,
    so re-deriving it would drift in the last decimal and read as the site having moved.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.requirements = None
        self.site_id = None
        self._sites = {}

    def remember(self, site):
        with self._lock:
            self._sites[site.id] = site
        return site

    def get(self, site_id):
        with self._lock:
            return self._sites.get(site_id)

    def snapshot_of(self):
        with self._lock:
            return self.requirements, self.site_id

    def select(self, site_id=None, requirements=None):
        with self._lock:
            if site_id is not None:
                self.site_id = site_id
            if requirements is not None:
                self.requirements = requirements


class _EnrichedSession:
    """Polls the one PerceptionSession and attaches what only the raw Scan can supply.

    Occupancy is computed here, inside the process, because it needs Scan.points. The normalized
    snapshot that leaves this method has the measurement already in it; nothing downstream ever
    needs the cloud.
    """

    def __init__(self, session, selection, voxel_size, settings, clock=time.time):
        self.session = session
        self.selection = selection
        self.voxel_size = voxel_size
        self.settings = settings
        self.clock = clock
        self.latest_scan = None

    def poll(self):
        scan = self.session.poll()
        if scan is None:
            return None
        self.latest_scan = scan
        snapshot = normalize_scan(scan, self.clock(),
                                  eligibility=lambda t: eligible_box(t, self.voxel_size))
        requirements, site_id = self.selection.snapshot_of()
        site = self.selection.get(site_id) if site_id else None
        if site is not None and requirements is not None:
            cells = _envelope(requirements)
            occupancy, complete = classify_cells(scan, cells, self.settings)
            snapshot = replace(snapshot, site_id=site_id, occupancy=occupancy,
                               occupancy_complete=complete)
        return replace(snapshot, images=self._images(scan, snapshot))

    def _images(self, scan, snapshot):
        data, ts = self.session.streams.get("rect", (b"", 0.0))
        usable = data and len(data) <= MAX_IMAGE_BYTES and data.startswith(JPEG_MAGIC)
        if usable and -0.05 <= self.clock() - ts <= self.settings.fresh_s:
            return (SceneImage("rect", "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"),
                               ts, snapshot.epoch, frame_id=snapshot.frame_id,
                               simulated=bool(self.session.mock),
                               description="Live rectified head camera; not motion authority."),)
        if self.session.mock:
            return (_plan_view(scan, snapshot),)
        # Live, with no usable frame: no image at all. The agent refuses to act without one,
        # which is the correct response to a camera that has stopped delivering. A stale frame
        # labelled fresh would be worse than none.
        return ()

    def close(self):
        close = getattr(self.session, "close", None)
        if callable(close):
            close()


def _envelope(requirements):
    """Every cell of the requested bounding box, not merely the cells that carry a block.

    The agent needs a verdict on the whole envelope, including the gaps: a box standing where a
    gap should be is exactly the obstruction worth catching.
    """
    width, height, depth = requirements.extents
    return [(x, y, z) for x in range(width) for y in range(height) for z in range(depth)]


class RobotObservations(ObservationWorker):
    """The perception half of the live provider, running in the agent's own process."""

    def __init__(self, voxel_size, *, session_factory=None, settings=None, mock=False,
                 interval=0.1, timeout=0.2, clock=time.time, **session_kwargs):
        self.settings = settings or perception.Settings(
            box_size=voxel_size[0], cell=voxel_size[0])
        self.voxel_size = tuple(voxel_size)
        self.selection = _Selection()
        self._sessions = []
        factory = session_factory or (
            lambda: perception.PerceptionSession(mock=mock, settings=self.settings, **session_kwargs))

        def build():
            session = _EnrichedSession(factory(), self.selection, self.voxel_size,
                                       self.settings, clock)
            self._sessions.append(session)
            return session

        super().__init__(build, interval=interval, timeout=min(timeout, 0.2), clock=clock,
                         retry_errors=True, close_timeout=3 * timeout + 0.5)

    def capabilities(self):
        # possession stays False: there is no holding sensor on this side. The action provider
        # supplies possession, and preflight accepts it from either.
        return PerceptionCapabilities(inventory=True, sites=True, occupancy=True,
                                      monitoring=True, images=True)

    def observe(self, site_id=None):
        if site_id is not None:
            self.selection.select(site_id=site_id)
        return super().observe(site_id)

    def find_build_sites(self, requirements):
        self.selection.select(requirements=requirements)
        snapshot = super().observe()
        scan = self._scan()
        if scan is None or not snapshot.valid:
            return ()
        site = find_clear_site(scan, requirements, self.settings,
                               snapshot.base_position, snapshot.base_yaw, self.clock)
        return (self.selection.remember(site),) if site is not None else ()

    def check_build_site(self, site_id, requirements):
        """Re-serve the site exactly as registered, having re-checked it is still clear.

        The cached frame is returned unchanged rather than re-derived: the agent compares
        origin/col/row for exact equality, and a frame recomputed from a newer pose differs in
        the last decimal even when nothing has moved.
        """
        self.selection.select(site_id=site_id, requirements=requirements)
        site = self.selection.get(site_id)
        snapshot = super().observe()
        scan = self._scan()
        if site is None or scan is None or not snapshot.valid:
            return None
        if scan.pose.epoch != site.epoch:
            return None     # a new epoch is a new world origin; the old frame means nothing in it
        current = find_clear_site(scan, requirements, self.settings,
                                  snapshot.base_position, snapshot.base_yaw, self.clock)
        if current is None:
            return replace(site, ts=scan.ts, clearance_valid=False, feasible=False)
        return replace(site, ts=scan.ts)

    def monitor(self, request, outcome):
        """Assess the running action against fresh sensing, for its actual phase.

        This reports what it can see, and says so when it cannot see. Unknown is None, which the
        agent turns into a stop -- the correct outcome, and the reason this must not fall back to
        a generic freshness flag.
        """
        try:
            snapshot = self.observe(request.site.id if request.site else None)
        except (TimeoutError, RuntimeError) as exc:
            return MotionObservation(request.request_id, outcome.action_id, outcome.phase,
                                     None, None, f"sensing unavailable: {exc}")
        if (snapshot.epoch, snapshot.frame_id) != (request.epoch, request.frame_id):
            return MotionObservation(request.request_id, outcome.action_id, outcome.phase,
                                     snapshot, False, "pose epoch changed during motion")
        if not snapshot.valid or not snapshot.pose_valid:
            return MotionObservation(request.request_id, outcome.action_id, outcome.phase,
                                     snapshot, None, "localization is invalid or stale")

        safe, reason = True, ""
        if outcome.phase in {"carrying", "lowering", "releasing"}:
            holding = outcome.holding or snapshot.holding
            if holding is None or holding.status == "unknown":
                safe, reason = None, "possession is unknown while the load is being moved"
            elif holding.status == "empty":
                safe, reason = False, "the carried load was lost"
        if safe is True and outcome.phase == "approaching":
            box = next((b for b in snapshot.boxes if b.id == request.step.box_id), None)
            if box is None or not box.current:
                safe, reason = None, "the approach target is no longer observed"
        return MotionObservation(request.request_id, outcome.action_id, outcome.phase,
                                 snapshot, safe, reason)

    def _scan(self):
        session = self._sessions[-1] if self._sessions else None
        return getattr(session, "latest_scan", None)


# Possession. A box held between the forearms occupies a known volume; a detection there is
# independent measured evidence, which a rise in joint tracking error is not. The volume is
# generous around the box because the cradle tilts it back against the upper arms.
CARRY_TOLERANCE = 0.6     # fraction of a box edge the centre may sit off the cradle midpoint
CARRY_MIN_POINTS = 8      # depth returns needed before "nothing there" means empty rather than blind


class CarryVolume:
    """Possession evidence from what the sensor sees between the forearms.

    Answers unknown far more readily than it answers empty. The forearms and the box itself
    occlude the volume at exactly the moment the answer matters, so "I cannot see it" is a
    frequent and correct verdict; reporting empty there would tell the agent a box had been
    dropped when it is simply hidden.

    Reads the most recently polled scan rather than requesting a new one, because this is called
    from the provider's status path, which is bounded at 250 ms and must not block on sensing.
    """

    def __init__(self, observations, rig, voxel_size, settings=None):
        self.observations = observations
        self.rig = rig
        self.voxel_size = tuple(voxel_size)
        self.settings = settings or observations.settings

    def __call__(self):
        return self.holding()

    def holding(self):
        from agent_types import Holding
        now = time.time()
        scan = self.observations._scan()
        centre = self.rig.carry_centre()
        if scan is None or centre is None or not scan.pose.valid:
            return Holding("unknown", ts=now, source="carry-volume-unobserved")
        if now - scan.ts > self.settings.fresh_s:
            return Holding("unknown", ts=scan.ts, source="carry-volume-stale")

        reach = self.voxel_size[0] * CARRY_TOLERANCE
        for detection in list(getattr(scan, "boxes", ())) + list(getattr(scan, "unknown", ())):
            if all(abs(detection.pos[i] - centre[i]) <= reach for i in range(3)):
                return Holding("holding", detection.id, scan.ts, "carry-volume-detector")

        # No box was identified there. Calling that empty needs proof the volume was actually
        # looked through, and the proof is NOT depth returns inside it -- an empty volume has
        # none, because the rays pass straight through. What shows it is empty is returns landing
        # BEYOND it on the same lines of sight: the camera looks along +x, so a return further out
        # than the far face means nothing was in the way.
        points = np.asarray(getattr(scan, "points", []), dtype=float).reshape(-1, 3)
        if len(points):
            finite = points[np.isfinite(points).all(axis=1)]
            through = finite[(np.abs(finite[:, 1] - centre[1]) <= reach)
                             & (np.abs(finite[:, 2] - centre[2]) <= reach)
                             & (finite[:, 0] > centre[0] + reach)]
            if len(through) >= CARRY_MIN_POINTS:
                return Holding("empty", ts=scan.ts, source="carry-volume-see-through")
        return Holding("unknown", ts=scan.ts, source="carry-volume-occluded")


def _plan_view(scan, snapshot, size=128):
    """A simulated overhead sketch of the scene, for mock runs only.

    Mock perception serves no camera frames, and the agent refuses to act without an image, so an
    offline end-to-end run needs something to look at. It is marked simulated, so nothing can
    mistake it for a camera view, and the live path never reaches it: a live session with no
    usable frame returns no image at all, which correctly stops the agent.
    """
    pad = bytes.fromhex("00")
    pixels = bytearray(bytes((242, 244, 247)) * size * size)
    points = [(0.0, 0.0)] + [(b.position[0], b.position[1]) for b in snapshot.boxes]
    if scan.build is not None:
        points.append((scan.build.origin[0], scan.build.origin[1]))
    left, right = min(p[0] for p in points) - .5, max(p[0] for p in points) + .5
    bottom, top = min(p[1] for p in points) - .5, max(p[1] for p in points) + .5

    def square(position, color, radius):
        x = round((position[0] - left) / (right - left) * (size - 1))
        y = round((top - position[1]) / (top - bottom) * (size - 1))
        for row in range(max(0, y - radius), min(size, y + radius + 1)):
            for col in range(max(0, x - radius), min(size, x + radius + 1)):
                offset = (row * size + col) * 3
                pixels[offset:offset + 3] = bytes(color)

    if scan.build is not None:
        square(scan.build.origin, (170, 175, 185), 4)
    for box in snapshot.boxes:
        square((box.position[0], box.position[1]), (45, 165, 85), 3)
    square((0.0, 0.0), (225, 160, 20), 4)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(pad + pixels[r * size * 3:(r + 1) * size * 3] for r in range(size))
    png = (PNG_MAGIC + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    return SceneImage("overview", "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
                      snapshot.captured_at, snapshot.epoch, frame_id=snapshot.frame_id,
                      simulated=True, description="SIMULATED plan view; mock perception, not a camera.")
