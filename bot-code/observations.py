"""In-process perception for the interface-v2 agent: sites, occupancy and motion monitoring.

**There are no fiducial markers in this world.** That single fact shapes everything here, so it is
worth stating plainly against what perception.py offers:

  - perception.py's default detector is `aruco`, and its `Scan.tracks` are built only inside the
    marker loop. With no markers that list is empty, every classification falls back to "unknown",
    and nothing is ever a pick candidate. Running it that way sees no boxes at all.
  - So this provider always starts the session with `detector=True`, and every box below is a
    YOLO-World proposal. perception.py hardcodes `pick_candidate=False` on those by design, and
    that is not overridden: eligibility is decided here instead, which is where the perception
    handoff asks for it ("a higher confidence threshold in an adapter/selection policy").
  - A proposal reports the *visible surface centroid* of a box, never a centre, with `grasp_pose`
    always None and no size at all. Nothing here invents either. The action side approaches the
    face and lets contact find the box, and BoxObservation.size stays None because a fabricated
    size would turn an assumption into evidence.
  - perception.py classifies loose vs protected against its ArUco anchor, which does not exist, so
    it reports everything as "unassigned". That judgement is re-made here against the measured
    floor site the controller actually selected.

The rest of the design follows from running in one process. Reasoning, perception and actions all
share a single PerceptionSession: two would each hold their own Odometry, starting at Pose(0,0,0)
with separate epoch counters, so their world origins would disagree and geometry from one would be
meaningless in the other. It also keeps Scan.points in reach, so occupancy is measured from the
depth cloud without the cloud ever crossing a wire.

ObservationWorker already supports this shape. Its poll loop normalizes anything that is not
already an ObservationSnapshot, and PerceptionSession.poll is exactly the session it expects; the
HTTP adapter is the remote development path, not the deployed one.

What this refuses to do is as important as what it provides. An unmeasured cell is unknown, never
empty. A footprint that was never observed is not clear. A monitor that cannot see is not safe. A
proposal without a support plane is not a box. Each of those would be an invented certificate, and
the agent is built to act on them.
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

from types import SimpleNamespace

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


def world_to_base(point, pose):
    """Session-world xyz -> the scan's base-at-capture frame. Planar, like the pose itself."""
    dx, dy = point[0] - pose.x, point[1] - pose.y
    c, s = math.cos(-pose.yaw), math.sin(-pose.yaw)
    return (c * dx - s * dy, s * dx + c * dy, point[2])


def _cell_bounds(site, cell, voxel_size, pose):
    """(bottom, top, base-frame centre) of a cell of the SELECTED site.

    The site the controller chose is the frame that matters. Measuring against perception's own
    anchor instead would sample the depth cloud in a different place entirely and report occupancy
    for cells nobody asked about.
    """
    centre = world_to_base(site.cell_center(cell, voxel_size), pose)
    half = voxel_size[1] / 2.0
    return centre[2] - half, centre[2] + half, centre


def classify_cells(scan, site, cells, voxel_size, settings):
    """Measured occupancy per cell of the selected site: occupied, empty, or unknown.

    A column height near the cell's top face means something fills it. A height near its bottom
    face means the supporting surface is bare, which is the only positive evidence of empty there
    is. Anything else -- no depth return, too few points, an occluded view, a height that matches
    neither face -- is unknown. Absence of measurement is never emptiness.
    """
    if site is None or not scan.pose.valid or time.time() - scan.ts > settings.fresh_s:
        return tuple(CellObservation(tuple(c), "unknown", scan.ts) for c in cells), False
    tolerance = voxel_size[1] * HEIGHT_TOL_FRAC
    radius = voxel_size[0] * 0.35
    marked = _boxes_by_cell(scan, site, voxel_size)
    observations_out, complete = [], True
    for cell in cells:
        key = tuple(cell)
        bottom, top, centre = _cell_bounds(site, key, voxel_size, scan.pose)
        height = perception._height_at(scan.points, centre, radius)
        if height is None:
            observations_out.append(CellObservation(key, "unknown", scan.ts))
            complete = False
        elif abs(height - top) <= tolerance:
            observations_out.append(CellObservation(key, "occupied", scan.ts, marked.get(key)))
        elif abs(height - bottom) <= tolerance:
            observations_out.append(CellObservation(key, "empty", scan.ts))
        else:
            # A height matching neither face: a partial stack, a leaning box, or a bad return.
            observations_out.append(CellObservation(key, "unknown", scan.ts))
            complete = False
    return tuple(observations_out), complete


def _boxes_by_cell(scan, site, voxel_size):
    """Detector-proposed box id per cell of the selected site, where one sits in a cell.

    Attribution is best effort. A stack occludes the boxes underneath it, so a cell can read
    occupied with no id at all -- that is honest, and the agent tolerates it everywhere except a
    cell it has already confirmed.
    """
    found = {}
    if site is None:
        return found
    for item in getattr(scan, "objects", ()) or ():
        base = item.get("position_base_m")
        if base is None or not item.get("current"):
            continue
        for cell in found_cells_near(site, SimpleNamespace(pos=base, id=item.get("id")),
                                     voxel_size, scan.pose):
            found[cell] = item.get("id")
    return found


def found_cells_near(site, detection, voxel_size, pose, tolerance=0.5):
    """Cells of the site whose centre this detection is sitting in, if any."""
    world = site.cell_center((0, 0, 0), voxel_size)
    base0 = world_to_base(world, pose)
    offset = [detection.pos[i] - base0[i] for i in range(3)]
    col = world_to_base((site.origin[0] + site.col[0], site.origin[1] + site.col[1],
                         site.origin[2] + site.col[2]), pose)
    origin = world_to_base(site.origin, pose)
    axis = [col[i] - origin[i] for i in range(3)]
    norm = math.sqrt(sum(a * a for a in axis)) or 1.0
    along = sum(offset[i] * axis[i] / norm for i in range(3)) / voxel_size[0]
    layer = offset[2] / voxel_size[1]
    x, y = round(along), round(layer)
    if abs(along - x) > tolerance or abs(layer - y) > tolerance or y < 0:
        return ()
    return ((x, y, 0),)


def box_rejection(track, voxel_size):
    """Why this detector proposal may not be picked, or None if it may be.

    Returning the reason rather than a bare False is the difference between "no box was eligible"
    and "three were rejected for low confidence" -- the first tells you nothing when a build will
    not start, and this is the layer that knows.
    """
    if track.get("source") != "box_detector":
        return "not a detector proposal"
    if track.get("label") not in (None, DETECTOR_LABEL):
        return f"label {track.get('label')!r}"
    if not track.get("current"):
        return "remembered, not currently seen"
    if track.get("classification") == "protected":
        return "inside the build footprint"
    if track.get("classification") != "loose":
        return f"classification {track.get('classification')!r}"
    if track.get("identity_status") in {"ambiguous", "pose_epoch_changed"}:
        return f"identity {track.get('identity_status')}"
    if track.get("partial_view"):
        return "clipped by the frame"
    status = track.get("depth_status")
    if status not in DEPTH_OK:
        return f"depth {status}"
    score = track.get("score")
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        return "no confidence score"
    if score < SCORE_SELECT:
        return f"confidence {score:.2f} below {SCORE_SELECT:.2f}"
    confirmations = track.get("confirmations") or 0
    if confirmations < MIN_CONFIRMATIONS:
        # A single frame at this confidence is a guess. Repetition is what makes it evidence,
        # and it is the half of the bar that went UP when the score half came down.
        return f"seen {confirmations}x, needs {MIN_CONFIRMATIONS} confirmations at this confidence"
    return None


def eligible_box(track, voxel_size):
    """Whether a detector proposal may be selected as material to pick.

    Nothing here measures a box. The detector proposes a rectangle it believes is a cardboard box,
    and the depth association gives that rectangle a position and a support plane. So this is a
    confidence policy over evidence, and its job is to reject the cases that look like boxes and
    are not: a flat region of floor or wall, a proposal with no usable depth, one clipped by the
    image edge whose centroid is meaningless, and a group of objects whose silhouette happens to
    read as one cuboid -- an observed failure mode, not a hypothetical one.

    Size is deliberately not checked: the detector cannot measure it, and a fabricated size would
    turn an assumption into evidence.
    """
    return box_rejection(track, voxel_size) is None


def detector_tracks(scan, site, voxel_size, pose):
    """Detector proposals as tracks, classified against the site the controller chose.

    perception classifies loose vs protected against its own ArUco anchor, which does not exist
    here, so everything it reports is "unassigned". This re-does that judgement against the
    measured floor site instead: a box standing inside the footprint is part of the build and must
    not be picked up again.
    """
    tracks = []
    for item in getattr(scan, "objects", ()) or ():
        world = item.get("world_position_m")
        base = item.get("position_base_m")
        if world is None or base is None:
            continue
        protected = site is not None and _inside_footprint(site, base, voxel_size, pose)
        tracks.append({
            "id": item.get("id"),
            "world": [float(v) for v in world],
            "pos": [float(v) for v in base],
            "last_seen": float(item.get("last_seen", scan.ts)),
            "current": bool(item.get("current")),
            "size": None,                     # the detector cannot measure one
            "classification": "protected" if protected else "loose",
            "source": "box_detector",
            "score": item.get("score"),
            "label": item.get("label"),
            "depth_status": item.get("depth_status"),
            "partial_view": bool(item.get("partial_view")),
            "identity_status": item.get("identity_status"),
            "support_clearance_m": item.get("support_clearance_m"),
            "confirmations": item.get("confirmations", 0),
        })
    return tracks


def _inside_footprint(site, base_point, voxel_size, pose, margin=0.04):
    """Is a base-frame point standing inside the site's build footprint?"""
    origin = world_to_base(site.origin, pose)
    c, r = site.col, site.row
    # express the offset in the site's own axes, rotated into the base frame by -yaw
    ca, sa = math.cos(-pose.yaw), math.sin(-pose.yaw)
    cb = (ca * c[0] - sa * c[1], sa * c[0] + ca * c[1])
    rb = (ca * r[0] - sa * r[1], sa * r[0] + ca * r[1])
    dx, dy = base_point[0] - origin[0], base_point[1] - origin[1]
    u = dx * cb[0] + dy * cb[1]
    v = dx * rb[0] + dy * rb[1]
    return (-margin <= u <= site.dimensions[0] + margin
            and -margin <= v <= site.dimensions[2] + margin)


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

    # anything the detector currently proposes counts as an obstruction to build over,
    # regardless of whether it passed the stricter pick policy
    boxes = [o["position_base_m"] for o in (getattr(scan, "objects", ()) or ())
             if o.get("position_base_m") is not None and o.get("current")]
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
    # Plain Python floats, deliberately: agent_types.vector_valid tests `type(v) in (float, int)`,
    # so a numpy scalar arriving from the depth data would fail validation and the site would be
    # discarded without explanation.
    origin = (float(centre[0] - col[0] * width / 2.0 - row[0] * depth / 2.0),
              float(centre[1] - col[1] * width / 2.0 - row[1] * depth / 2.0),
              float(floor))
    col = tuple(float(v) for v in col)
    row = tuple(float(v) for v in row)
    return BuildSite(
        id=f"floor@{centre[0]:.2f},{centre[1]:.2f}", origin=origin, col=col, row=row,
        dimensions=(float(width), float(height), float(depth)),
        ts=float(scan.ts), epoch=int(scan.pose.epoch),
        frame_id="session-world", valid=True, floor_valid=True, clearance_valid=True,
        feasible=True, cost=float(distance))


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
        requirements, site_id = self.selection.snapshot_of()
        site = self.selection.get(site_id) if site_id else None
        # Boxes come from the detector, never from markers: there are none in this world. The
        # scan's own tracks list stays empty, so substitute the detector's proposals before
        # normalising, classified against the site the controller actually chose.
        source = SimpleNamespace(
            ts=scan.ts, pose=scan.pose, warnings=getattr(scan, "warnings", []),
            tracks=detector_tracks(scan, site, self.voxel_size, scan.pose))
        snapshot = normalize_scan(source, self.clock(),
                                  eligibility=lambda tr: eligible_box(tr, self.voxel_size))
        rejected = [box_rejection(tr, self.voxel_size) for tr in source.tracks]
        reasons = sorted({r for r in rejected if r})
        if reasons and not any(r is None for r in rejected):
            snapshot = replace(snapshot, warnings=snapshot.warnings + (
                f"{len(rejected)} box proposal(s), none eligible: " + "; ".join(reasons),))
        if site is not None and requirements is not None:
            cells = _envelope(requirements)
            occupancy, complete = classify_cells(scan, site, cells, self.voxel_size, self.settings)
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
        # fresh_s has to exceed the pipeline's own latency or nothing is ever fresh. Measured on
        # this robot: GPU inference is ~20 ms, but the CPU depth association adds ~26 ms, results
        # land 235 ms old at p50 and 284 ms at p95, and upstream camera.rect publishes on a 500 ms
        # window. Perception's 0.8 s default sits barely above that sum, so sightings flickered
        # across the current/remembered boundary and a survey would report several proposals with
        # none of them current. This is not a slow detector -- it is a freshness window set for a
        # faster pipeline than this one.
        self.settings = settings or perception.Settings(
            box_size=voxel_size[0], cell=voxel_size[0], fresh_s=PERCEPTION_FRESH_S)
        self.voxel_size = tuple(voxel_size)
        self.selection = _Selection()
        self._sessions = []
        # detector=True is not the perception default: without markers it is the only way a box is
        # seen at all, so this provider always asks for it.
        session_kwargs.setdefault("detector", True)
        session_kwargs.setdefault("detector_backend", "tensorrt")
        factory = session_factory or (
            lambda: perception.PerceptionSession(mock=mock, settings=self.settings, **session_kwargs))

        def build():
            session = _EnrichedSession(factory(), self.selection, self.voxel_size,
                                       self.settings, clock)
            self._sessions.append(session)
            return session

        # A detector session's first poll loads a TensorRT engine, which takes seconds; the worker
        # would otherwise declare it unclosable while it is merely starting. observe() keeps its
        # tight bound -- callers retry -- but shutdown has to outlast a cold start.
        close_timeout = 30.0 if session_kwargs.get("detector") else 3 * timeout + 0.5
        super().__init__(build, interval=interval, timeout=min(timeout, 0.2), clock=clock,
                         retry_errors=True, close_timeout=close_timeout)

    def capabilities(self):
        # possession stays False: there is no holding sensor on this side. The action provider
        # supplies possession, and preflight accepts it from either.
        return PerceptionCapabilities(inventory=True, sites=True, occupancy=True,
                                      monitoring=True, images=True)

    def observe(self, site_id=None):
        """Snapshot scoped to the selected site, with occupancy measured for THAT site.

        Selecting a site changes what the polling thread measures, so the cached snapshot from
        before the selection has no occupancy in it. Returning that one would report occupancy
        permanently unavailable. Wait, briefly and boundedly, for a snapshot that actually carries
        the requested site -- still well inside the 250 ms this call is allowed.
        """
        if site_id is None:
            return super().observe(None)
        self.selection.select(site_id=site_id)
        deadline = time.monotonic() + self.timeout
        snapshot = super().observe(site_id)
        while snapshot.site_id != site_id and time.monotonic() < deadline:
            time.sleep(self.interval / 2 or 0.005)
            snapshot = super().observe(site_id)
        return snapshot

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
CARRY_MIN_POINTS = 8

# Markerless box selection. There are no fiducial markers on these boxes, so every box comes from
# the YOLO-World detector and every gate below is this adapter's policy, not perception's. The
# perception handoff asks for exactly that: a higher confidence threshold applied in an
# adapter/selection policy rather than inside the detector.
DETECTOR_LABEL = "cardboard_box"
# Thresholds, lowered on the robot owner's explicit instruction (2026-09-13) after measuring
# this robot with the boxes it actually has to work with. Written down because the numbers look
# alarming without the reason.
#
# The handoff recommends 0.25 to select and 0.30 to grasp, and says not to drop under 0.20. On
# this hardware the boxes are small and the head camera is wide-angle, so a box at working range
# occupies a small, distorted patch. Measured over a full session, the SAME box scored anywhere
# from 0.10 to 0.37 -- a mean sitting right on those thresholds with variance straddling them, so
# eligibility flickered frame to frame. The boxes cannot be changed.
#
# Lowering the single-frame bar alone would be exactly the invented confidence this design exists
# to refuse. So the bar comes down and a second, independent requirement goes up: the track must
# have been confirmed across several frames. One frame at 0.30 is a guess; thirty consecutive
# frames at 0.15 is a box. The net evidence required does not fall -- it changes shape, from
# instantaneous confidence to confidence over time, which is the evidence this detector can
# actually supply.
SCORE_SELECT = 0.12       # eligible as a target, WITH the confirmation count below
SCORE_PICKUP = 0.20       # required again, immediately before closing on it
MIN_CONFIRMATIONS = 4     # sightings of the same track before it may be selected at all
PERCEPTION_FRESH_S = 1.8  # s; must clear rect cadence + detector latency with room to spare
DEPTH_OK = frozenset({"surface_supported"})   # has real depth AND rests on a measured plane      # depth returns needed before "nothing there" means empty rather than blind


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
        for item in getattr(scan, "objects", ()) or ():
            base = item.get("position_base_m")
            if base is None or not item.get("current"):
                continue
            if all(abs(base[i] - centre[i]) <= reach for i in range(3)):
                return Holding("holding", item.get("id"), scan.ts, "carry-volume-detector")

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
