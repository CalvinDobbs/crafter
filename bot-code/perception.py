# /// script
# dependencies = [
#   "bbos",
#   "numpy",
#   "opencv-python-headless",
#   "fastapi",
#   "uvicorn",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Read-only perception and a session-local, surrounding observation grid.

Only this module and fixtures/scan_sample.json belong to perception. No motor
Writers, HTTP motion calls, agent state, or shared-contract edits belong here.
Use PerceptionSession for persistent memory; scan() is a one-shot convenience.

Frames: base is +x forward, +y left, +z up, meters. Session world starts at the
first wheel sample. Wheel distances and optional IMU yaw increments estimate
T_world_base; this is drifting planar dead reckoning, NOT SLAM or a navigation
safety map. Gaps/reset jumps invalidate the map rather than mixing frames.
Depth points already use base coordinates; do not apply IMU pitch twice.

camera.rect.left and camera.points share acquisition timestamps. Sparse points
map back to the rectified image through idx_2d[:num_points], not reshape().
Anchor 49 is reserved in DICT_4X4_50. Its 3D plane supplies orientation as well
as position. Local column/row axes follow printed marker right/up. The build
origin is offset from the marker, leaving the marker outside the build area.
Missing anchor means UNKNOWN classification, never a fixed-origin fallback.

Scan.boxes contains current loose candidates, not a guarantee of graspability.
Scan.protected contains build-zone observations, not confirmed placements.
Scan.stacked is a deprecated protected alias for callers of the old scaffold.
Unseen tracks are remembered with age, never offered as fresh pick candidates.
verify_pick returns False for an unmoved box and None for ambiguous/missing
observations; disappearance alone cannot prove a grasp. verify_place reports
geometric consistency only, not successful execution of a motor command.

Run from anywhere: uv run /home/bracketbot/crafter/bot-code/perception.py --viz --mock
Live: --viz --pose-source wheel-imu; use --pose-source wheel to exclude IMU yaw.
--self-test runs synthetic regressions without bbos. /scan is a versioned JSON
snapshot; /frame is the matching rectified camera view. All sensor reads occur
in one worker, not concurrent web handlers. Mock buttons never move hardware.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from contracts import Detection, BOX_SIZE, CELL

FIXTURE = Path(__file__).parent / "fixtures" / "scan_sample.json"
ARUCO_DICT = "DICT_4X4_50"   # marker id == box id; ANCHOR_ID marks the grid
ANCHOR_ID = 49
FOOTPRINT = 3               # default build grid cells per side; not the surrounding map
GRID_MARGIN = 0.04          # m slack when classifying in/out of footprint
HEIGHT_DISK_R = 0.02        # m radius for column-height sampling
HEIGHT_TOL = BOX_SIZE * 0.25 # verify_place z tolerance
MIN_MASK_PTS = 8


@dataclass
class Settings:
    box_size: float = BOX_SIZE
    cell: float = CELL
    build_cols: int = FOOTPRINT
    build_rows: int = FOOTPRINT
    anchor_offset: tuple = (0.12, 0.12)
    resolution: float = 0.1
    radius: float = 2.0
    memory_s: float = 30.0
    anchor_ttl: float = 15.0
    fresh_s: float = 0.8

    def __post_init__(self):
        positive = (self.box_size, self.cell, self.resolution, self.radius,
                    self.memory_s, self.anchor_ttl, self.fresh_s)
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("sizes, radii and timeouts must be finite and positive")
        if not (1 <= self.build_cols <= 32 and 1 <= self.build_rows <= 32):
            raise ValueError("build dimensions must be in 1..32")
        if self.cell < self.box_size or len(self.anchor_offset) != 2:
            raise ValueError("cell pitch must fit the box; anchor offset needs two values")
        if not np.isfinite(self.anchor_offset).all():
            raise ValueError("anchor offset must be finite")


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    ts: float = 0.0
    source: str = "unavailable"
    valid: bool = False
    epoch: int = 0
    warning: str = ""

    def rotation(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def to_world(self, points):
        return np.asarray(points, dtype=float) @ self.rotation().T + [self.x, self.y, 0]

    def to_base(self, points):
        return (np.asarray(points, dtype=float) - [self.x, self.y, 0]) @ self.rotation()


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class Odometry:
    def __init__(self, diameter=0.165, width=0.3275, source="wheel-imu"):
        if source not in ("wheel", "wheel-imu") or min(diameter, width) <= 0:
            raise ValueError("invalid odometry configuration")
        self.diameter, self.width, self.source = diameter, width, source
        self.pose = Pose(source=source)
        self.previous = None
        self.history = deque(maxlen=500)

    def update(self, ts, turns, imu_yaw=None):
        turns = np.asarray(turns, dtype=float)
        if turns.shape != (2,) or not np.isfinite(turns).all() or not math.isfinite(ts):
            return self.invalidate("invalid wheel sample")
        if self.source == "wheel-imu" and (imu_yaw is None or not math.isfinite(imu_yaw)):
            return self.invalidate("IMU stale; select wheel mode explicitly if needed")
        if self.previous is None:
            self.previous = (ts, turns.copy(), imu_yaw)
            self.pose.ts, self.pose.valid, self.pose.warning = ts, True, ""
            self.history.append(Pose(**asdict(self.pose)))
            return self.pose
        old_ts, old_turns, old_yaw = self.previous
        if ts <= old_ts:
            return self.invalidate("wheel timestamp reset")
        dt = ts - old_ts
        dl, dr = (turns - old_turns) * math.pi * self.diameter
        wheel_yaw = (dr - dl) / self.width
        dyaw = wheel_yaw
        if self.source == "wheel-imu":
            imu_delta = _wrap(imu_yaw - old_yaw)
            if abs(_wrap(imu_delta - wheel_yaw)) > 0.12 + 0.5 * dt:
                return self.invalidate("IMU/wheel yaw disagreement or reset")
            dyaw = 0.8 * imu_delta + 0.2 * wheel_yaw
        if dt > 0.5 or max(abs(dl), abs(dr)) > 0.015 + 0.8 * dt:
            return self.invalidate("wheel gap/jump; starting a new map epoch")
        distance = (dl + dr) / 2
        mid_yaw = self.pose.yaw + dyaw / 2
        distance *= np.sinc(dyaw / (2 * math.pi))
        self.pose.x += distance * math.cos(mid_yaw)
        self.pose.y += distance * math.sin(mid_yaw)
        self.pose.yaw = _wrap(self.pose.yaw + dyaw)
        self.pose.ts, self.pose.valid, self.pose.warning = ts, True, ""
        self.previous = (ts, turns.copy(), imu_yaw)
        self.history.append(Pose(**asdict(self.pose)))
        return self.pose

    def invalidate(self, reason):
        if self.previous is not None or self.pose.valid:
            self.pose = Pose(source=self.source, epoch=self.pose.epoch + 1, warning=reason)
            self.history.clear()
        self.pose.valid, self.pose.warning = False, reason
        self.previous = None
        return self.pose

    def at(self, ts):
        if not self.history or not self.pose.valid:
            return Pose(ts=ts, epoch=self.pose.epoch, warning=self.pose.warning)
        samples = list(self.history)
        for a, b in zip(samples, samples[1:]):
            if a.ts <= ts <= b.ts and b.ts - a.ts <= 0.2:
                t = (ts - a.ts) / (b.ts - a.ts)
                return Pose(a.x + t * (b.x-a.x), a.y + t * (b.y-a.y),
                            _wrap(a.yaw + t * _wrap(b.yaw-a.yaw)), ts,
                            self.source, True, self.pose.epoch)
        nearest = min(samples, key=lambda p: abs(p.ts-ts))
        if abs(nearest.ts-ts) <= 0.06:
            return Pose(**asdict(nearest))
        return Pose(ts=ts, epoch=self.pose.epoch, warning="no pose aligned to camera timestamp")


@dataclass
class BuildFrame:
    origin: list
    col: list
    row: list
    marker: list
    ts: float

    def __post_init__(self):
        values = np.asarray([self.origin, self.col, self.row, self.marker], dtype=float)
        if values.shape != (4, 3) or not np.isfinite(values).all():
            raise ValueError("build pose needs finite 3D vectors")
        if not np.allclose([np.linalg.norm(self.col), np.linalg.norm(self.row)], 1, atol=0.01):
            raise ValueError("build axes must be unit vectors")
        if abs(np.dot(self.col, self.row)) > 0.01:
            raise ValueError("build axes must be perpendicular")
        if np.cross(self.col, self.row)[2] < 0.9:
            raise ValueError("build anchor must lie on an approximately horizontal surface")

    def transformed(self, pose, inverse=False):
        point_fn = pose.to_base if inverse else pose.to_world
        rotation = pose.rotation().T if inverse else pose.rotation()
        return BuildFrame(point_fn(self.origin).tolist(),
                          (rotation @ self.col).tolist(), (rotation @ self.row).tolist(),
                          point_fn(self.marker).tolist(), self.ts)


@dataclass
class Scan:
    boxes: list[Detection] = field(default_factory=list)    # current loose candidates
    stacked: list[Detection] = field(default_factory=list)  # deprecated alias: protected, NOT placed
    anchor: list[float] | None = None
    ts: float = 0.0
    protected: list[Detection] = field(default_factory=list)
    unknown: list[Detection] = field(default_factory=list)
    build: BuildFrame | None = None
    pose: Pose = field(default_factory=Pose)
    tracks: list = field(default_factory=list)
    surface_cells: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    anchor_seen: bool = False
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)), repr=False)


# ---- grid math --------------------------------------------------------------

def resolve_anchor(scan_anchor):
    """Require a measured pose. Never invent a grid from a missing anchor."""
    if not isinstance(scan_anchor, BuildFrame):
        raise ValueError("a BuildFrame with orientation is required; an xyz anchor is insufficient")
    return scan_anchor


def cell_center(x: int, y_layer: int, z: int, anchor_pos, settings=None) -> list[float]:
    a, cfg = resolve_anchor(anchor_pos), settings or Settings()
    up = np.cross(a.col, a.row)
    return (np.asarray(a.origin) + np.asarray(a.col) * x * cfg.cell
            + np.asarray(a.row) * z * cfg.cell
            + up * (y_layer * cfg.box_size + cfg.box_size / 2)).tolist()


def _grid_uv(pos, anchor_pos, settings=None) -> tuple[float, float]:
    a, cfg = resolve_anchor(anchor_pos), settings or Settings()
    d = np.asarray(pos, dtype=float) - a.origin
    return float(np.dot(d, a.col) / cfg.cell), float(np.dot(d, a.row) / cfg.cell)


def is_in_grid(pos, anchor_pos, margin=GRID_MARGIN, settings=None) -> bool:
    cfg = settings or Settings()
    u, v = _grid_uv(pos, anchor_pos, cfg)
    m = margin / cfg.cell
    return (-0.5-m <= u <= cfg.build_cols-0.5+m
            and -0.5-m <= v <= cfg.build_rows-0.5+m)


# ---- detection core ---------------------------------------------------------

def _detect_aruco(rgb: np.ndarray):
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT)),
        cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    return [] if ids is None else [(int(mid), c[0]) for mid, c in zip(ids.flatten(), corners)]


def _points_for_mask(mask, points, idx_2d=None):
    points = np.asarray(points, dtype=float)
    if points.ndim == 3:
        if points.shape[:2] != mask.shape:
            raise ValueError("organized cloud and rectified mask dimensions differ")
        return points[mask]
    if points.ndim != 2 or points.shape[1] != 3 or idx_2d is None:
        raise ValueError("sparse clouds require idx_2d, not reshape")
    ids = np.asarray(idx_2d)
    if ids.shape != (len(points),) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("invalid sparse pixel index array")
    valid = (ids >= 0) & (ids < mask.size)
    return points[valid][mask.ravel()[ids[valid]]]


def _mask_positions(mask, points, idx_2d=None):
    pts = _points_for_mask(mask, points, idx_2d)
    pts = pts[np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 1e-3)]
    return np.median(pts, axis=0) if len(pts) >= MIN_MASK_PTS else None


def _marker_plane(corners, points, idx_2d, image_shape):
    """Fit the printed marker plane using its OWN valid depth samples."""
    import cv2
    h, w = image_shape[:2]
    ids = np.asarray(idx_2d)
    valid = ((ids >= 0) & (ids < h*w) & np.isfinite(points).all(axis=1)
             & (np.linalg.norm(points, axis=1) > 1e-3))
    pixels = np.column_stack((ids[valid] % w, ids[valid] // w)).astype(np.float32)
    transform = cv2.getPerspectiveTransform(np.asarray(corners, np.float32),
                                            np.array([[-.5,.5],[.5,.5],[.5,-.5],[-.5,-.5]], np.float32))
    if not len(pixels):
        return None
    uv = cv2.perspectiveTransform(pixels[:, None, :], transform)[:, 0]
    inside = np.isfinite(uv).all(axis=1) & (np.max(np.abs(uv), axis=1) < 0.4)
    pts, uv = points[valid][inside], uv[inside]
    if len(pts) < MIN_MASK_PTS or np.min(np.ptp(uv, axis=0)) < .3:
        return None
    a = np.column_stack((uv, np.ones(len(uv))))
    coeff, _, rank, _ = np.linalg.lstsq(a, pts, rcond=None)
    if rank < 3:
        return None
    residual = np.linalg.norm(a @ coeff - pts, axis=1)
    good = residual < max(.006, min(.02, 2.5 * float(np.median(residual))))
    if good.sum() < MIN_MASK_PTS:
        return None
    coeff, _, rank, _ = np.linalg.lstsq(a[good], pts[good], rcond=None)
    lengths = np.linalg.norm(coeff[:2], axis=1)
    if rank < 3 or min(lengths) < .008 or max(lengths)/min(lengths) > 1.6:
        return None
    if np.percentile(np.linalg.norm(a[good] @ coeff - pts[good], axis=1), 90) > .015:
        return None
    col = coeff[0] / lengths[0]
    row = coeff[1] - col * np.dot(col, coeff[1])
    if np.linalg.norm(row) < .008:
        return None
    return coeff[2], col, row / np.linalg.norm(row)


def observe(rgb, points, idx_2d, ts, settings, camera_origin):
    observations, anchor, warnings = [], None, []
    markers = _detect_aruco(rgb)
    counts = {mid: sum(other == mid for other, _ in markers) for mid, _ in markers}
    for mid, corners in markers:
        if counts[mid] != 1:
            warnings.append(f"duplicate marker {mid}; identity ambiguous")
            continue
        plane = _marker_plane(corners, points, idx_2d, rgb.shape)
        if plane is None:
            warnings.append(f"marker {mid}: insufficient/invalid depth; no metric pose")
            continue
        center, col, row = plane
        if mid == ANCHOR_ID:
            origin = center + col * settings.anchor_offset[0] + row * settings.anchor_offset[1]
            try:
                anchor = BuildFrame(origin.tolist(), col.tolist(), row.tolist(), center.tolist(), ts)
            except ValueError as e:
                warnings.append(f"anchor rejected: {e}")
        else:
            normal = np.cross(col, row)
            if np.dot(normal, center - camera_origin) < 0:
                normal *= -1
            box_center = center + normal * settings.box_size / 2
            observations.append(Detection(mid, box_center.tolist(), size=settings.box_size))
    return observations, anchor, warnings, markers


class WorldModel:
    def __init__(self, settings=None):
        self.cfg = settings or Settings()
        self.epoch = None
        self.tracks, self.surfaces = {}, {}
        self.anchor = None

    def update(self, observations, pose, ts, anchor=None, points=None, warnings=()):
        cfg = self.cfg
        s = Scan(ts=ts, pose=pose, warnings=list(warnings))
        if points is not None:
            s.points = np.asarray(points, dtype=float).copy()
        if not pose.valid:
            s.unknown = list(observations)
            s.warnings.append(pose.warning or "pose unavailable; no persistent mapping")
            return s
        if self.epoch != pose.epoch:
            self.tracks.clear()
            self.surfaces.clear()
            self.anchor = None
            self.epoch = pose.epoch
        if anchor is not None:
            self.anchor = anchor.transformed(pose)
            s.anchor_seen = True
        if self.anchor and ts - self.anchor.ts <= cfg.anchor_ttl:
            s.build = self.anchor.transformed(pose, inverse=True)
            s.anchor = s.build.marker
        else:
            s.warnings.append("build anchor unavailable/stale; boxes are UNKNOWN, not pick candidates")
        by_id = {}
        for d in observations:
            p = np.asarray(d.pos, dtype=float)
            if p.shape != (3,) or not np.isfinite(p).all() or not 0 <= d.id < ANCHOR_ID:
                s.warnings.append("invalid box observation discarded")
                continue
            if d.id in by_id:
                by_id[d.id] = None
            else:
                by_id[d.id] = d
        for mid, d in by_id.items():
            if d is not None:
                self.tracks[mid] = (pose.to_world(d.pos), ts, d.size)
        for mid, (world, seen, size) in list(self.tracks.items()):
            age = ts-seen
            base = pose.to_base(world)
            if age > cfg.memory_s or np.linalg.norm(base[:2]) > cfg.radius * 1.5:
                del self.tracks[mid]
                continue
            current = mid in by_id and by_id[mid] is not None
            label = "unknown" if s.build is None else (
                "protected" if is_in_grid(base, s.build, settings=cfg) else "loose")
            s.tracks.append({"id": mid, "world": world.tolist(), "pos": base.tolist(),
                             "age": age, "current": current, "classification": label,
                             "size": size, "last_seen": seen})
            if current:
                d = Detection(mid, base.tolist(), size=size)
                getattr(s, {"loose": "boxes", "protected": "protected", "unknown": "unknown"}[label]).append(d)
        s.stacked = list(s.protected)
        if len(s.points):
            world = pose.to_world(s.points[::4])
            valid = np.isfinite(world).all(axis=1) & (np.linalg.norm(world[:, :2]-[pose.x, pose.y], axis=1) <= cfg.radius)
            world = world[valid]
            if len(world):
                keys, inv = np.unique(np.floor(world[:, :2]/cfg.resolution).astype(int), axis=0, return_inverse=True)
                lo, hi = np.full(len(keys), np.inf), np.full(len(keys), -np.inf)
                np.minimum.at(lo, inv, world[:, 2])
                np.maximum.at(hi, inv, world[:, 2])
                for key, low, high in zip(keys, lo, hi):
                    self.surfaces[tuple(key)] = (float(low), float(high), ts)
        for key, (low, high, seen) in list(self.surfaces.items()):
            world = [(key[0]+.5)*cfg.resolution, (key[1]+.5)*cfg.resolution, high]
            base = pose.to_base(world)
            if ts-seen > cfg.memory_s or np.linalg.norm(base[:2]) > cfg.radius*1.5:
                del self.surfaces[key]
                continue
            s.surface_cells.append({"world": world, "pos": base.tolist(), "z_min": low,
                                    "z_max": high, "age": ts-seen})
        return s


class LiveSource:
    def __init__(self, source="wheel-imu"):
        from bbos import Reader, Config
        self.stack = contextlib.ExitStack()
        self.points = self.stack.enter_context(Reader("camera.points", keeptime=False))
        self.rect = self.stack.enter_context(Reader("camera.rect", keeptime=False, aligned_to=self.points))
        self.wheels = self.stack.enter_context(Reader("drive.state", keeptime=False))
        self.imu = self.stack.enter_context(Reader("imu.orientation", keeptime=False))
        drive = Config("drive")
        self.odom = Odometry(drive.wheel_diam, drive.robot_width, source)
        self.camera_origin = np.asarray(Config("depth").camera_to_base_3x4)[:, 3]
        self.imu_sample = None
        self.last_frame = 0.0

    def poll(self):
        if self.imu.ready():
            d = self.imu.data
            self.imu_sample = (_stamp(d), math.radians(float(d["rpy"][2])))
        if self.wheels.ready():
            d, yaw = self.wheels.data, None
            ts = _stamp(d)
            if self.imu_sample and abs(ts-self.imu_sample[0]) < .15:
                yaw = self.imu_sample[1]
            self.odom.update(ts, np.array(d["pos"], dtype=float), yaw)
        if self.odom.pose.valid and time.time()-self.odom.pose.ts > .5:
            self.odom.invalidate("wheel stream stale")
        if not self.points.ready():
            return None
        d = self.points.data
        ts = _stamp(d)
        self.rect.ready()
        rgb = self.rect.data
        if rgb is None or abs(_stamp(rgb)-ts) > .001:
            return None
        if ts <= self.last_frame or abs(time.time()-ts) > .8:
            return None
        self.last_frame = ts
        n = int(d["num_points"])
        if not 0 <= n <= len(d["points"]):
            raise ValueError("invalid depth point count")
        return (np.array(rgb["left"], copy=True), np.array(d["points"][:n], dtype=float, copy=True),
                np.array(d["idx_2d"][:n], copy=True), ts, self.odom.at(ts))

    def close(self):
        self.stack.close()


def _stamp(data):
    return int(data["timestamp"].astype("datetime64[ns]").astype(np.int64)) / 1e9


# ---- public API -------------------------------------------------------------

class PerceptionSession:
    def __init__(self, mock=False, settings=None, pose_source="wheel-imu"):
        self.mock, self.settings = mock, settings or Settings()
        self.world = WorldModel(self.settings)
        self.source = None if mock else LiveSource(pose_source)
        self.scene = MockSource(self.settings) if mock else None
        self.latest = Scan()
        self.jpeg = b""

    def poll(self):
        if self.mock:
            observations, anchor, points, ts, pose = self.scene.next()
            warnings = ["SIMULATED observations; not camera detection. Same world classifier as live."]
        else:
            data = self.source.poll()
            if data is None:
                return None
            rgb, points, indices, ts, pose = data
            observations, anchor, warnings, markers = observe(
                rgb, points, indices, ts, self.settings, self.source.camera_origin)
            import cv2
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for mid, corners in markers:
                cv2.polylines(image, [corners.astype(np.int32)], True, (0, 210, 255), 1)
                cv2.putText(image, str(mid), tuple(corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,210,255), 1)
            ok, encoded = cv2.imencode(".jpg", image)
            self.jpeg = encoded.tobytes() if ok else b""
        self.latest = self.world.update(observations, pose, ts, anchor, points, warnings)
        return self.latest

    def close(self):
        if self.source:
            self.source.close()


def scan(mock=False, *, timeout=4.0, settings=None, pose_source="wheel-imu") -> Scan:
    session = PerceptionSession(mock, settings, pose_source)
    try:
        until = time.monotonic()+timeout
        while time.monotonic() < until:
            s = session.poll()
            if s is not None and (mock or s.pose.valid):
                return s
            time.sleep(.005)
        raise TimeoutError("no synchronized depth and valid wheel/IMU pose; no map was fabricated")
    finally:
        session.close()


def detect_boxes(mock=False) -> list[Detection]:
    s = scan(mock)
    return s.boxes + s.protected + s.unknown


def scan_all(skills=None, mock=False, sweeps=1) -> list[Detection]:
    """Legacy signature, but perception must never command a physical sweep."""
    if sweeps != 1:
        raise ValueError("perception is read-only; the motion owner must sweep and re-observe via PerceptionSession")
    return scan(mock).boxes


# ---- measured-world helpers -------------------------------------------------

def _height_at(points: np.ndarray, xy, radius=HEIGHT_DISK_R) -> float | None:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 1e-3)
    pts = pts[valid]
    selected = pts[np.linalg.norm(pts[:, :2]-np.asarray(xy)[:2], axis=1) < radius]
    return float(np.percentile(selected[:, 2], 90)) if len(selected) >= MIN_MASK_PTS else None


def column_heights(structure_cells, anchor_pos=None, mock=False, *, snapshot=None, settings=None) -> dict:
    s = snapshot if snapshot is not None else scan(mock)
    cfg = settings or Settings()
    if s.build is None or time.time()-s.ts > cfg.fresh_s:
        return {tuple(c): None for c in structure_cells}
    return {tuple(c): _height_at(s.points, cell_center(*c, s.build, cfg), cfg.box_size*.35)
            for c in structure_cells}


def verify_pick(box: Detection, mock=False, *, before=None, after=None) -> bool | None:
    """Missing/moved/occluded is UNKNOWN. This API alone cannot prove a grasp."""
    if before is None:
        return None
    s = after if after is not None else scan(mock)
    if not s.pose.valid or not before.pose.valid or s.pose.epoch != before.pose.epoch or s.ts <= before.ts:
        return None
    if time.time()-s.ts > Settings().fresh_s:
        return None
    for d in s.boxes+s.protected+s.unknown:
        if d.id == box.id:
            distance = np.linalg.norm(s.pose.to_world(d.pos)-before.pose.to_world(box.pos))
            if distance <= GRID_MARGIN:
                return False
    return None


def verify_place(cell, anchor_pos=None, expected_top=None, mock=False, *, snapshot=None, settings=None) -> bool | None:
    s = snapshot if snapshot is not None else scan(mock)
    cfg = settings or Settings()
    if s.build is None or not s.pose.valid:
        return None
    expected = expected_top if expected_top is not None else cell_center(*cell, s.build, cfg)[2]+cfg.box_size/2
    h = column_heights([cell], snapshot=s, settings=cfg).get(tuple(cell))
    return None if h is None else abs(h-expected) <= cfg.box_size*.25


# ---- fixtures ---------------------------------------------------------------

class MockSource:
    def __init__(self, settings):
        self.settings = settings
        self.fx = json.loads(FIXTURE.read_text())
        self.pose = Pose(source="mock", valid=True)
        self.all_visible = False
        self.anchor_enabled = True
        self.pose_enabled = True
        self.step = int(self.fx.get("initial_step", 0))

    def command(self, action):
        if action == "turn_left":
            self.pose.yaw = _wrap(self.pose.yaw + math.pi/4)
        elif action == "turn_right":
            self.pose.yaw = _wrap(self.pose.yaw - math.pi/4)
        elif action in ("forward", "back"):
            d = .15 if action == "forward" else -.15
            self.pose.x += d*math.cos(self.pose.yaw)
            self.pose.y += d*math.sin(self.pose.yaw)
        elif action == "visibility":
            self.all_visible = not self.all_visible
        elif action == "anchor":
            self.anchor_enabled = not self.anchor_enabled
        elif action == "pose":
            self.pose_enabled = not self.pose_enabled
            self.pose.epoch += 1
        elif action == "stack_next":
            self.step = min(self.step+1, len(self.fx["stack_sequence"]))
        elif action == "reset":
            self.__init__(self.settings)
            self.pose.epoch = int(time.time()*1000)
        else:
            raise ValueError("unknown mock action")

    def next(self):
        ts = time.time()
        self.pose.ts, self.pose.valid = ts, self.pose_enabled
        self.pose.warning = "" if self.pose_enabled else "mock pose loss: no candidates"
        pose = Pose(**asdict(self.pose))
        cfg = self.settings
        a = self.fx["anchor"]
        marker = np.asarray(a["world"], dtype=float)
        col, row = np.asarray(a["col"], dtype=float), np.asarray(a["row"], dtype=float)
        origin = marker + col*cfg.anchor_offset[0]+row*cfg.anchor_offset[1]
        world_anchor = BuildFrame(origin.tolist(), col.tolist(), row.tolist(), marker.tolist(), ts)
        def visible(world):
            p = pose.to_base(world)
            return self.all_visible or (np.linalg.norm(p[:2]) < cfg.radius
                                       and abs(math.atan2(p[1], p[0])) < math.radians(48))
        anchor = world_anchor.transformed(pose, inverse=True) if self.anchor_enabled and visible(marker) else None
        observations, surfaces = [], []
        for b in self.fx["observations"]:
            world = np.array(b["world"], dtype=float)
            sequence = self.fx["stack_sequence"][:self.step]
            if b["id"] in sequence:
                layer = sequence.index(b["id"])
                world = np.asarray(cell_center(0, layer, 0, world_anchor, cfg))
            if visible(world):
                observations.append(Detection(b["id"], pose.to_base(world).tolist(), size=cfg.box_size))
                for u in np.linspace(-.4, .4, 7):
                    for v in np.linspace(-.4, .4, 7):
                        surfaces.append(pose.to_base(world + [u*cfg.box_size, v*cfg.box_size, cfg.box_size/2]))
        return observations, anchor, np.asarray(surfaces).reshape(-1, 3), ts, pose


def scan_to_dict(s, settings=None, mock=False):
    cfg = settings or Settings()
    now = time.time()
    stale = not s.ts or now-s.ts > cfg.fresh_s
    build = None
    if s.build:
        build = asdict(s.build)
        build["age"] = max(0, now-s.build.ts)
        build["valid"] = not stale and build["age"] <= cfg.anchor_ttl and s.pose.valid
        build["cells"] = [cell_center(x, 0, z, s.build, cfg) for x in range(cfg.build_cols) for z in range(cfg.build_rows)]
    tracks = []
    for track in s.tracks:
        t = dict(track)
        t["age"] = max(0, now-t["last_seen"])
        t["current"] = t["current"] and not stale
        t["pick_candidate"] = bool(t["current"] and t["classification"] == "loose" and build and build["valid"])
        tracks.append(t)
    return {"schema_version": 2, "mock": mock, "ts": s.ts, "stale": stale,
            "frame": "base_at_capture", "pose": asdict(s.pose), "build": build,
            "anchor_seen": s.anchor_seen, "tracks": tracks,
            "boxes": [asdict(d) for d in s.boxes] if not stale else [],
            "protected": [asdict(d) for d in s.protected], "unknown": [asdict(d) for d in s.unknown],
            "surface_cells": s.surface_cells, "settings": asdict(cfg),
            "warnings": s.warnings + (["snapshot stale; no pick candidates"] if stale else []),
            "map_semantics": "observed surfaces only; blank cells UNKNOWN, not free; not a navigation map"}


# ---- debug visualizer ---------------------------------------------------------

def serve_viz(port=8007, mock=False, settings=None, pose_source="wheel-imu", host="127.0.0.1"):
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import HTMLResponse
    import uvicorn
    cfg = settings or Settings()
    lock, stop = threading.Lock(), threading.Event()
    shared = {"scan": Scan(), "jpeg": b"", "error": "waiting for sensors", "actions": deque()}

    def worker():
        session = None
        try:
            session = PerceptionSession(mock, cfg, pose_source)
            while not stop.is_set():
                with lock:
                    actions = list(shared["actions"])
                    shared["actions"].clear()
                for action in actions:
                    session.scene.command(action)
                result = session.poll()
                if result is not None:
                    with lock:
                        shared.update(scan=result, jpeg=session.jpeg, error="")
                stop.wait(.2 if mock else .005)
        except Exception as e:
            with lock:
                shared["error"] = f"{type(e).__name__}: {e}"
        finally:
            if session:
                session.close()

    @contextlib.asynccontextmanager
    async def lifespan(app):
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=3)

    app = FastAPI(lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _VIZ_PAGE

    @app.get("/scan")
    def scan_json():
        with lock:
            out = scan_to_dict(shared["scan"], cfg, mock)
            if shared["error"]:
                out["warnings"].append(shared["error"])
            return out

    @app.get("/frame")
    def frame():
        with lock:
            if mock:
                return Response(status_code=204)
            if not shared["jpeg"] or time.time()-shared["scan"].ts > cfg.fresh_s:
                return Response(status_code=503)
            return Response(shared["jpeg"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/mock/{action}")
    def mock_action(action: str):
        if not mock:
            raise HTTPException(403, "mock controls disabled on live robot")
        if action not in {"turn_left", "turn_right", "forward", "back", "visibility", "anchor", "pose", "stack_next", "reset"}:
            raise HTTPException(400, "unknown action")
        with lock:
            if len(shared["actions"]) >= 16:
                raise HTTPException(429, "mock command queue full")
            shared["actions"].append(action)
        return {"ok": True, "mock_only": True}

    print(f"[viz] http://{host}:{port} mock={mock} pose={pose_source}; read-only sensors", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")


_VIZ_PAGE = r"""<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crafter Perception</title>
<style>
*{box-sizing:border-box}body{margin:0;padding:18px;background:#0b111b;color:#dbe5f5;font:14px system-ui,sans-serif}h1{font-size:22px;margin:0 0 5px}p{color:#91a6c3}button,select{background:#1b2a40;color:#eaf1ff;border:1px solid #3c506d;border-radius:5px;padding:8px;cursor:pointer}header{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}.layout{display:grid;grid-template-columns:minmax(480px,2fr) minmax(280px,1fr);gap:16px;margin-top:12px}section{background:#111d2c;border:1px solid #2c3b52;border-radius:8px;padding:12px}canvas{display:block;width:100%;background:#0c1522}#side{max-height:260px}#cam{width:100%;margin-top:10px}#warning{color:#ffcf80;white-space:pre-wrap}#info{white-space:pre-wrap;font:12px monospace}table{width:100%;border-collapse:collapse;font:12px monospace}td,th{padding:7px 3px;text-align:left;border-bottom:1px solid #29384e}#mock{display:none;margin-top:10px;gap:6px;flex-wrap:wrap}.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;margin:10px 0}.loose{color:#58dfb6}.protected{color:#ffb85c}.unknown{color:#c8a0ef}.muted{color:#8c9aaf}.status{font:13px monospace}@media(max-width:850px){.layout{grid-template-columns:1fr}}a{color:#92c5ff}
</style></head><body>
<header><div><h1>Crafter / perception</h1><p>Surrounding observation grid. Empty space is unknown, not confirmed free.</p></div><div><select id="view"><option value="robot">Robot-centered, heading up</option><option value="world">Session-world view</option></select><select id="range"><option value="2">2 m radius</option><option value="1">1 m radius</option><option value="3">3 m radius</option></select></div></header>
<div class="status" id="status">Connecting...</div>
<div id="mock"><button data-action="turn_left">Turn +45°</button><button data-action="turn_right">Turn -45°</button><button data-action="forward">Forward 15 cm</button><button data-action="back">Back 15 cm</button><button data-action="visibility">Toggle all / limited visibility</button><button data-action="anchor">Toggle anchor</button><button data-action="pose">Toggle pose loss</button><button data-action="stack_next">Simulate next placement</button><button data-action="reset">Reset mock</button></div>
<div class="layout"><section><div class="legend"><span class="loose">Green: loose observation</span><span class="protected">Amber: protected build zone</span><span class="unknown">Purple: unknown</span><span class="muted">Dashed: remembered, NOT pickable</span></div><canvas id="map" width="800" height="800"></canvas><p>Blue cells: measured surfaces (height shown by intensity). No collision-free or grasp-reach claims.</p></section><section><h3>Box heights / base-frame side view</h3><canvas id="side" width="600" height="240"></canvas><p id="cameraLabel">Waiting for mode...</p><img id="cam" hidden alt="Rectified camera frame with marker outlines"><div id="warning"></div><pre id="info"></pre><table><thead><tr><th>ID</th><th>Class</th><th>Age</th><th>Base xyz (m)</th></tr></thead><tbody id="rows"></tbody></table></section></div>
<script>
const cv=document.getElementById('map'),ctx=cv.getContext('2d');
const side=document.getElementById('side'),sc=side.getContext('2d');
let data=null;
function framePoint(p,d){if(document.getElementById('view').value==='robot')return p;const c=Math.cos(d.pose.yaw),s=Math.sin(d.pose.yaw);return [d.pose.x+c*p[0]-s*p[1],d.pose.y+s*p[0]+c*p[1],p[2]];}
function px(p,d){const q=framePoint(p,d),scale=360/Number(document.getElementById('range').value);return [400-q[1]*scale,400-q[0]*scale];}
function path(points,d,close=false){ctx.beginPath();points.forEach((p,i)=>{const q=px(p,d);i?ctx.lineTo(...q):ctx.moveTo(...q)});if(close)ctx.closePath();}
function dot(p,c,r,label,d){ctx.fillStyle=c;ctx.beginPath();ctx.arc(...px(p,d),r,0,Math.PI*2);ctx.fill();if(label){ctx.fillStyle='#e8f0ff';ctx.fillText(label,px(p,d)[0]+9,px(p,d)[1]-8);}}
function gridSquare(center,col,row,half,d){path([[1,1],[-1,1],[-1,-1],[1,-1]].map(([u,v])=>center.map((n,i)=>n+u*half*col[i]+v*half*row[i])),d,true);}
function draw(d){
ctx.clearRect(0,0,800,800);ctx.font='12px monospace';const radius=Number(document.getElementById('range').value),scale=360/radius;
ctx.strokeStyle='#1d2e43';ctx.lineWidth=1;
for(let t=-radius;t<=radius+.001;t+=d.settings.resolution){ctx.beginPath();ctx.moveTo(400+t*scale,40);ctx.lineTo(400+t*scale,760);ctx.stroke();ctx.beginPath();ctx.moveTo(40,400+t*scale);ctx.lineTo(760,400+t*scale);ctx.stroke();}
for(const c of d.surface_cells){ctx.fillStyle=`rgba(49,130,188,${Math.max(.07,.28*(1-c.age/d.settings.memory_s))})`;const yaw=d.pose.yaw,co=Math.cos(yaw),si=Math.sin(yaw);gridSquare(c.pos,[co,-si,0],[si,co,0],d.settings.resolution/2,d);ctx.fill();}
if(d.build){ctx.strokeStyle=d.build.valid?'#ffb85c':'#755f48';for(const c of d.build.cells){gridSquare(c,d.build.col,d.build.row,d.settings.cell/2,d);ctx.stroke();}dot(d.build.marker,'#ff7373',5,'anchor 49',d);ctx.strokeStyle='#ef6479';path([d.build.origin,d.build.origin.map((v,i)=>v+.2*d.build.col[i])],d);ctx.stroke();ctx.strokeStyle='#73d497';path([d.build.origin,d.build.origin.map((v,i)=>v+.2*d.build.row[i])],d);ctx.stroke();}
ctx.strokeStyle='#3c5b7e';ctx.setLineDash([6,7]);path([[1.5*Math.cos(.838),1.5*Math.sin(.838),0],[0,0,0],[1.5*Math.cos(.838),-1.5*Math.sin(.838),0]],d);ctx.stroke();ctx.setLineDash([]);
const grouped=new Map();for(const t of d.tracks){const p=px(t.pos,d),k=p.map(v=>Math.round(v/15)).join(',');if(!grouped.has(k))grouped.set(k,[]);grouped.get(k).push(t);}
for(const group of grouped.values()){const t=group[0],color={loose:'#58dfb6',protected:'#ffb85c',unknown:'#c8a0ef'}[t.classification];ctx.globalAlpha=t.current?1:.45;ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(t.current?[]:[3,3]);const p=px(t.pos,d);ctx.beginPath();ctx.arc(...p,8,0,Math.PI*2);ctx.stroke();if(t.current)dot(t.pos,color,5,null,d);ctx.fillStyle='#e8f0ff';ctx.fillText(group.map(b=>'#'+b.id).join(' / '),p[0]+11,p[1]-10);ctx.globalAlpha=1;ctx.setLineDash([]);}
// robot
ctx.fillStyle='#69b7ff';path([[.11,0,0],[-.06,.06,0],[-.06,-.06,0]],d,true);ctx.fill();dot([0,0,0],'#69b7ff',4,null,d);
ctx.fillStyle='#aec1da';ctx.fillText('Forward +x / left +y   |   '+d.settings.resolution.toFixed(2)+' m cells',18,22);ctx.fillText('FOV wedge is illustrative, not a calibrated visibility/free-space mask.',18,786);
sc.clearRect(0,0,600,240);const maxZ=Math.max(.4,...d.tracks.map(t=>t.pos[2]+t.size)),zscale=175/maxZ;
sc.strokeStyle='#52657b';sc.beginPath();sc.moveTo(30,215);sc.lineTo(580,215);sc.stroke();sc.font='12px monospace';
for(const t of d.tracks){const x=300-t.pos[1]*220/radius,z=215-t.pos[2]*zscale;sc.globalAlpha=t.current?1:.4;sc.fillStyle={loose:'#58dfb6',protected:'#ffb85c',unknown:'#c8a0ef'}[t.classification];sc.fillRect(x-8,z-t.size*zscale/2,16,Math.max(4,t.size*zscale));sc.fillText('#'+t.id+' '+t.pos[2].toFixed(2)+'m',x+12,z);sc.globalAlpha=1;}
sc.fillStyle='#a5b6cc';sc.fillText('Height +z; horizontal = base +y (left)',12,16);
}
async function tick(){try{const response=await fetch('/scan',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);data=await response.json();const d=data;draw(d);document.getElementById('status').textContent=`${d.mock?'SIMULATION':'LIVE / READ ONLY'} | pose=${d.pose.source} ${d.pose.valid?'valid':'UNAVAILABLE'} | epoch=${d.pose.epoch} | ${d.stale?'STALE':'fresh snapshot'} | anchor=${d.anchor_seen?'seen':d.build?'remembered':'unknown'}`;document.getElementById('mock').style.display=d.mock?'flex':'none';document.getElementById('warning').textContent=d.warnings.join('\n');document.getElementById('info').textContent=`Session pose: x=${d.pose.x.toFixed(2)} y=${d.pose.y.toFixed(2)} yaw=${(d.pose.yaw*180/Math.PI).toFixed(1)}°\n${d.map_semantics}\nAnchor age: ${d.build?d.build.age.toFixed(1)+'s':'unavailable'}\n${d.mock?'Buttons affect this fixture only.':'Wheel/IMU drifts; floor planarity, wheel signs and geometry require calibration.'}`;const rows=document.getElementById('rows');rows.replaceChildren();for(const t of d.tracks){const tr=document.createElement('tr');for(const v of [t.id,t.classification+(t.current?'':' (memory)'),t.age.toFixed(1)+'s',t.pos.map(n=>n.toFixed(2)).join(', ')]){const td=document.createElement('td');td.textContent=v;tr.appendChild(td);}rows.appendChild(tr);}document.getElementById('cameraLabel').textContent=d.mock?'No camera image in mock; geometric observations are simulated.':'Rectified camera image. Markers outlined, depth correspondence uses idx_2d.';const image=document.getElementById('cam');image.hidden=d.mock;if(!d.mock)image.src='/frame?t='+d.ts;}catch(e){document.getElementById('status').textContent='DISCONNECTED: '+e.message;}finally{setTimeout(tick,500);}}
document.getElementById('cam').onerror=()=>{document.getElementById('cameraLabel').textContent='Camera frame unavailable or stale.';};
for(const b of document.querySelectorAll('[data-action]'))b.onclick=async()=>{try{const r=await fetch('/mock/'+b.dataset.action,{method:'POST'});if(!r.ok)throw Error(await r.text());}catch(e){document.getElementById('warning').textContent=e.message;}};
document.getElementById('view').onchange=()=>data&&draw(data);document.getElementById('range').onchange=()=>data&&draw(data);tick();
</script></body></html>"""


def self_test():
    import unittest
    import cv2

    class PerceptionTests(unittest.TestCase):
        def setUp(self):
            self.cfg = Settings()
            self.t = time.time()
            self.pose = Pose(ts=self.t, source="test", valid=True)
            self.anchor = BuildFrame([.6, 0, .7], [1, 0, 0], [0, 1, 0], [.48, -.12, .7], self.t)

        def test_pose_round_trip(self):
            p = Pose(.3, -.7, 2.1, self.t, "test", True)
            xyz = np.array([[.3, .7, .8], [-1, .5, .2]])
            np.testing.assert_allclose(p.to_base(p.to_world(xyz)), xyz, atol=1e-12)

        def test_grid_rotates_and_translates(self):
            p = Pose(.2, -.1, math.pi/2, self.t, "test", True)
            local_anchor = self.anchor.transformed(p, inverse=True)
            cell = cell_center(1, 2, 0, self.anchor)
            np.testing.assert_allclose(cell_center(1, 2, 0, local_anchor), p.to_base(cell), atol=1e-12)
            self.assertTrue(is_in_grid(p.to_base(cell), local_anchor))
            self.assertFalse(is_in_grid(p.to_base([-.5, 0, .7]), local_anchor))

        def test_missing_orientation_rejected(self):
            with self.assertRaises(ValueError):
                resolve_anchor([.6, 0, .7])
            with self.assertRaises(ValueError):
                resolve_anchor(None)

        def test_valid_anchor_dictionary(self):
            dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
            self.assertLess(ANCHOR_ID, len(dictionary.bytesList))
            marker = cv2.aruco.generateImageMarker(dictionary, ANCHOR_ID, 140)
            image = np.full((200, 200, 3), 255, np.uint8)
            image[30:170, 30:170] = marker[:, :, None]
            self.assertEqual([mid for mid, _ in _detect_aruco(image)], [ANCHOR_ID])

        def test_sparse_indices_not_reshape(self):
            mask = np.zeros((3, 4), bool)
            mask[1, 2] = True
            points = np.array([[9, 9, 9], [1, 2, 3], [4, 5, 6], [8, 8, 8]])
            np.testing.assert_equal(_points_for_mask(mask, points, np.array([0, 6, 11, -1])), [[1, 2, 3]])
            with self.assertRaises(ValueError):
                _points_for_mask(mask, points)
            with self.assertRaises(ValueError):
                _points_for_mask(mask, np.zeros((4, 3, 3)))

        def test_marker_depth_pose(self):
            yy, xx = np.mgrid[30:171:4, 30:171:4]
            indices = (yy*200+xx).ravel()
            uv = np.column_stack(((xx.ravel()-100)/160, (100-yy.ravel())/160))
            points = np.array([.6, .25, .7])+uv[:, :1]*[0, -.1, 0]+uv[:, 1:]*[.1, 0, 0]
            corners = np.array([[20,20],[180,20],[180,180],[20,180]], np.float32)
            center, col, row = _marker_plane(corners, points, indices, (200, 200, 3))
            np.testing.assert_allclose(center, [.6, .25, .7], atol=1e-6)
            np.testing.assert_allclose(col, [0, -1, 0], atol=1e-6)
            np.testing.assert_allclose(row, [1, 0, 0], atol=1e-6)
            self.assertIsNone(_marker_plane(corners, points[:2], indices[:2], (200,200,3)))

        def test_gate_is_not_stack_confirmation(self):
            world = WorldModel()
            obs = [Detection(0, [.6, 0, .725]), Detection(1, [-.5, 0, .725])]
            s = world.update(obs, self.pose, self.t, self.anchor)
            self.assertEqual([d.id for d in s.boxes], [1])
            self.assertEqual([d.id for d in s.protected], [0])
            self.assertFalse(any(t['classification'] == 'stacked' for t in s.tracks))

        def test_no_anchor_means_unknown(self):
            s = WorldModel().update([Detection(1, [.5, 0, .7])], self.pose, self.t)
            self.assertFalse(s.boxes)
            self.assertEqual([d.id for d in s.unknown], [1])

        def test_memory_reprojects_but_is_not_pickable(self):
            world = WorldModel()
            original = [-.5, .1, .7]
            world.update([Detection(1, original)], self.pose, self.t, self.anchor)
            moved = Pose(.2, .1, math.pi, self.t+.1, "test", True)
            s = world.update([], moved, self.t+.1)
            self.assertFalse(s.boxes)
            self.assertFalse(s.tracks[0]['current'])
            np.testing.assert_allclose(s.tracks[0]['pos'], moved.to_base(original))
            self.assertFalse(scan_to_dict(s)['tracks'][0]['pick_candidate'])

        def test_reobserved_id_updates_one_track(self):
            world = WorldModel()
            world.update([Detection(1, [-.5, 0, .7])], self.pose, self.t, self.anchor)
            s = world.update([Detection(1, [.6, 0, .725])], self.pose, self.t+.1)
            self.assertEqual(len(s.tracks), 1)
            self.assertEqual([d.id for d in s.protected], [1])
            self.assertFalse(s.boxes)

        def test_duplicate_ids_not_current(self):
            s = WorldModel().update([Detection(1,[.5,0,.7]), Detection(1,[-.5,0,.7])], self.pose, self.t, self.anchor)
            self.assertFalse(s.tracks)

        def test_ttl_and_pose_epoch(self):
            world = WorldModel()
            obs = [Detection(1, [-.5, 0, .7])]
            world.update(obs, self.pose, self.t, self.anchor)
            late = world.update(obs, self.pose, self.t+20)
            self.assertIsNone(late.build)
            self.assertFalse(late.boxes)
            self.assertFalse(world.update([], self.pose, self.t+51).tracks)
            world.update(obs, self.pose, self.t+52, self.anchor)
            new_pose = Pose(ts=self.t+53, source="test", valid=True, epoch=1)
            self.assertFalse(world.update([], new_pose, self.t+53).tracks)

        def test_lost_pose_no_candidates(self):
            pose = Pose(ts=self.t, warning="lost")
            s = WorldModel().update([Detection(1, [-.5, 0, .7])], pose, self.t, self.anchor)
            self.assertFalse(s.boxes)
            self.assertIsNone(s.build)

        def test_occlusion_is_not_grasp_success(self):
            world = WorldModel()
            box = Detection(1, [-.5, 0, .7])
            before = world.update([box], self.pose, self.t, self.anchor)
            after = world.update([], self.pose, self.t+.01)
            self.assertIsNone(verify_pick(box, before=before, after=after))
            after = world.update([box], self.pose, self.t+.02)
            self.assertIs(verify_pick(box, before=before, after=after), False)
            self.assertIsNone(verify_pick(box))

        def test_height_unknown_without_samples(self):
            s = WorldModel().update([], self.pose, self.t, self.anchor)
            self.assertIsNone(verify_place((0,0,0), snapshot=s))
            s.points = np.tile([.6, 0, .75], (16,1))
            self.assertTrue(verify_place((0,0,0), snapshot=s))
            self.assertFalse(verify_place((0,2,0), snapshot=s))

        def test_wheel_odometry_straight(self):
            odom = Odometry(source="wheel")
            odom.update(self.t, [0, 0])
            turns = .05/(math.pi*odom.diameter)
            p = odom.update(self.t+.1, [turns, turns])
            self.assertTrue(p.valid)
            self.assertAlmostEqual(p.x, .05)
            self.assertAlmostEqual(p.y, 0)
            self.assertAlmostEqual(odom.at(self.t+.05).x, .025, places=5)

        def test_wheel_imu_rotation(self):
            odom = Odometry(source="wheel-imu")
            odom.update(self.t, [0,0], math.radians(179))
            angle = math.radians(2)
            turns = angle*odom.width/(2*math.pi*odom.diameter)
            p = odom.update(self.t+.1, [-turns, turns], math.radians(-179))
            self.assertTrue(p.valid)
            self.assertAlmostEqual(p.yaw, angle)
            self.assertAlmostEqual(p.x, 0)

        def test_odometry_gaps_and_bad_imu(self):
            odom = Odometry(source="wheel-imu")
            odom.update(self.t, [0,0], 0)
            self.assertFalse(odom.update(self.t+.1, [0,0], None).valid)
            odom.update(self.t+.2, [0,0], 0)
            self.assertFalse(odom.update(self.t+.3, [0,0], 2).valid)
            odom.update(self.t+.4, [0,0], 0)
            self.assertFalse(odom.update(self.t+1, [0,0], 0).valid)
            self.assertFalse(odom.history)

        def test_mock_uses_real_classifier(self):
            session = PerceptionSession(mock=True)
            first = session.poll()
            self.assertEqual({d.id for d in first.protected}, {3, 5, 7})
            self.assertEqual({d.id for d in first.boxes}, {0, 1})
            session.scene.command('visibility')
            all_seen = session.poll()
            self.assertEqual(len(all_seen.tracks), 8)
            session.scene.command('stack_next')
            third = session.poll()
            self.assertIn(2, [d.id for d in third.protected])
            self.assertNotIn(2, [d.id for d in third.boxes])
            self.assertTrue(verify_place((0,2,0), snapshot=third))

        def test_mock_turn_preserves_world(self):
            session = PerceptionSession(mock=True)
            session.scene.command('visibility')
            before = session.poll()
            original = {t['id']: t['world'] for t in before.tracks}
            session.scene.command('turn_left')
            session.scene.command('forward')
            after = session.poll()
            for t in after.tracks:
                np.testing.assert_allclose(t['world'], original[t['id']], atol=1e-12)

        def test_sweeps_never_move_hardware(self):
            class NoMotion:
                def rotate(self, angle):
                    raise AssertionError('perception attempted motion')
            with self.assertRaises(ValueError):
                scan_all(NoMotion(), sweeps=2)

        def test_stale_snapshot_has_no_candidates(self):
            s = WorldModel().update([Detection(1,[-.5,0,.7])], self.pose, self.t, self.anchor)
            s.ts -= 10
            out = scan_to_dict(s)
            self.assertTrue(out['stale'])
            self.assertFalse(out['boxes'])
            self.assertFalse(out['tracks'][0]['pick_candidate'])
            json.dumps(out, allow_nan=False)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PerceptionTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--viz", action="store_true", help="serve read-only debug UI")
    ap.add_argument("--port", type=int, default=8007)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--pose-source", choices=["wheel", "wheel-imu"], default="wheel-imu")
    ap.add_argument("--box-size", type=float, default=BOX_SIZE)
    ap.add_argument("--build-cols", type=int, default=FOOTPRINT)
    ap.add_argument("--build-rows", type=int, default=FOOTPRINT)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    cfg = Settings(box_size=a.box_size, cell=a.box_size, build_cols=a.build_cols, build_rows=a.build_rows)
    if a.self_test:
        self_test()
    elif a.viz:
        serve_viz(a.port, a.mock, cfg, a.pose_source, a.host)
    else:
        print(json.dumps(scan_to_dict(scan(a.mock, settings=cfg, pose_source=a.pose_source), cfg, a.mock), indent=2, allow_nan=False))
