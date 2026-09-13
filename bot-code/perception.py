# /// script
# dependencies = [
#   "bbos",
#   "numpy",
#   "opencv-python-headless",
#   "fastapi",
#   "uvicorn",
#   "onnxruntime==1.22.1",
#   "tokenizers==0.21.4",
#   "wsproto==1.2.0",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Read-only perception and a session-local, surrounding observation grid.

Only this module and fixtures/scan_sample.json belong to perception. No motor
Writers, HTTP motion calls, agent state, or shared-contract edits belong here.
Use PerceptionSession for persistent memory; scan() is a one-shot convenience.
Reasoning-layer integration is provider-based; this module supplies observations,
not action eligibility, possession or validated grasp poses. Scan.tracks and JSON
now include the same automatic object observations. Legacy scan_all(sweeps>1)
rejects requests instead of moving the robot or merging stale coordinates.

Frames: base is +x forward, +y left, +z up, meters. Session world starts at the
first wheel sample. Wheel distances and optional IMU yaw increments estimate
T_world_base; this is drifting planar dead reckoning, NOT SLAM or a navigation
safety map. Each live session starts with a unique JS-safe pose epoch; gaps/reset
jumps invalidate the map rather than mixing coordinate frames.
Depth points use the depth daemon's calibrated axes. Perception normalizes
heading into +x forward using the fixed forward-facing head camera's extrinsic;
--depth-yaw-deg can override this alignment. Do not apply IMU pitch twice.
The map assumes a level floor and requires a physical sign/scale check before
motion use. The module never claims navigation or successful manipulation.

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
Live GPU: --viz --detector boxes --detector-backend tensorrt --precision fp16.
Use --pose-source wheel to exclude IMU yaw. On a new robot, --prepare-detector
fetches pinned ONNX assets; --prepare-gpu-detector builds a resource-guarded engine.
The engine is keyed by model, vocabulary, input shape, precision and device/runtime.
--detector-backend cpu is an explicit reference backend, never a silent fallback.
YOLO weights (AGPL-3.0), cached CLIP embeddings and engines stay under
~/.cache/crafter-perception. No camera images are uploaded for inference.

A Reader-owning capture thread maintains only the newest immutable frame packet.
Large IPC reads are paced; the GPU worker receives RGB plus matched depth/indices,
and performs neural inference and CPU depth localization off the acquisition loop.
Surface-map rebuilding is capped at 2 Hz, but current depth is retained for queries.

WebSocket /live?view=camera|live|rect|raw|range|boxes pushes a single binary envelope:
4-byte big-endian JSON-byte-count, UTF-8 JSON metadata, then JPEG bytes. The image
metadata carries its timestamp/revision and overlays. Slow clients do not create
an unbounded frame queue. The browser decodes only one image with one latest slot.
Map updates are less frequent and are reprojected from world coordinates on display.
The live view uses detector evidence or confidence-checked visual tracking for at
most 350 ms; visual prediction NEVER updates measured observation timestamps.
The boxes view remains an explicitly delayed, timestamp-matched diagnostic snapshot.
/frame?view=live|rect|raw|range|boxes keeps HTTP access; a first unsubscribed request
may return 503 while requesting the next frame. /scan includes stage timings and
source/result freshness. --capture /tmp/new-frame.jpg --image-view range saves one
new diagnostic JPEG without overwriting. --self-test is robot-free.
GET /objects exposes session-local track IDs, original bboxes, raw detector scores,
identity status, age, and visible-surface estimates. Null position means missing
or unreliable geometry. These are NOT box-center/grasp poses; pick_candidate is
always false for automatic proposals. Use PerceptionSession(detector=True,
detector_backend="tensorrt") or /objects for automatic detections; scan_all is a
legacy marker interface. Current means recently observed, not safe to grasp.
False positives, missed boxes, merge/split events, odometry drift and depth jitter
remain possible. Camera/base calibration and build-zone setup are still required.
All sensor reads occur in one worker, not concurrent web handlers. Mock buttons
never move hardware.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace
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
    depth_yaw_deg: float | None = None

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
        if self.depth_yaw_deg is not None and not math.isfinite(self.depth_yaw_deg):
            raise ValueError("depth heading override must be finite")


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
    diagnostics: dict = field(default_factory=dict)
    objects: list = field(default_factory=list)
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
        self.surface_snapshot = []
        self.anchor = None

    def update(self, observations, pose, ts, anchor=None, points=None, warnings=(), map_update=True):
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
            self.surface_snapshot = []
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
        if len(s.points) and map_update:
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
        if map_update:
            self.surface_snapshot = []
            for key,(low,high,seen) in list(self.surfaces.items()):
                world = [(key[0]+.5)*cfg.resolution,(key[1]+.5)*cfg.resolution,high]
                if ts-seen>cfg.memory_s or math.hypot(world[0]-pose.x,world[1]-pose.y)>cfg.radius*1.5:
                    del self.surfaces[key]
                    continue
                self.surface_snapshot.append({"world":world,"z_min":low,"z_max":high,"last_seen":seen})
        if self.surface_snapshot:
            positions = pose.to_base([cell["world"] for cell in self.surface_snapshot])
            s.surface_cells = [{**cell,"pos":pos.tolist(),"age":ts-cell["last_seen"]}
                               for cell,pos in zip(self.surface_snapshot,positions)]
        return s


def depth_heading_rotation(extrinsic, yaw_deg=None):
    m = np.asarray(extrinsic, dtype=float)
    if m.shape != (3, 4) or not np.isfinite(m).all():
        raise ValueError("invalid depth extrinsic")
    if yaw_deg is None:
        forward = m[:2, 2]
        if np.linalg.norm(forward) < .1:
            raise ValueError("down-looking camera has no reliable heading; supply --depth-yaw-deg")
        yaw = -math.atan2(forward[1], forward[0])
    else:
        yaw = math.radians(yaw_deg)
    return Pose(yaw=yaw).rotation(), math.degrees(yaw)


class LiveSource:
    def __init__(self, source="wheel-imu", settings=None):
        from bbos import Reader, Config
        self.stack = contextlib.ExitStack()
        self.points = self.stack.enter_context(Reader("camera.points", keeptime=False))
        self.rect = self.stack.enter_context(Reader("camera.rect", keeptime=False, aligned_to=self.points))
        self.wheels = self.stack.enter_context(Reader("drive.state", keeptime=False))
        self.imu = self.stack.enter_context(Reader("imu.orientation", keeptime=False))
        self.head = self.stack.enter_context(Reader("camera.head.jpeg", keeptime=False))
        self.raw_jpeg, self.raw_ts = b"", 0.0
        self.sensor_ts = {"wheel": 0.0, "imu": 0.0}
        drive = Config("drive")
        self.odom = Odometry(drive.wheel_diam, drive.robot_width, source)
        import secrets
        self.odom.pose.epoch = secrets.randbits(48)
        extrinsic = np.asarray(Config("depth").camera_to_base_3x4)
        self.depth_rotation, self.depth_yaw_deg = depth_heading_rotation(
            extrinsic, (settings or Settings()).depth_yaw_deg)
        self.camera_origin = self.depth_rotation @ extrinsic[:, 3]
        self.imu_sample = None
        self.last_frame = 0.0
        self.next_head_poll,self.next_depth_poll = 0.0,0.0

    def poll(self):
        now = time.monotonic()
        head_due = now>=self.next_head_poll
        if head_due:
            self.next_head_poll = now+.1
        if head_due and self.head.ready():
            d = self.head.data
            n = int(d["jpeg_len"])
            if 0 < n <= len(d["jpeg"]):
                self.raw_jpeg, self.raw_ts = bytes(d["jpeg"][:n]), _stamp(d)
                self.sensor_ts["camera"] = self.raw_ts
        if self.imu.ready():
            d = self.imu.data
            self.sensor_ts["imu"] = _stamp(d)
            self.imu_sample = (_stamp(d), math.radians(float(d["rpy"][2])))
        if self.wheels.ready():
            d, yaw = self.wheels.data, None
            ts = _stamp(d)
            self.sensor_ts["wheel"] = ts
            if self.imu_sample and abs(ts-self.imu_sample[0]) < .15:
                yaw = self.imu_sample[1]
            self.odom.update(ts, np.array(d["pos"], dtype=float), yaw)
        if self.odom.pose.valid and time.time()-self.odom.pose.ts > .5:
            self.odom.invalidate("wheel stream stale")
        if now<self.next_depth_poll:
            return None
        self.next_depth_poll = now+.025
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
        self.sensor_ts["depth"] = ts
        n = int(d["num_points"])
        if not 0 <= n <= len(d["points"]):
            raise ValueError("invalid depth point count")
        points = np.asarray(d["points"][:n], dtype=float) @ self.depth_rotation.T
        return (np.array(rgb["left"], copy=True), points,
                np.array(d["idx_2d"][:n], copy=True), ts, self.odom.at(ts))

    def close(self):
        self.stack.close()


class CaptureWorker:
    """One Reader-owning thread; a slow consumer only replaces the latest packet."""
    def __init__(self,source="wheel-imu",settings=None,source_factory=None):
        from types import SimpleNamespace
        self.source_factory = source_factory or (lambda: LiveSource(source,settings))
        self.lock,self.stop = threading.Lock(),threading.Event()
        self.latest,self.meta,self.error = None,None,None
        self.sequence,self.consumed,self.dropped = 0,0,0
        self.raw_jpeg,self.raw_ts,self.sensor_ts = b"",0.0,{}
        self.camera_origin,self.depth_yaw_deg = np.zeros(3),0.0
        self.odom = SimpleNamespace(pose=Pose())
        self.frame_times = deque(maxlen=120)
        self.arrival_ages = deque(maxlen=120)
        self.thread = threading.Thread(target=self._run,daemon=True,name="perception-capture")
        self.thread.start()

    def _run(self):
        source = None
        try:
            source = self.source_factory()
            while not self.stop.is_set():
                packet = source.poll()
                meta = (source.raw_jpeg,source.raw_ts,dict(source.sensor_ts),
                        Pose(**asdict(source.odom.pose)),source.camera_origin,source.depth_yaw_deg)
                with self.lock:
                    self.meta = meta
                    if packet is not None:
                        for array in packet[:3]:
                            array.setflags(write=False)
                        if self.sequence>self.consumed:
                            self.dropped += 1
                        self.latest = packet
                        self.sequence += 1
                        self.frame_times.append(time.monotonic())
                        self.arrival_ages.append(max(0,time.time()-packet[3])*1000)
                self.stop.wait(.002)
        except Exception as e:
            with self.lock:
                self.error = f"{type(e).__name__}: {e}"
        finally:
            if source is not None:
                source.close()

    def poll(self):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            if self.meta is not None:
                raw,ts,sensors,pose,origin,yaw = self.meta
                self.raw_jpeg,self.raw_ts,self.sensor_ts = raw,ts,sensors
                self.odom.pose,self.camera_origin,self.depth_yaw_deg = pose,origin,yaw
            if self.sequence==self.consumed:
                return None
            self.consumed = self.sequence
            return self.latest

    def stats(self):
        with self.lock:
            times = list(self.frame_times)
            fps = (len(times)-1)/(times[-1]-times[0]) if len(times)>1 and times[-1]>times[0] else 0.0
            return {"capture_fps":fps,"capture_packets":self.sequence,"replaced_packets":self.dropped,
                    "arrival_age_p50_ms":float(np.percentile(self.arrival_ages,50)) if self.arrival_ages else None,
                    "arrival_age_p95_ms":float(np.percentile(self.arrival_ages,95)) if self.arrival_ages else None}

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise TimeoutError("read-only capture worker did not stop within its deadline")


def _stamp(data):
    return int(data["timestamp"].astype("datetime64[ns]").astype(np.int64)) / 1e9


# ---- public API -------------------------------------------------------------

MODEL_CACHE = Path.home() / ".cache" / "crafter-perception" / "yolo-world-v2"
MODEL_ASSETS = (
    ("detector.onnx", "https://huggingface.co/Instemic/yolo-world-onnx/resolve/7e1d02c9467c32b141df81890737490764776785/yolov8s-worldv2.onnx",
     51142204, "381ced485b23ed8f06de3e82bb2745e1420c181c64f0a176784c34a959d550a1"),
    ("text.onnx", "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/d15189d7028b43f1d3e65039190477f6af591c2a/onnx/text_model_quantized.onnx",
     64504507, "73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a"),
    ("tokenizer.json", "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/d15189d7028b43f1d3e65039190477f6af591c2a/tokenizer.json",
     2224119, "git:bc1f77d20440541dd073ebae6f6c401087c7d34e"),
)


def _check_asset(path, size, expected):
    import hashlib
    if path.stat().st_size != size:
        raise ValueError(f"unexpected size for {path.name}")
    git = expected.startswith("git:")
    digest = hashlib.sha1() if git else hashlib.sha256()
    if git:
        digest.update(f"blob {size}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected.removeprefix("git:"):
        raise ValueError(f"checksum mismatch for {path.name}")


def prepare_detector(cache=MODEL_CACHE):
    """Download pinned ONNX data, never remote Python code. YOLO weights: AGPL-3.0."""
    import os
    import tempfile
    import urllib.request
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    for name, url, size, checksum in MODEL_ASSETS:
        target = cache / name
        if target.exists():
            _check_asset(target, size, checksum)
            print(f"Verified cached {name}", flush=True)
            continue
        print(f"Downloading {name} ({size/1e6:.1f} MB)", flush=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as f:
                temporary = Path(f.name)
                with urllib.request.urlopen(url, timeout=60) as response:
                    total = 0
                    while chunk := response.read(1024*1024):
                        total += len(chunk)
                        if total > size:
                            raise ValueError(f"oversized download for {name}")
                        f.write(chunk)
            _check_asset(temporary, size, checksum)
            os.link(temporary, target)
            print(f"Verified {name}", flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return cache


def _ort_options():
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 3
    return options


DETECTOR_PROMPTS = ("cardboard box", "person", "chair", "backpack", "laptop", "table")


def _hash_file(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_key(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:20]


def _text_embeddings(cache):
    cache = Path(cache)
    key = _cache_key({"assets": [(a[0],a[3]) for a in MODEL_ASSETS[1:]], "prompts": DETECTOR_PROMPTS})
    path = cache/f"text-embeddings-{key}.npy"
    if path.exists():
        values = np.load(path,allow_pickle=False)
        if values.shape!=(1,len(DETECTOR_PROMPTS),512) or values.dtype!=np.float32 or not np.isfinite(values).all():
            raise ValueError("invalid cached text embeddings")
        return values
    import onnxruntime as ort
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(cache/"tokenizer.json"))
    tokenizer.enable_truncation(max_length=77)
    tokenizer.enable_padding(length=77,pad_id=49407,pad_token="<|endoftext|>")
    encoded = tokenizer.encode_batch(list(DETECTOR_PROMPTS))
    inputs = {"input_ids":np.array([e.ids for e in encoded],np.int64),
              "attention_mask":np.array([e.attention_mask for e in encoded],np.int64)}
    session = ort.InferenceSession(str(cache/"text.onnx"),_ort_options(),providers=["CPUExecutionProvider"])
    embeddings = session.run(["text_embeds"],{i.name:inputs[i.name] for i in session.get_inputs()})[0]
    values = (embeddings/np.linalg.norm(embeddings,axis=-1,keepdims=True))[None].astype(np.float32)
    with path.open("xb") as f:
        np.save(f,values,allow_pickle=False)
    return values


def _tensorrt():
    import importlib
    import sys
    try:
        return importlib.import_module("tensorrt")
    except ModuleNotFoundError:
        if sys.version_info[:2]!=(3,10):
            raise RuntimeError("JetPack TensorRT bindings require Python 3.10; CPU fallback is not automatic")
        system_path = "/usr/lib/python3.10/dist-packages"
        if not (Path(system_path)/"tensorrt"/"tensorrt.so").exists():
            raise RuntimeError("JetPack TensorRT bindings are unavailable")
        if system_path not in sys.path:
            sys.path.append(system_path)
        return importlib.import_module("tensorrt")


def _gpu_identity():
    import ctypes as c
    driver = c.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [c.c_uint]
    driver.cuDeviceGetName.argtypes = [c.c_void_p,c.c_int,c.c_int]
    driver.cuDeviceComputeCapability.argtypes = [c.POINTER(c.c_int),c.POINTER(c.c_int),c.c_int]
    driver.cuDriverGetVersion.argtypes = [c.POINTER(c.c_int)]
    name,major,minor,version = c.create_string_buffer(128),c.c_int(),c.c_int(),c.c_int()
    for code in (driver.cuInit(0),driver.cuDeviceGetName(name,128,0),
                 driver.cuDeviceComputeCapability(c.byref(major),c.byref(minor),0),
                 driver.cuDriverGetVersion(c.byref(version))):
        if code:
            raise RuntimeError(f"CUDA device query failed: {code}")
    return {"name":name.value.decode(),"sm":f"{major.value}{minor.value}","cuda_driver":version.value}


def _engine_spec(size, precision, embeddings, trt_version=None):
    import hashlib
    if size%32 or not 320<=size<=960 or precision not in ("fp16","fp32"):
        raise ValueError("invalid TensorRT build settings")
    return {"model":MODEL_ASSETS[0][3],"prompts":DETECTOR_PROMPTS,
            "embeddings":hashlib.sha256(embeddings.tobytes()).hexdigest(),
            "image_shape":[1,3,math.ceil(size*.75/32)*32,size],
            "text_shape":[1,len(DETECTOR_PROMPTS),512],"precision":precision,
            "tensorrt":trt_version or _tensorrt().__version__,"device":_gpu_identity(),
            "workspace_mib":128,"builder_level":1,"format":1}


def _available_memory_mib():
    fields = dict(line.split(":",1) for line in Path("/proc/meminfo").read_text().splitlines())
    return int(fields["MemAvailable"].split()[0])/1024


def prepare_gpu_detector(cache=MODEL_CACHE,size=512,precision="fp16"):
    """Build only this app's engine; abort our builder on excessive memory pressure."""
    import subprocess
    import uuid
    cache = Path(cache)
    for name,_,count,checksum in MODEL_ASSETS:
        _check_asset(cache/name,count,checksum)
    embeddings = _text_embeddings(cache)
    import importlib.metadata
    version = next((d.version for d in importlib.metadata.distributions(path=["/usr/lib/python3.10/dist-packages"])
                    if d.metadata.get("Name")=="tensorrt"),None)
    if version is None:
        raise RuntimeError("JetPack TensorRT metadata unavailable")
    spec = _engine_spec(size,precision,embeddings,version)
    stem = cache/f"trt-{_cache_key(spec)}"
    engine,manifest = stem.with_suffix(".engine"),stem.with_suffix(".json")
    if engine.exists() and manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["spec"]!=json.loads(json.dumps(spec)) or saved["engine_sha256"]!=_hash_file(engine):
            raise ValueError("engine cache validation failed; refusing to overwrite it")
        print(f"Verified TensorRT cache: {engine}",flush=True)
        return engine
    if engine.exists() or manifest.exists():
        raise ValueError("incomplete engine cache; choose a clean cache location")
    if _available_memory_mib()<600:
        raise RuntimeError("less than 600 MiB available; engine build deferred to protect running services")
    temporary = cache/f"build-{uuid.uuid4().hex}.engine"
    log = temporary.with_suffix(".log")
    shapes = "images:"+"x".join(map(str,spec["image_shape"]))+",txt_feats:"+"x".join(map(str,spec["text_shape"]))
    command = ["/usr/src/tensorrt/bin/trtexec",f"--onnx={cache/'detector.onnx'}",f"--saveEngine={temporary}",
               f"--optShapes={shapes}","--memPoolSize=workspace:128","--builderOptimizationLevel=1",
               "--maxAuxStreams=0","--skipInference",f"--tempdir={cache}"]
    if precision=="fp16":
        command.append("--fp16")
    print(f"Building {precision} on {spec['device']['name']}; build log: {log}",flush=True)
    process = None
    try:
        with log.open("xb") as output:
            process = subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT)
            started = time.monotonic()
            while process.poll() is None:
                if _available_memory_mib()<300:
                    raise RuntimeError(f"builder stopped for low available memory; see {log}")
                if time.monotonic()-started>900:
                    raise TimeoutError(f"bounded engine build timed out; see {log}")
                time.sleep(.1)
        if process.returncode or not temporary.exists():
            raise RuntimeError(f"TensorRT build failed (exit {process.returncode}); see {log}")
        import os
        os.link(temporary,engine)
        with manifest.open("x") as f:
            json.dump({"spec":spec,"engine_sha256":_hash_file(engine)},f,indent=2)
        print(f"Built and checksummed: {engine}",flush=True)
        return engine
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        temporary.unlink(missing_ok=True)


class CudaRuntime:
    def __init__(self):
        import ctypes as c
        self.c = c
        self.lib = c.CDLL("/usr/local/cuda/lib64/libcudart.so")
        pointer,size = c.c_void_p,c.c_size_t
        signatures = {"cudaMalloc":[c.POINTER(pointer),size],"cudaFree":[pointer],
                      "cudaHostAlloc":[c.POINTER(pointer),size,c.c_uint],"cudaFreeHost":[pointer],
                      "cudaStreamCreateWithFlags":[c.POINTER(pointer),c.c_uint],
                      "cudaStreamSynchronize":[pointer],"cudaStreamDestroy":[pointer],
                      "cudaMemcpyAsync":[pointer,pointer,size,c.c_int,pointer]}
        for name,args in signatures.items():
            fn = getattr(self.lib,name)
            fn.argtypes,fn.restype = args,c.c_int
        self.lib.cudaGetErrorString.argtypes = [c.c_int]
        self.lib.cudaGetErrorString.restype = c.c_char_p

    def call(self,name,*args):
        code = getattr(self.lib,name)(*args)
        if code:
            raise RuntimeError(f"{name}: {self.lib.cudaGetErrorString(code).decode()}")


class TensorRTSession:
    def __init__(self,cache,size,precision,embeddings):
        trt = _tensorrt()
        cache = Path(cache)
        spec = _engine_spec(size,precision,embeddings)
        stem = cache/f"trt-{_cache_key(spec)}"
        engine_path,manifest = stem.with_suffix(".engine"),stem.with_suffix(".json")
        if not engine_path.exists() or not manifest.exists():
            raise FileNotFoundError("TensorRT engine missing; run --prepare-gpu-detector first")
        saved = json.loads(manifest.read_text())
        if saved["spec"]!=json.loads(json.dumps(spec)) or saved["engine_sha256"]!=_hash_file(engine_path):
            raise ValueError("TensorRT engine/model/vocabulary cache mismatch")
        self.cuda = CudaRuntime()
        c = self.cuda.c
        self.stream,self.buffers,self.input_names,self.output_names = c.c_void_p(),{},[],[]
        self.info = {"backend":"tensorrt","precision":precision,"device":spec["device"],"engine":engine_path.name}
        self.closed,self.last_ms = False,0.0
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT engine deserialization failed")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT context creation failed")
        dtypes = {trt.DataType.FLOAT:np.float32,trt.DataType.HALF:np.float16,
                  trt.DataType.INT32:np.int32,trt.DataType.INT64:np.int64,trt.DataType.BOOL:np.bool_}
        try:
            self.cuda.call("cudaStreamCreateWithFlags",c.byref(self.stream),1)
            for name,shape in (("images",spec["image_shape"]),("txt_feats",spec["text_shape"])):
                if not self.context.set_input_shape(name,shape):
                    raise ValueError(f"TensorRT rejected shape for {name}")
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = tuple(self.context.get_tensor_shape(name))
                if any(d<=0 for d in shape):
                    raise ValueError(f"unresolved TensorRT shape: {name} {shape}")
                dtype = np.dtype(dtypes[self.engine.get_tensor_dtype(name)])
                count = int(np.prod(shape)); nbytes=count*dtype.itemsize
                device,host = c.c_void_p(),c.c_void_p()
                self.buffers[name] = {"device":device,"host":host,"nbytes":nbytes}
                self.cuda.call("cudaMalloc",c.byref(device),nbytes)
                self.cuda.call("cudaHostAlloc",c.byref(host),nbytes,0)
                raw = (c.c_ubyte*nbytes).from_address(host.value)
                self.buffers[name]["array"] = np.frombuffer(raw,dtype=dtype).reshape(shape)
                if not self.context.set_tensor_address(name,device.value):
                    raise RuntimeError(f"cannot bind TensorRT tensor {name}")
                (self.input_names if self.engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT else self.output_names).append(name)
        except Exception:
            self.close()
            raise

    def run(self,outputs,inputs):
        c = self.cuda.c
        started = time.monotonic()
        for name in self.input_names:
            b = self.buffers[name]
            if inputs[name].shape!=b["array"].shape:
                raise ValueError(f"TensorRT shape mismatch for {name}: {inputs[name].shape}")
            np.copyto(b["array"],inputs[name],casting="same_kind")
            self.cuda.call("cudaMemcpyAsync",b["device"],b["host"],b["nbytes"],1,self.stream)
        if not self.context.execute_async_v3(self.stream.value):
            raise RuntimeError("TensorRT GPU execution failed; no CPU fallback")
        for name in self.output_names:
            b = self.buffers[name]
            self.cuda.call("cudaMemcpyAsync",b["host"],b["device"],b["nbytes"],2,self.stream)
        self.cuda.call("cudaStreamSynchronize",self.stream)
        self.last_ms = (time.monotonic()-started)*1000
        return [self.buffers[name]["array"].copy() for name in (outputs or self.output_names)]

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.stream.value:
            self.cuda.call("cudaStreamSynchronize",self.stream)
        self.context = None
        for b in self.buffers.values():
            b.pop("array",None)
            if b["device"].value:
                self.cuda.call("cudaFree",b["device"])
            if b["host"].value:
                self.cuda.call("cudaFreeHost",b["host"])
        self.buffers.clear()
        if self.stream.value:
            self.cuda.call("cudaStreamDestroy",self.stream)
        self.engine,self.runtime = None,None


class BoxDetector:
    """YOLO-World with explicit CPU or native TensorRT GPU execution."""
    def __init__(self, cache=MODEL_CACHE, size=640, threshold=.2, backend="cpu", precision="fp16"):
        if backend not in ("cpu","tensorrt"):
            raise ValueError("unknown detector backend")
        if size % 32 or not 320 <= size <= 960 or not 0 < threshold < 1:
            raise ValueError("invalid detector size/threshold")
        cache = Path(cache)
        for name, _, count, checksum in MODEL_ASSETS:
            if not (cache/name).exists():
                raise FileNotFoundError("model assets missing; run --prepare-detector first")
            _check_asset(cache/name, count, checksum)
        self.prompts = list(DETECTOR_PROMPTS)
        self.embeddings = _text_embeddings(cache)
        self.backend = backend
        if backend=="tensorrt":
            self.session = TensorRTSession(cache,size,precision,self.embeddings)
            self.runtime_info = self.session.info
        else:
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(cache/"detector.onnx"),_ort_options(),providers=["CPUExecutionProvider"])
            self.runtime_info = {"backend":"cpu","providers":self.session.get_providers(),"precision":"fp32"}
        self.size, self.threshold = size, threshold
        self.score_range = None
        self.timings = []

    def close(self):
        if self.backend=="tensorrt":
            self.session.close()

    def detect(self, rgb):
        import cv2
        h,w = rgb.shape[:2]
        found = self._detect_single(rgb)
        if w>=400 and h>=240:
            x,y = w//4,h//2
            for item in self._detect_single(rgb[y:,x:w-x]):
                item["bbox"] = (np.asarray(item["bbox"])+[x,y,x,y]).tolist()
                found.append(item)
        if not found:
            return []
        boxes = [d["bbox"][:2]+[d["bbox"][2]-d["bbox"][0],d["bbox"][3]-d["bbox"][1]] for d in found]
        keep = cv2.dnn.NMSBoxes(boxes,[d["score"] for d in found],self.threshold,.45)
        return [found[int(i)] for i in np.asarray(keep).reshape(-1)][:12]

    def _detect_single(self, rgb):
        import cv2
        started = time.monotonic()
        h, w = rgb.shape[:2]
        scale = self.size / max(h, w)
        nw, nh = round(w*scale), round(h*scale)
        padded_w, padded_h = math.ceil(nw/32)*32, math.ceil(nh/32)*32
        left, top = (padded_w-nw)//2, (padded_h-nh)//2
        image = np.full((padded_h,padded_w,3), 114, np.uint8)
        image[top:top+nh,left:left+nw] = cv2.resize(rgb, (nw,nh), interpolation=cv2.INTER_LINEAR)
        tensor = image.transpose(2,0,1)[None].astype(np.float32)/255
        prepared = time.monotonic()
        out = self.session.run(None, {"images": tensor, "txt_feats": self.embeddings})[0]
        inferred = time.monotonic()
        if hasattr(self,"timings"):
            self.timings.append({"preprocess_ms":(prepared-started)*1000,"inference_ms":(inferred-prepared)*1000})
            self.timings = self.timings[-2:]
        if out.shape[1] != 4+len(self.prompts):
            raise ValueError(f"unexpected detector output {out.shape}")
        values = out[0].T
        scores = values[:,4:]
        self.score_range = [float(scores.min()), float(scores.max())]
        if scores.min() < -.001 or scores.max() > 1.001:
            raise ValueError("pinned detector should output sigmoid scores, not logits")
        scores = np.clip(scores, 0, 1)
        target = (scores.argmax(axis=1)==0) & (scores[:,0]>=self.threshold)
        boxes, confidence = values[target,:4], scores[target,0]
        if not len(boxes):
            return []
        xy = (boxes[:,:2]-boxes[:,2:]/2-[left,top])/scale
        wh = boxes[:,2:]/scale
        nms = cv2.dnn.NMSBoxes(np.column_stack((xy,wh)).tolist(), confidence.tolist(), self.threshold, .45)
        result = []
        for index in np.asarray(nms).reshape(-1):
            x1,y1 = np.maximum(xy[index], [0,0])
            x2,y2 = np.minimum(xy[index]+wh[index], [w,h])
            if x2-x1 >= 3 and y2-y1 >= 3:
                result.append({"bbox": [float(x1),float(y1),float(x2),float(y2)],
                               "score": float(confidence[index]), "label": "cardboard_box"})
        return sorted(result, key=lambda d: -d["score"])[:12]


def benchmark_detector(image_path, cache=MODEL_CACHE, size=640, threshold=.2, output=None, backend="cpu", precision="fp16", iterations=10):
    import cv2
    import resource
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("benchmark image cannot be decoded")
    start = time.monotonic()
    detector = BoxDetector(cache,size,threshold,backend,precision)
    load_s = time.monotonic()-start
    durations = []
    for _ in range(iterations):
        start = time.monotonic()
        detections = detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        durations.append(time.monotonic()-start)
    print(json.dumps({"load_s": load_s, "inference_s": durations,
                      "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      "runtime":detector.runtime_info,"p50_ms":float(np.percentile(durations,50)*1000),
                      "p95_ms":float(np.percentile(durations,95)*1000),"passes":detector.timings,
                      "score_range": detector.score_range, "detections": detections}, indent=2))
    detector.close()
    if output:
        for d in detections:
            x1,y1,x2,y2 = [round(v) for v in d["bbox"]]
            cv2.rectangle(image, (x1,y1), (x2,y2), (80,240,80), 2)
            cv2.putText(image, f'cardboard box {d["score"]:.2f}', (x1,max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX,.45,(80,240,80),1)
        with Path(output).open("xb") as f:
            f.write(_encode_jpeg(image))


def _bbox_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    wh = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(wh))
    union = float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2]))-intersection
    return intersection/max(union, 1e-9)


def _support_plane(points):
    if len(points) < 30:
        return None
    points = points[::max(1,len(points)//600)]
    rng, best = np.random.default_rng(7), None
    for _ in range(40):
        a,b,c = points[rng.choice(len(points),3,replace=False)]
        normal = np.cross(b-a,c-a)
        length = np.linalg.norm(normal)
        if length < 1e-7:
            continue
        normal /= length
        if abs(normal[2]) < .8:
            continue
        mask = np.abs((points-a) @ normal) < .018
        if best is None or mask.sum() > best.sum():
            best = mask
    if best is None or best.sum() < max(25,.55*len(points)):
        return None
    center = np.median(points[best],axis=0)
    _,_,v = np.linalg.svd(points[best]-center, full_matrices=False)
    normal = v[-1]
    if normal[2] < 0:
        normal = -normal
    return (normal, -float(center @ normal)) if normal[2] > .8 else None


def localize_box(detection, points, indices, shape, camera_origin):
    """Return a visible-surface estimate, never a grasp pose or invented center."""
    h,w = shape[:2]
    x1,y1,x2,y2 = detection["bbox"]
    bw,bh = x2-x1,y2-y1
    pixels = np.column_stack((indices%w, indices//w))
    valid = ((indices>=0)&(indices<h*w)&np.isfinite(points).all(axis=1)
             &(np.linalg.norm(points,axis=1)>1e-3))
    inside = (valid & (pixels[:,0]>x1+.15*bw)&(pixels[:,0]<x2-.15*bw)
              &(pixels[:,1]>y1+.15*bh)&(pixels[:,1]<y2-.15*bh))
    outer = (valid & (pixels[:,0]>x1-.5*bw)&(pixels[:,0]<x2+.5*bw)
             &(pixels[:,1]>y1-.5*bh)&(pixels[:,1]<y2+.5*bh))
    whole = ((pixels[:,0]>=x1)&(pixels[:,0]<=x2)&(pixels[:,1]>=y1)&(pixels[:,1]<=y2))
    pts = points[inside]
    result = dict(detection, position_base_m=None, position_kind="visible_surface_centroid",
                  depth_status="missing", depth_points=len(pts), support_clearance_m=None,
                  partial_view=bool(x1<2 or y1<2 or x2>w-2 or y2>h-2), grasp_pose=None)
    if len(pts)<MIN_MASK_PTS:
        return result
    plane = _support_plane(points[outer & ~whole])
    if plane is not None:
        normal, offset = plane
        heights = pts @ normal+offset
        pts = pts[(heights>.022)&(heights<.7)]
        if len(pts)<MIN_MASK_PTS:
            result["depth_status"] = "background_or_flat_surface"
            return result
        result["support_clearance_m"] = float(np.median(pts @ normal+offset))
        result["depth_status"] = "surface_supported"
    else:
        result["depth_status"] = "weak_no_support_plane"
    distances = np.linalg.norm(pts-camera_origin,axis=1)
    median = float(np.median(distances))
    pts = pts[np.abs(distances-median)<.10]
    if len(pts)<MIN_MASK_PTS:
        result["depth_status"] = "inconsistent_depth"
        return result
    result["position_base_m"] = np.median(pts,axis=0).tolist()
    result["depth_points"] = len(pts)
    return result


class ObjectTracker:
    def __init__(self):
        import uuid
        self.session_id = uuid.uuid4().hex[:12]
        self.tracks, self.next_id, self.epoch = {}, 1000, None

    def update(self, detections, pose, ts):
        if self.epoch != pose.epoch:
            self.tracks.clear()
            self.epoch = pose.epoch
        self.tracks = {i:t for i,t in self.tracks.items() if 0<=ts-t["last_seen"]<30}
        used, current = set(), []
        for d in detections:
            world = pose.to_world(d["position_base_m"]).tolist() if pose.valid and d["position_base_m"] is not None else None
            matches = []
            for mid, old in self.tracks.items():
                if mid in used or ts-old["last_seen"]>10:
                    continue
                previous = old["_pose"]
                still = (pose.valid and previous.valid and math.hypot(pose.x-previous.x,pose.y-previous.y)<.12
                         and abs(_wrap(pose.yaw-previous.yaw))<.15)
                iou = _bbox_iou(d["bbox"],old["bbox"]) if still else 0.0
                age_cost = .15 * (ts-old["last_seen"])
                if world is not None and old["world_position_m"] is not None:
                    distance = float(np.linalg.norm(np.asarray(world)-old["world_position_m"]))
                    if iou>.3 and distance<.35:
                        matches.append((.8*(1-iou)+.2*distance/.35+age_cost,mid))
                    elif distance < .18:
                        matches.append((.7*distance/.18+.3+age_cost,mid))
                elif iou>.5 and ts-old["last_seen"]<3:
                    matches.append((1-iou+age_cost,mid))
            matches.sort()
            ambiguous = len(matches)>1 and matches[1][0]-matches[0][0]<.12
            mid = matches[0][1] if matches and not ambiguous else self.next_id
            if mid == self.next_id:
                self.next_id += 1
            confirmations = self.tracks.get(mid,{}).get("confirmations",0)+1
            item = dict(d, id=mid, track_id=f"box-{mid-999:03d}", tracker_session=self.session_id,
                        world_position_m=world, observed_position_base_m=d["position_base_m"],
                        last_seen=ts, pose_epoch=pose.epoch, confirmations=confirmations,
                        identity_status="ambiguous" if ambiguous else "tracked" if confirmations>1 else "new",
                        _pose=Pose(**asdict(pose)))
            self.tracks[mid] = item
            used.add(mid)
            current.append(item)
        return current

    def snapshot(self, pose, build, now, cfg):
        output = []
        for item in self.tracks.values():
            if not 0 <= now-item["last_seen"] <= cfg.memory_s:
                continue
            d = {k:v for k,v in item.items() if not k.startswith("_")}
            same_epoch = pose.epoch==item["pose_epoch"]
            world = item["world_position_m"] if same_epoch else None
            d["world_position_m"] = world
            if not same_epoch:
                d["identity_status"] = "pose_epoch_changed"
            position = pose.to_base(world).tolist() if pose.valid and world is not None else None
            d.update(position_base_m=position, age_s=max(0,now-item["last_seen"]),
                     current=same_epoch and now-item["last_seen"]<=cfg.fresh_s, pick_candidate=False,
                     position_frame="base_at_pose_timestamp", pose_timestamp=pose.ts, snapshot_ts=now,
                     zone="unassigned" if build is None or position is None else
                     "protected" if is_in_grid(position,build,settings=cfg) else "outside_build")
            output.append(d)
        return output


class LiveOverlay:
    """Display-only short-horizon flow; never changes observation/geometry timestamps."""
    def __init__(self):
        self.gray,self.objects,self.ts,self.epoch = None,[],0.0,None

    def update(self,rgb,objects,ts,epoch):
        import cv2
        self.gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        self.objects = [{k:o[k] for k in ("id","track_id","bbox","score","identity_status")} for o in objects]
        self.ts,self.epoch = ts,epoch

    def draw(self,rgb,ts,epoch):
        import cv2
        image = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        stats = {"source":"none","detected_at":self.ts,"display_frame_at":ts,"count":0}
        if self.gray is None or epoch!=self.epoch or not 0<=ts-self.ts<=.35:
            return image,stats
        same = abs(ts-self.ts)<1e-6
        current = None if same else cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        if not same and current.shape!=self.gray.shape:
            return image,stats
        stats["source"] = "detected" if same else "visual_tracking"
        for obj in self.objects:
            box = np.asarray(obj["bbox"],float).copy()
            if not same:
                if obj["identity_status"]=="ambiguous":
                    continue
                x1,y1,x2,y2 = np.rint(box).astype(int)
                mask = np.zeros(self.gray.shape,np.uint8)
                mask[max(0,y1):min(mask.shape[0],y2),max(0,x1):min(mask.shape[1],x2)] = 255
                features = cv2.goodFeaturesToTrack(self.gray,24,.03,3,mask=mask)
                if features is None or len(features)<4:
                    continue
                moved,status,_ = cv2.calcOpticalFlowPyrLK(self.gray,current,features,None,winSize=(15,15),maxLevel=2)
                if moved is None:
                    continue
                back,back_status,_ = cv2.calcOpticalFlowPyrLK(current,self.gray,moved,None,winSize=(15,15),maxLevel=2)
                if back is None:
                    continue
                keep = ((status.ravel()==1)&(back_status.ravel()==1)
                        &(np.linalg.norm((back-features).reshape(-1,2),axis=1)<1.5))
                if keep.sum()<4:
                    continue
                shifts = (moved-features).reshape(-1,2)[keep]
                delta = np.median(shifts,axis=0)
                if not np.isfinite(delta).all() or np.linalg.norm(delta)>max(15,.5*(box[2]-box[0])):
                    continue
                if np.median(np.linalg.norm(shifts-delta,axis=1))>2:
                    continue
                box += np.tile(delta,2)
            x1,y1,x2,y2 = np.rint(box).astype(int)
            color = (80,230,100) if same else (40,195,255)
            cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
            suffix = f' {obj["score"]:.2f}' if same else ' tracked'
            cv2.putText(image,obj["track_id"]+suffix,(max(0,x1),max(14,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1)
            stats["count"] += 1
        return image,stats


def _detector_process(jobs, results, cache, size, threshold, backend="cpu", precision="fp16"):
    import os
    os.nice(5)
    detector = None
    try:
        detector = BoxDetector(cache,size,threshold,backend,precision)
        results.put({"ready": True,"runtime":detector.runtime_info})
        while True:
            job = jobs.get()
            if job is None:
                return
            ts,rgb,points,indices,camera_origin = job
            started = time.monotonic()
            detections = detector.detect(rgb)
            inferred = time.monotonic()
            localized = [localize_box(d,points,indices,rgb.shape,camera_origin) for d in detections]
            results.put({"ts":ts,"detections":detections,"localized":localized,
                         "inference_s":inferred-started,"localize_s":time.monotonic()-inferred,"passes":detector.timings})
    except Exception as e:
        results.put({"error": f"{type(e).__name__}: {e}"})
    finally:
        if detector is not None:
            detector.close()


class DetectorWorker:
    def __init__(self, cache=MODEL_CACHE, size=512, threshold=.10, backend="cpu", precision="fp16"):
        import multiprocessing
        context = multiprocessing.get_context("spawn")
        self.jobs, self.results = context.Queue(1), context.Queue(2)
        self.process = context.Process(target=_detector_process, args=(self.jobs,self.results,cache,size,threshold,backend,precision), daemon=True)
        self.process.start()
        self.pending = None
        self.max_input_age_s = .8
        self.tracker = ObjectTracker()
        self.overlay = LiveOverlay()
        self.debug_image_requested = True
        self.completions,self.result_ages = deque(maxlen=120),deque(maxlen=120)
        self.image = (b"",0.0)
        self.status = {"enabled": True,"state":"loading","error":"","inference_s":None,"requested_backend":backend}

    def poll(self, data, camera_origin):
        import queue
        import cv2
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            result = None
        if result is not None:
            if "error" in result:
                self.status.update(state="error", error=result["error"])
                self.pending = None
            elif result.get("ready"):
                self.status.update(state="ready",runtime=result.get("runtime",{}))
            elif self.pending is not None:
                rgb,points,indices,ts,pose = self.pending
                if result["ts"] != ts:
                    raise ValueError("detector returned a mismatched frame timestamp")
                localized = result.get("localized")
                if localized is None:
                    localized = [localize_box(d,points,indices,rgb.shape,camera_origin) for d in result["detections"]]
                objects = self.tracker.update(localized,pose,ts)
                overlay = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                for d in objects:
                    x1,y1,x2,y2 = [round(v) for v in d["bbox"]]
                    color = (80,230,100) if d["position_base_m"] is not None else (40,180,255)
                    cv2.rectangle(overlay,(x1,y1),(x2,y2),color,2)
                    cv2.putText(overlay,f'{d["track_id"]} score {d["score"]:.2f}',(max(0,x1),max(13,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1)
                if self.debug_image_requested:
                    self.image = (_encode_jpeg(overlay),ts)
                self.overlay.update(rgb,objects,ts,pose.epoch)
                self.completions.append(time.monotonic())
                self.result_ages.append(max(0,time.time()-ts)*1000)
                duration = self.completions[-1]-self.completions[0]
                fps = (len(self.completions)-1)/duration if duration>0 else 0.0
                self.pending = None
                self.status.update(state="ready",inference_s=result["inference_s"],passes=result.get("passes",[]),frame_ts=ts,
                                   detector_fps=fps,localize_ms=result.get("localize_s",0)*1000,
                                   result_age_p50_ms=float(np.percentile(self.result_ages,50)),
                                   result_age_p95_ms=float(np.percentile(self.result_ages,95)),
                                   detections=len(objects),localized=sum(d["position_base_m"] is not None for d in objects))
        if not self.process.is_alive() and self.status["state"]!="error":
            self.status.update(state="error",error="detector process stopped")
            self.pending = None
        if data is not None and not -.05<=time.time()-data[3]<=getattr(self,"max_input_age_s",.8):
            self.status["frames_skipped_stale"] = self.status.get("frames_skipped_stale",0)+1
            data = None
        if data is not None and self.pending is not None:
            self.status["frames_skipped_busy"] = self.status.get("frames_skipped_busy",0)+1
        if data is not None and self.status["state"]=="ready" and self.pending is None:
            rgb,points,indices,ts,pose = data
            self.pending = (rgb,points,indices,ts,Pose(**asdict(pose)))
            self.jobs.put_nowait((ts,rgb,points.astype(np.float32,copy=False),indices,camera_origin))
        return result is not None

    def close(self):
        try:
            self.jobs.put_nowait(None)
        except Exception:
            pass
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
        for q in (self.jobs,self.results):
            q.cancel_join_thread()
            q.close()


def range_image(points, indices, shape, camera_origin, maximum=3.0):
    """Camera-range diagnostic in meters; pixels with no valid depth stay black."""
    import cv2
    h, w = shape[:2]
    points, indices = np.asarray(points), np.asarray(indices)
    if maximum <= 0 or indices.shape != (len(points),):
        raise ValueError("invalid range image inputs")
    distances = np.linalg.norm(points - np.asarray(camera_origin), axis=1)
    valid = (np.isfinite(points).all(axis=1) & np.isfinite(distances)
             & (distances > .02) & (indices >= 0) & (indices < h*w))
    depth = np.full(h*w, np.inf, dtype=np.float32)
    np.minimum.at(depth, indices[valid], distances[valid])
    known = np.isfinite(depth)
    scaled = np.zeros(h*w, np.uint8)
    scaled[known] = np.clip(depth[known] / maximum * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled.reshape(h, w), cv2.COLORMAP_TURBO)
    colored[~known.reshape(h, w)] = 0
    return colored


def _encode_jpeg(image):
    import cv2
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return encoded.tobytes() if ok else b""


class PerceptionSession:
    def __init__(self, mock=False, settings=None, pose_source="wheel-imu", detector=False,
                 model_cache=MODEL_CACHE, detector_size=512, detector_threshold=.10,
                 detector_backend="cpu", precision="fp16"):
        self.mock, self.settings = mock, settings or Settings()
        self.world = WorldModel(self.settings)
        if not mock:
            import cv2
            cv2.setNumThreads(1)
        self.source = None if mock else CaptureWorker(pose_source,self.settings)
        self.scene = MockSource(self.settings) if mock else None
        self.latest = Scan()
        self.jpeg = b""
        self.streams = {name: (b"", 0.0) for name in ("rect", "range", "raw", "boxes", "live")}
        self.requested_views = set(self.streams)
        self.last_packet,self.overlay_info = None,{}
        self.last_map_ts = 0.0
        self.processing_ms = deque(maxlen=120)
        self.detector = None
        try:
            if detector and not mock:
                self.detector = DetectorWorker(model_cache,detector_size,detector_threshold,detector_backend,precision)
                self.detector.max_input_age_s = self.settings.fresh_s
        except Exception:
            if self.source:
                self.source.close()
            raise

    def _live_image(self,packet):
        if packet is None or "live" not in self.requested_views:
            return
        rgb,_,_,ts,pose = packet
        if self.detector:
            image,self.overlay_info = self.detector.overlay.draw(rgb,ts,pose.epoch)
        else:
            import cv2
            image = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            self.overlay_info = {"source":"none","display_frame_at":ts,"count":0}
        self.streams["live"] = (_encode_jpeg(image),ts)

    def poll(self):
        started = time.monotonic()
        stages = {}
        if self.mock:
            observations, anchor, points, ts, pose = self.scene.next()
            warnings = ["SIMULATED observations; not camera detection. Same world classifier as live."]
        else:
            data = self.source.poll()
            detector_changed = False
            if self.detector:
                self.detector.debug_image_requested = "boxes" in self.requested_views
                detector_changed = self.detector.poll(data, self.source.camera_origin)
                self.streams["boxes"] = self.detector.image
            stages['result_handling_ms'] = (time.monotonic()-started)*1000
            import cv2
            if "raw" in self.requested_views and self.source.raw_ts-self.streams["raw"][1] >= .15 and self.source.raw_jpeg:
                raw = cv2.imdecode(np.frombuffer(self.source.raw_jpeg, np.uint8), cv2.IMREAD_COLOR)
                if raw is not None:
                    self.streams["raw"] = (_encode_jpeg(raw[:, :raw.shape[1]//2]), self.source.raw_ts)
            if data is None:
                if detector_changed and self.latest.ts:
                    self._live_image(self.last_packet)
                    self.latest = _attach_object_tracks(self.latest,self.detector.tracker.snapshot(
                        self.latest.pose,self.latest.build,time.time(),self.settings))
                    return self.latest
                return None
            self.last_packet = data
            rgb, points, indices, ts, pose = data
            marker_started = time.monotonic()
            observations, anchor, warnings, markers = observe(
                rgb, points, indices, ts, self.settings, self.source.camera_origin)
            stages['marker_ms'] = (time.monotonic()-marker_started)*1000
            encoding_started = time.monotonic()
            import cv2
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for mid, corners in markers:
                cv2.polylines(image, [corners.astype(np.int32)], True, (0, 210, 255), 1)
                cv2.putText(image, str(mid), tuple(corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,210,255), 1)
            if "rect" in self.requested_views:
                self.jpeg = _encode_jpeg(image)
                self.streams["rect"] = (self.jpeg,ts)
            if "range" in self.requested_views:
                self.streams["range"] = (_encode_jpeg(range_image(points,indices,rgb.shape,self.source.camera_origin)),ts)
            self._live_image(data)
            stages['encoding_tracking_ms'] = (time.monotonic()-encoding_started)*1000
        map_started = time.monotonic()
        update_map = self.mock or ts-self.last_map_ts>=.5 or self.world.epoch!=pose.epoch
        self.latest = self.world.update(observations,pose,ts,anchor,points,warnings,map_update=update_map)
        if update_map:
            self.last_map_ts = ts
        stages['map_ms'] = (time.monotonic()-map_started)*1000
        if self.detector:
            self.latest = _attach_object_tracks(self.latest,self.detector.tracker.snapshot(pose,self.latest.build,time.time(),self.settings))
        if not self.mock:
            self.processing_ms.append((time.monotonic()-started)*1000)
            self.latest.diagnostics = {"depth_yaw_deg": self.source.depth_yaw_deg,
                                       "point_count": len(points), "rect_shape": list(rgb.shape),
                                       "marker_ids": [mid for mid, _ in markers],
                                       "pose_camera_skew_s": abs(pose.ts-ts),
                                       "imu_units": "publisher degrees -> internal radians",
                                       "capture": self.source.stats(),"overlay":dict(self.overlay_info),"stages":stages,
                                       "processing_p95_ms":float(np.percentile(self.processing_ms,95))}
        return self.latest

    def close(self):
        try:
            if self.source:
                self.source.close()
        finally:
            if self.detector:
                self.detector.close()


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


def _attach_object_tracks(snapshot,objects):
    tracks = [dict(t) for t in snapshot.tracks if t.get("source")!="box_detector"]
    for obj in objects:
        if obj["position_base_m"] is None or obj["world_position_m"] is None:
            continue
        tracks.append({"id":obj["id"],"name":obj["track_id"],"source":"box_detector",
                       "pos":list(obj["position_base_m"]),"world":list(obj["world_position_m"]),
                       "age":obj["age_s"],"last_seen":obj["last_seen"],"current":obj["current"],
                       "classification":"protected" if obj["zone"]=="protected" else
                       "loose" if obj["zone"]=="outside_build" else "unknown",
                       "size":0.0,"position_kind":obj["position_kind"],"depth_status":obj["depth_status"],
                       "identity_status":obj["identity_status"],"score":obj["score"],"pick_candidate":False})
    return replace(snapshot,objects=objects,tracks=tracks)


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
        t["pick_candidate"] = bool(t.get("source")!="box_detector" and t["current"] and t["classification"]=="loose" and build and build["valid"])
        tracks.append(t)
    objects = []
    for item in s.objects:
        o = dict(item)
        o["age_s"] = max(0, now-o["last_seen"])
        o["current"] = o["current"] and not stale and o["age_s"] <= cfg.fresh_s
        o["visibility"] = "recent_detection" if o["age_s"] < 3 else "remembered"
        if stale:
            o["position_base_m"] = None
        objects.append(o)
    return {"schema_version": 2, "mock": mock, "ts": s.ts, "stale": stale,
            "frame": "base_at_capture", "pose": asdict(s.pose), "build": build,
            "anchor_seen": s.anchor_seen, "tracks": tracks, "objects": objects,
            "boxes": [asdict(d) for d in s.boxes] if not stale else [],
            "protected": [asdict(d) for d in s.protected], "unknown": [asdict(d) for d in s.unknown],
            "surface_cells": s.surface_cells, "settings": asdict(cfg), "diagnostics": s.diagnostics,
            "warnings": s.warnings + (["snapshot stale; no pick candidates"] if stale else []),
            "map_semantics": "observed surfaces only; blank cells UNKNOWN, not free; not a navigation map"}


# ---- debug visualizer ---------------------------------------------------------

def serve_viz(port=8007,mock=False,settings=None,pose_source="wheel-imu",host="127.0.0.1",
              detector=False,model_cache=MODEL_CACHE,detector_size=512,detector_threshold=.10,
              detector_backend="cpu",precision="fp16"):
    import asyncio
    import struct
    from fastapi import FastAPI,HTTPException,Response,WebSocket,WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    import uvicorn
    globals()["WebSocket"] = WebSocket
    cfg = settings or Settings()
    lock,stop = threading.RLock(),threading.Event()
    views = {"camera","live","rect","range","raw","boxes"}
    clients,leases = {},{}
    shared = {"scan":Scan(),"streams":{},"revisions":{},"sensor_ts":{},"sensor_pose":Pose(),
              "error":"waiting for sensors","actions":deque(),"loop":None,"overlay":{},
              "detector_status":{"enabled":detector,"state":"loading" if detector else "disabled"}}

    def notify():
        loop = shared["loop"]
        if loop is not None and not loop.is_closed():
            for client in list(clients.values()):
                loop.call_soon_threadsafe(client["event"].set)

    def worker():
        session = None
        try:
            session = PerceptionSession(mock,cfg,pose_source,detector,model_cache,detector_size,detector_threshold,
                                        detector_backend,precision)
            while not stop.is_set():
                with lock:
                    actions = list(shared["actions"])
                    shared["actions"].clear()
                    active = {c["view"] for c in clients.values()} | {v for v,end in leases.items() if end>time.monotonic()}
                if "camera" in active:
                    active.remove("camera")
                    active.add("live")
                    with lock:
                        image,image_ts = shared["streams"].get("live",(b"",0.0))
                    if not image or time.time()-image_ts>cfg.fresh_s:
                        active.add("raw")
                session.requested_views = active
                for action in actions:
                    session.scene.command(action)
                result = session.poll()
                with lock:
                    changed = False
                    for name,value in session.streams.items():
                        old = shared["streams"].get(name,(None,0))
                        if value[0] is not old[0]:
                            shared["revisions"][name] = shared["revisions"].get(name,0)+1
                            changed = True
                    shared["streams"] = dict(session.streams)
                    shared["overlay"] = dict(session.overlay_info)
                    if session.detector:
                        shared["detector_status"] = dict(session.detector.status)
                    if session.source:
                        shared["sensor_ts"] = dict(session.source.sensor_ts)
                        shared["sensor_pose"] = Pose(**asdict(session.source.odom.pose))
                    if result is not None:
                        changed = changed or result.ts!=shared["scan"].ts
                        shared.update(scan=result,error="")
                    if changed:
                        notify()
                stop.wait(.2 if mock else .002)
        except Exception as e:
            with lock:
                shared["error"] = f"{type(e).__name__}: {e}"
                notify()
        finally:
            if session:
                session.close()

    @contextlib.asynccontextmanager
    async def lifespan(app):
        shared["loop"] = asyncio.get_running_loop()
        thread = threading.Thread(target=worker,daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=5)

    app = FastAPI(lifespan=lifespan)

    @app.get("/",response_class=HTMLResponse)
    def index():
        return _VIZ_PAGE

    @app.get("/scan")
    def scan_json():
        with lock:
            out = scan_to_dict(shared["scan"],cfg,mock)
            now = time.time()
            out["published_at"] = now
            out["streams"] = {name:{"ts":ts,"age_s":max(0,now-ts) if ts else None,
                                     "fresh":bool(data) and 0<=now-ts<=(3.0 if name=="boxes" else cfg.fresh_s)}
                              for name,(data,ts) in shared["streams"].items()}
            out["telemetry"] = {"pose":asdict(shared["sensor_pose"]),
                                "ages":{name:max(0,now-ts) if ts else None for name,ts in shared["sensor_ts"].items()}}
            out["detector"] = "simulated observations" if mock else (
                f"YOLO-World / requested {detector_backend}; see runtime status for actual backend" if detector else "ArUco only")
            out["detector_status"] = dict(shared["detector_status"])
            out["live_overlay"] = dict(shared["overlay"])
            if out["detector_status"].get("error"):
                out["warnings"].append(out["detector_status"]["error"])
            if shared["error"]:
                out["warnings"].append(shared["error"])
            return out

    @app.get("/objects")
    def object_json():
        d = scan_json()
        return {"schema_version":1,"snapshot_ts":d["ts"],"pose":d["pose"],"objects":d["objects"],
                "detector_status":d["detector_status"],
                "limitations":"surface estimates, not grasp poses; IDs are session-local; predictions never refresh geometry"}

    @app.get("/frame")
    def frame(view: str="rect"):
        if view not in views-{"camera"}:
            raise HTTPException(400,"unknown image view")
        with lock:
            leases[view] = time.monotonic()+3
            if mock:
                return Response(status_code=204)
            data,ts = shared["streams"].get(view,(b"",0.0))
            if not data or not 0<=time.time()-ts<=(3.0 if view=="boxes" else cfg.fresh_s):
                return Response(status_code=503,headers={"Retry-After":"1"})
            return Response(data,media_type="image/jpeg",headers={"Cache-Control":"no-store","X-Frame-Timestamp":str(ts)})

    @app.websocket("/live")
    async def live(socket: WebSocket):
        from urllib.parse import urlsplit
        selected = socket.query_params.get("view","camera")
        origin = socket.headers.get("origin")
        allowed_origin = not origin or urlsplit(origin).netloc in {socket.headers.get("host"),socket.headers.get("x-forwarded-host")}
        if origin and urlsplit(origin).hostname in {"localhost","127.0.0.1","::1"}:
            allowed_origin = True
        with lock:
            accepted = allowed_origin and selected in views and len(clients)<4
        if not accepted:
            await socket.close(code=1008)
            return
        await socket.accept()
        token,event = id(socket),asyncio.Event()
        with lock:
            accepted = len(clients)<4
            if accepted:
                clients[token] = {"view":selected,"event":event}
        if not accepted:
            await socket.close(code=1008)
            return
        disconnected = asyncio.Event()
        async def receive_control():
            try:
                message = await socket.receive()
                if message["type"]!="websocket.disconnect":
                    await socket.close(code=1008)
            except (WebSocketDisconnect,RuntimeError):
                pass
            finally:
                disconnected.set()
                event.set()
        receiver = asyncio.create_task(receive_control())
        last_key,last_sent,last_map,last_epoch = None,0.0,0.0,None
        try:
            while not stop.is_set() and not disconnected.is_set():
                event.clear()
                now = time.monotonic()
                with lock:
                    out = scan_json()
                    view = selected
                    if view=="camera":
                        view = "live" if out["streams"].get("live",{}).get("fresh") else "raw"
                    jpeg,ts = shared["streams"].get(view,(b"",0.0))
                    if not out["streams"].get(view,{}).get("fresh"):
                        jpeg = b""
                    key = (view,shared["revisions"].get(view,0),bool(jpeg))
                    if key==last_key and now-last_sent<.5:
                        send = False
                    else:
                        send = True
                        out["image"] = {"view":view,"timestamp":ts,"revision":key[1],"bytes":len(jpeg),
                                        "overlay":dict(shared["overlay"]) if view=="live" else None}
                        out["transport"] = "websocket"
                        epoch = out["pose"]["epoch"]
                        if now-last_map<.5 and last_epoch==epoch:
                            out.pop("surface_cells",None)
                        else:
                            last_map,last_epoch = now,epoch
                if send:
                    header = json.dumps(out,separators=(",",":"),allow_nan=False).encode()
                    await asyncio.wait_for(socket.send_bytes(struct.pack("!I",len(header))+header+jpeg),timeout=.5)
                    last_key,last_sent = key,now
                try:
                    await asyncio.wait_for(event.wait(),timeout=.2)
                except asyncio.TimeoutError:
                    pass
        except (WebSocketDisconnect,asyncio.TimeoutError,RuntimeError):
            pass
        finally:
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver
            with lock:
                clients.pop(token,None)
            with contextlib.suppress(RuntimeError,WebSocketDisconnect):
                await socket.close()

    @app.post("/mock/{action}")
    def mock_action(action: str):
        if not mock:
            raise HTTPException(403,"mock controls disabled on live robot")
        if action not in {"turn_left","turn_right","forward","back","visibility","anchor","pose","stack_next","reset"}:
            raise HTTPException(400,"unknown action")
        with lock:
            if len(shared["actions"])>=16:
                raise HTTPException(429,"mock command queue full")
            shared["actions"].append(action)
        return {"ok":True,"mock_only":True}

    print(f"[viz] http://{host}:{port} mock={mock} pose={pose_source} backend={detector_backend}; read-only sensors",flush=True)
    uvicorn.run(app,host=host,port=port,log_level="warning",ws="wsproto")


_VIZ_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crafter | Perception monitor</title>
<style>
:root{color-scheme:dark;--bg:#0b1018;--panel:#131d2a;--edge:#2a394f;--text:#e8effa;--muted:#9dafc6;--green:#62dfb6;--amber:#ffbf69;--blue:#74baff;--purple:#c7a5ff}
*{box-sizing:border-box}body{margin:0;padding:20px;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif;max-width:1800px;margin-inline:auto}
h1{font-size:25px;line-height:1.2;margin:0}h2{font-size:16px;margin:0}p{margin:5px 0;color:var(--muted)}small{color:var(--muted)}button,select,a.button{background:#1c2b3f;border:1px solid #405371;color:var(--text);border-radius:6px;font:inherit;padding:7px 10px;cursor:pointer;text-decoration:none}button:hover,select:hover{border-color:var(--blue)}button:focus-visible,select:focus-visible,a:focus-visible{outline:2px solid var(--blue)}a{color:var(--blue)}[hidden]{display:none!important}
header,.panelhead,.toolbar,.legend{display:flex;align-items:center;gap:10px;flex-wrap:wrap}header,.panelhead{justify-content:space-between}header{margin-bottom:16px}.eyebrow{font-size:11px;letter-spacing:.16em;color:var(--blue);margin-bottom:5px}.pill{display:inline-block;border:1px solid var(--edge);padding:4px 10px;border-radius:20px;font:12px ui-monospace,monospace}.ok{color:var(--green)}.warn{color:var(--amber)}.bad{color:#ff8b97}.health{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:12px 0}.health article{background:var(--panel);border:1px solid var(--edge);padding:12px 15px;border-radius:8px}.health strong{display:block;font-size:18px;margin:3px 0}.health small{display:block;min-height:20px}
.workspace,.lower{display:grid;grid-template-columns:1.1fr 1fr;gap:14px;margin-top:14px}.panel{min-width:0;background:var(--panel);border:1px solid var(--edge);border-radius:9px;overflow:hidden}.panelhead{padding:12px 14px;border-bottom:1px solid var(--edge)}.panelbody{padding:12px 14px}.camera-stage{background:#080e16;aspect-ratio:4/3;display:grid;place-items:center;position:relative}.camera-stage img{width:100%;height:100%;object-fit:contain;position:absolute;inset:0}.placeholder{text-align:center;max-width:380px;padding:24px;color:var(--muted)}.placeholder strong{display:block;color:var(--text);font-size:18px;margin-bottom:8px}.camera-caption{padding:10px 14px;min-height:58px;font-size:13px}.range-legend{padding:8px 14px;border-top:1px solid var(--edge)}.ramp{height:9px;border-radius:4px;background:linear-gradient(90deg,#30123b,#455ad1,#1bd0d5,#a4fc3c,#f8b52a,#7a0403);margin:5px 0}.range-labels{display:flex;justify-content:space-between;font:12px ui-monospace,monospace}.legend{font-size:12px;padding:9px 14px;color:var(--muted)}.legend i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}.legend span{white-space:nowrap}canvas{display:block;width:100%;background:#0b1521}#map{height:clamp(330px,43vw,590px)}#side{height:245px}.map-foot{padding:8px 14px;color:var(--muted);font-size:12px;border-top:1px solid var(--edge)}.notice{padding:10px 14px;background:#142238;border-left:3px solid var(--blue);border-radius:4px;color:#bdd3f0}.mock-controls{padding:10px 14px;background:#26231d;border:1px solid #5c4a2c;border-radius:6px;margin:10px 0}.mock-controls .toolbar{margin-top:8px}.table-wrap{overflow:auto;max-height:285px}table{border-collapse:collapse;width:100%;font:13px ui-monospace,monospace}th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--edge);white-space:nowrap}th{color:var(--muted);font-size:11px;text-transform:uppercase;position:sticky;top:0;background:var(--panel)}tr.selected{background:#243a50}tbody tr{cursor:pointer}tbody tr:hover{background:#1e2f43}.empty{padding:25px;text-align:center;color:var(--muted)}#selection{padding:9px 14px;color:var(--blue);font:12px ui-monospace,monospace;min-height:38px}.diagnostics{margin-top:14px;border:1px solid var(--edge);border-radius:8px;padding:12px 14px;background:var(--panel)}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 ui-monospace,monospace}#warnings{color:var(--amber)}footer{color:var(--muted);font-size:12px;margin:14px 0}.nowrap{white-space:nowrap}
@media(max-width:1050px){.workspace,.lower{grid-template-columns:1fr 1fr}body{padding:12px}.panelhead{align-items:flex-start}.health strong{font-size:16px}}
@media(max-width:760px){.workspace,.lower{grid-template-columns:1fr}.health{grid-template-columns:1fr 1fr}#map{height:420px}.camera-stage{max-height:65vh}h1{font-size:22px}}
</style></head><body>
<header><div><div class="eyebrow">CRAFTER / SENSOR WORKBENCH</div><h1>Perception monitor</h1><p>Inspect the scene. No motors, no autonomous actions.</p></div><div class="toolbar"><span id="mode" class="pill">CONNECTING</span><span id="connection" class="pill">Waiting for server</span><a href="/scan" target="_blank" rel="noopener" class="button">Snapshot JSON</a></div></header>
<div class="health">
<article><small>CAMERA</small><strong id="camera-state">Waiting</strong><small id="camera-detail">Checking image stream</small></article>
<article><small>DEPTH</small><strong id="depth-state">Waiting</strong><small id="depth-detail">Checking sparse pointcloud</small></article>
<article><small>ROBOT POSE</small><strong id="pose-state">Waiting</strong><small id="pose-detail">Wheel / IMU estimate</small></article>
<article><small>BUILD ANCHOR</small><strong id="anchor-state">Not observed</strong><small id="anchor-detail">Optional for scene inspection</small></article>
</div>
<div id="notice" class="notice">Camera and depth can be inspected without markers. Box identification is a separate stage.</div>
<div id="mock" class="mock-controls" hidden><strong>Simulation controls — fixture only, not the robot</strong><div class="toolbar"><button data-action="turn_left">Turn +45°</button><button data-action="turn_right">Turn −45°</button><button data-action="forward">Forward 15 cm</button><button data-action="back">Back 15 cm</button><button data-action="visibility">Toggle visibility</button><button data-action="anchor">Hide / show anchor</button><button data-action="pose">Lose / restore pose</button><button data-action="stack_next">Third box placement</button><button data-action="reset">Reset</button></div></div>
<main class="workspace">
<section class="panel"><div class="panelhead"><h2>Robot camera</h2><div class="toolbar"><select id="image-view" aria-label="Camera stream"><option value="camera">Live camera · auto</option><option value="live">Live overlays</option><option value="rect">Rectified + markers</option><option value="raw">Wide head camera</option><option value="range">Depth range</option><option value="boxes">Box detections · snapshot</option></select><a id="open-image" href="/frame" target="_blank" rel="noopener" class="button">Open image</a></div></div><div class="camera-stage"><div id="camera-empty" class="placeholder"><strong>Waiting for an image</strong>Checking camera and depth streams.</div><img id="cam" hidden alt="Robot camera or depth range image"></div><div id="range-legend" class="range-legend" hidden><div>Distance from camera · not height</div><div class="ramp"></div><div class="range-labels"><span>0 m</span><span>1.5 m</span><span>3 m+</span></div><small>Black pixels have no valid depth measurement.</small></div><div id="cameraLabel" class="camera-caption">Images are read-only. A box visible in RGB is not yet a tracked box.</div></section>
<section class="panel"><div class="panelhead"><h2>Surrounding map</h2><div class="toolbar"><select id="view" aria-label="Map frame"><option value="robot">Robot-centered</option><option value="world">Session world</option></select><select id="range" aria-label="Map radius"><option value="2">2 m radius</option><option value="1">1 m radius</option><option value="3">3 m radius</option><option value="fit">Fit objects</option></select></div></div><div class="legend"><span><i style="background:#62dfb6"></i>Loose</span><span><i style="background:#ffbf69"></i>Protected</span><span><i style="background:#c7a5ff"></i>Unknown</span><span>Dashed = memory, not a fresh candidate</span></div><canvas id="map" width="640" height="520"></canvas><div id="map-foot" class="map-foot">Blue cells are surface observations. Empty cells are unknown—not free space.</div></section>
</main>
<div class="lower"><section class="panel"><div class="panelhead"><h2>Height inspection</h2><small id="height-mode">Auto-scaled to observations</small></div><canvas id="side" width="640" height="245"></canvas><div class="map-foot">Objects sharing a horizontal position share a column. This is geometry, not proof of a successful placement.</div></section><section class="panel"><div class="panelhead"><h2>Box tracks</h2><span id="track-count" class="pill">0 tracks</span></div><div id="selection">Select a row to highlight its location.</div><div class="table-wrap"><table><thead><tr><th>ID / state</th><th>Age</th><th>Forward x</th><th>Left y</th><th>Height z</th></tr></thead><tbody id="rows"></tbody></table><div id="no-tracks" class="empty">No box tracks yet. Check the camera and detector status.</div></div></section></div>
<details class="diagnostics" open><summary>Sensor diagnostics and limitations</summary><div id="warnings"></div><pre id="info">Waiting for telemetry…</pre></details>
<footer>Read-only perception. Wheel/IMU dead reckoning drifts; this is not a navigation safety map. No pick/place controls are exposed here.</footer>
<script>
const $=id=>document.getElementById(id);
const cv=$('map'),ctx=cv.getContext('2d'),side=$('side'),sc=side.getContext('2d');
const colors={loose:'#62dfb6',protected:'#ffbf69',unknown:'#c7a5ff'};
let data=null, selected=null, frameKey='', bounds={w:640,h:520,scale:100}, labels=[];
let socket=null,pendingBundle=null,decoding=false,imageURL=null,generation=0,surfaceCache=[],surfaceEpoch=null,reconnectTimer=null;
let clientDrops=0,clientFrames=[],lastCapture=null;
const fmt=(v,n=2)=>Number.isFinite(v)?v.toFixed(n):'—';
const age=v=>Number.isFinite(v)?(v<1?Math.round(v*1000)+' ms':v.toFixed(1)+' s'):'no data';
const objectName=t=>t.name||('#'+t.id);
function tableTracks(d){return d.tracks.concat((d.objects||[]).filter(o=>!o.position_base_m).map(o=>({id:o.id,name:o.track_id,pos:[null,null,null],age:o.age_s,current:o.current,classification:'2D only',depth_status:o.depth_status,score:o.score,pick_candidate:false})));}
function prepare(canvas,c){const b=canvas.getBoundingClientRect(),w=Math.max(100,b.width),h=Math.max(100,b.height),dpr=Math.min(window.devicePixelRatio||1,2);if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,w,h);c.font='13px system-ui';return {w,h};}
function basePoint(world,d){const c=Math.cos(d.pose.yaw),s=Math.sin(d.pose.yaw),x=world[0]-d.pose.x,y=world[1]-d.pose.y;return [c*x+s*y,-s*x+c*y,world[2]];}
function framePoint(p,d){if($('view').value==='robot')return p;const c=Math.cos(d.pose.yaw),s=Math.sin(d.pose.yaw);return [d.pose.x+c*p[0]-s*p[1],d.pose.y+s*p[0]+c*p[1],p[2]];}
function radius(d){if($('range').value!=='fit')return Number($('range').value);const points=[[0,0,0],...d.tracks.map(t=>t.pos),...(d.build?d.build.cells:[])];return d.tracks.length?Math.max(.6,Math.min(5,Math.max(...points.map(p=>Math.max(...framePoint(p,d).slice(0,2).map(Math.abs))))+.25)):2;}
function px(p,d){const q=framePoint(p,d);return [bounds.w/2-q[1]*bounds.scale,bounds.h/2-q[0]*bounds.scale];}
function path(points,d,close=false){ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...px(p,d)):ctx.moveTo(...px(p,d)));if(close)ctx.closePath();}
function square(center,col,row,half,d){path([[1,1],[-1,1],[-1,-1],[1,-1]].map(([u,v])=>center.map((n,i)=>n+u*half*col[i]+v*half*row[i])),d,true);}
function label(text,p,color){const w=ctx.measureText(text).width+12,h=22;let r;for(const [dx,dy] of [[14,-30],[14,10],[-w-14,-30],[-w-14,12],[14,-55],[-w-14,36]]){const candidate={x:p[0]+dx,y:p[1]+dy,w,h};if(candidate.x<4||candidate.y<30||candidate.x+w>bounds.w-4||candidate.y+h>bounds.h-20)continue;if(!labels.some(a=>candidate.x<a.x+a.w&&candidate.x+w>a.x&&candidate.y<a.y+a.h&&candidate.y+h>a.y)){r=candidate;break;}}if(!r)return;labels.push(r);ctx.strokeStyle=color;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(r.x+w/2,r.y+h/2);ctx.stroke();ctx.fillStyle='#101c2a';ctx.fillRect(r.x,r.y,w,h);ctx.strokeRect(r.x,r.y,w,h);ctx.fillStyle='#eff6ff';ctx.fillText(text,r.x+6,r.y+15);}
function groups(d){const out=[];for(const t of d.tracks){let g=out.find(g=>Math.hypot(g[0].pos[0]-t.pos[0],g[0].pos[1]-t.pos[1])<.015);if(g)g.push(t);else out.push([t]);}return out;}
function drawMap(d){bounds=prepare(cv,ctx);const r=radius(d);bounds.scale=(Math.min(bounds.w,bounds.h)-72)/(2*r);labels=[];ctx.lineWidth=1;ctx.strokeStyle='#1b2b40';
for(let t=-r;t<=r+.001;t+=d.settings.resolution){ctx.beginPath();ctx.moveTo(bounds.w/2+t*bounds.scale,30);ctx.lineTo(bounds.w/2+t*bounds.scale,bounds.h-25);ctx.stroke();ctx.beginPath();ctx.moveTo(20,bounds.h/2+t*bounds.scale);ctx.lineTo(bounds.w-20,bounds.h/2+t*bounds.scale);ctx.stroke();}
for(const c of d.surface_cells){const height=Math.max(0,Math.min(1,c.z_max/1.8)),elapsed=c.last_seen?Math.max(0,d.published_at-c.last_seen):c.age;ctx.fillStyle=`rgba(49,${Math.round(90+height*110)},220,${Math.max(.06,.46*(1-elapsed/d.settings.memory_s))})`;const co=Math.cos(d.pose.yaw),si=Math.sin(d.pose.yaw);square(c.world?basePoint(c.world,d):c.pos,[co,-si,0],[si,co,0],d.settings.resolution/2,d);ctx.fill();}
if(d.build){ctx.lineWidth=1.5;ctx.strokeStyle=d.build.valid?'#ffbf69':'#806c53';for(const c of d.build.cells){square(c,d.build.col,d.build.row,d.settings.cell/2,d);ctx.stroke();}ctx.fillStyle='#ff8087';ctx.beginPath();ctx.arc(...px(d.build.marker,d),5,0,Math.PI*2);ctx.fill();label('Anchor 49',px(d.build.marker,d),'#ff8087');}
if(d.mock){ctx.strokeStyle='#537694';ctx.setLineDash([5,7]);path([[r*.8*Math.cos(.838),r*.8*Math.sin(.838),0],[0,0,0],[r*.8*Math.cos(.838),-r*.8*Math.sin(.838),0]],d);ctx.stroke();ctx.setLineDash([]);}
for(const group of groups(d)){const t=group[0],p=px(t.pos,d),color=colors[t.classification],current=group.some(t=>t.current),highlight=group.some(t=>t.id===selected);ctx.globalAlpha=current?1:.45;ctx.strokeStyle=highlight?'#fff':color;ctx.lineWidth=highlight?3:2;ctx.setLineDash(current?[]:[4,3]);ctx.beginPath();ctx.arc(...p,highlight?9:7,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);if(current){ctx.fillStyle=color;ctx.beginPath();ctx.arc(...p,3,0,Math.PI*2);ctx.fill();}ctx.globalAlpha=1;label(group.map(objectName).join(' / '),p,color);}
// robot
ctx.fillStyle='#74baff';path([[.13,0,0],[-.07,.065,0],[-.07,-.065,0]],d,true);ctx.fill();const robot=px([0,0,0],d);ctx.fillStyle='#a9d5ff';ctx.fillText('Robot',robot[0]+12,robot[1]+18);
ctx.fillStyle='#b4c5dc';ctx.fillText($('view').value==='robot'?'UP = forward +x   LEFT = +y':'Session world / fixed axes',14,19);ctx.fillText(`${Math.round(d.settings.resolution*100)} cm cells · ${fmt(r,1)} m radius`,14,bounds.h-8);
$('map-foot').textContent=`${d.surface_cells.length} observed surface cells · blank = unknown. ${d.stale?'Map snapshot STALE. ':''}${d.mock?'Dashed wedge is simulated visibility.':'No calibrated free-space or reachability inference.'}`;}
function drawHeights(d){const {w,h}=prepare(side,sc),gs=groups(d);if(!gs.length){sc.fillStyle='#9dafc6';sc.fillText('No box geometry to inspect yet.',25,h/2-8);sc.fillText('Camera/depth can be healthy without box tracks.',25,h/2+16);$('height-mode').textContent='Waiting for box detections';return;}
const anchored=d.build&&d.build.valid,datum=anchored?d.build.origin[2]:0;const lows=d.tracks.map(t=>t.pos[2]-t.size/2-datum),highs=d.tracks.map(t=>t.pos[2]+t.size/2-datum);const lo=(anchored?Math.min(0,...lows):Math.min(...lows))-.025,hi=Math.max(...highs)+.04,scale=(h-58)/(hi-lo),py=z=>h-33-(z-lo)*scale;
$('height-mode').textContent=d.objects?.length?'Visible-surface z estimates, not box centers':anchored?'Centimeters above build surface':'Absolute z · auto-zoomed';sc.font='12px system-ui';for(let i=0;i<=4;i++){const z=lo+(hi-lo)*i/4,y=py(z);sc.strokeStyle='#293b50';sc.beginPath();sc.moveTo(48,y);sc.lineTo(w-12,y);sc.stroke();sc.fillStyle='#9dafc6';sc.fillText(fmt(z*100,0),8,y+4);}
if(anchored){sc.strokeStyle='#ffbf69';sc.setLineDash([5,4]);sc.beginPath();sc.moveTo(48,py(0));sc.lineTo(w-12,py(0));sc.stroke();sc.setLineDash([]);sc.fillStyle='#ffbf69';sc.fillText('Build surface',55,py(0)-5);}
gs.forEach((g,i)=>{const x=60+(w-85)*(i+.5)/gs.length;for(const t of g){const top=py(t.pos[2]+t.size/2-datum),height=t.size*scale;sc.globalAlpha=t.current?1:.4;sc.fillStyle=colors[t.classification];sc.fillRect(x-15,top,30,Math.max(3,height-1));if(t.id===selected){sc.strokeStyle='#fff';sc.strokeRect(x-17,top-2,34,height+3);}if(height>14){sc.fillStyle='#071019';sc.fillText('#'+t.id,x-11,top+Math.min(height-3,16));}sc.globalAlpha=1;}sc.fillStyle='#dbe8f8';sc.fillText(g.map(t=>t.name?t.name.replace('box-','B'):'#'+t.id).join('/'),x-15,h-10);});}
function setHealth(name,value,detail,tone){$(name+'-state').textContent=value;$(name+'-state').className=tone;$(name+'-detail').textContent=detail;}
function camera(d){let view=$('image-view').value;if(d.transport==='websocket'&&d.image)view=d.image.view;else if(view==='camera')view=d.streams?.live?.fresh?'live':d.streams?.rect?.fresh?'rect':'raw';const s=d.streams?.[view];const image=$('cam'),empty=$('camera-empty');$('range-legend').hidden=view!=='range'||d.mock;
if(d.mock||!s?.fresh){image.hidden=true;empty.hidden=false;empty.textContent=d.mock?'Simulation has no camera pixels. Use the map controls to exercise memory and classification.':'Selected stream unavailable or stale. Try Camera · auto to inspect the independent head camera.';frameKey='';}else{const key=view+':'+s.ts;if(d.transport!=='websocket'&&frameKey!==key){frameKey=key;image.src='/frame?view='+view+'&t='+s.ts;}empty.hidden=true;image.hidden=false;}
$('open-image').href='/frame?view='+view;
$('cameraLabel').textContent=d.mock?'Mock geometry only; no computer-vision model runs on this scene.':({live:'LIVE rectified frames. Green = detector evidence; amber / tracked = short-lived visual prediction, NOT refreshed 3D evidence.',raw:'Wide left head camera. Independent of depth; no box recognition overlay.',rect:'Rectified 512×384 camera. Marker outlines only; untagged boxes are not identified.',range:'Sparse camera-range image: cool = nearer, warm = farther. Black = no valid depth.',boxes:'DELAYED detection snapshot. Labels, image and depth use the SAME captured frame. Scores are not calibrated probabilities.'}[view])+' Frame age: '+age(s?.age_s);}
function render(d){$('mode').textContent=d.mock?'SIMULATION':'LIVE · READ ONLY';$('mode').className='pill '+(d.mock?'warn':'ok');$('connection').textContent=d.stale?'Sensor snapshot stale':'Connected';$('connection').className='pill '+(d.stale?'warn':'ok');$('mock').hidden=!d.mock;
const streams=d.streams||{},ages=d.telemetry?.ages||{},cam=streams.raw?.fresh||streams.rect?.fresh||streams.live?.fresh||(Number.isFinite(ages.camera)&&ages.camera<d.settings.fresh_s),depth=!d.stale&&(d.diagnostics?.point_count||0)>0;const p=d.mock?d.pose:(d.telemetry?.pose||d.pose);
setHealth('camera',d.mock?'Simulated':cam?'Receiving':'No fresh image',d.mock?'No camera pixels':`Sensor ${age(ages.camera)} · displayed ${age(streams[d.image?.view||'live']?.age_s)}`,d.mock?'warn':cam?'ok':'bad');
setHealth('depth',d.mock?'Simulated':depth?'Receiving':'Unavailable',d.mock?d.surface_cells.length+' fixture surface cells':(d.diagnostics?.point_count||0).toLocaleString()+' points · '+age(ages.depth),d.mock?'warn':depth?'ok':'bad');
setHealth('pose',p.valid?p.source:'Unavailable',`x ${fmt(p.x)} · y ${fmt(p.y)} m · yaw ${fmt(p.yaw*180/Math.PI,1)}°`,p.valid?'ok':'bad');
setHealth('anchor',d.anchor_seen?'Seen':d.build?.valid?'Remembered':'Not set',d.build?'Last observed '+age(d.build.age):'Only needed for build-zone classification',d.build?.valid?'ok':'warn');
$('notice').textContent=d.mock?'Mock and live use the same classifier. These buttons only change simulated geometry.':d.detector_status?.enabled?`Detector: ${d.detector_status.runtime?.backend||d.detector_status.requested_backend||'loading'} ${d.detector_status.runtime?.precision||''} · ${fmt(d.detector_status.detector_fps,1)} detections/s · last pass ${fmt((d.detector_status.inference_s||0)*1000,0)} ms · p95 result age ${fmt(d.detector_status.result_age_p95_ms,0)} ms. Visual tracking does not refresh measured geometry.`:'Camera and depth inspection: active independently of markers. Enable --detector boxes for local markerless box proposals.';
$('track-count').textContent=tableTracks(d).length+' tracks · '+d.tracks.length+' on map';$('no-tracks').hidden=tableTracks(d).length>0;const rows=$('rows');rows.replaceChildren();for(const t of tableTracks(d)){const tr=document.createElement('tr');tr.className=t.id===selected?'selected':'';for(const v of [objectName(t)+' '+(t.name&&t.classification==='unknown'?'zone unassigned':t.classification)+(t.current?'':' / memory'),age(t.age),fmt(t.pos[0],3),fmt(t.pos[1],3),fmt(t.pos[2],3)]){const td=document.createElement('td');td.textContent=v;tr.appendChild(td);}tr.onclick=()=>{selected=t.id;render(data);};rows.appendChild(tr);}
const chosen=tableTracks(d).find(t=>t.id===selected);$('selection').textContent=chosen?`${objectName(chosen)} ${chosen.depth_status||''} score ${fmt(chosen.score)}: ${chosen.current?'observed now':'remembered only'} · ${chosen.pick_candidate?'candidate, reach/grasp NOT validated':'not a fresh pick candidate'}`:'Select a track row to highlight it in both views.';
$('warnings').textContent=d.warnings.join(' | ');$('info').textContent=`Capture pose epoch ${d.pose.epoch} · frame ${d.frame}\nWheel age ${age(ages.wheel)} · IMU age ${age(ages.imu)}\nDepth-to-forward yaw ${fmt(d.diagnostics?.depth_yaw_deg,1)}° · pose/camera skew ${age(d.diagnostics?.pose_camera_skew_s)}\n${d.detector||'Observation source: mock'}\n${p.warning||''}\n${d.map_semantics}`;
camera(d);drawMap(d);drawHeights(d);}
async function drain(){if(decoding)return;decoding=true;try{while(pendingBundle){const bundle=pendingBundle;pendingBundle=null;let url=null;
if(bundle.jpeg.byteLength){url=URL.createObjectURL(new Blob([bundle.jpeg],{type:'image/jpeg'}));const decoded=new Image();decoded.src=url;try{await decoded.decode();}catch(e){URL.revokeObjectURL(url);continue;}}
if(bundle.generation!==generation||pendingBundle){clientDrops++;if(url)URL.revokeObjectURL(url);continue;}
if(url){const previous=imageURL;imageURL=url;$('cam').src=url;if(previous)URL.revokeObjectURL(previous);}
data=bundle.meta;data.surface_cells=surfaceEpoch===data.pose.epoch?surfaceCache:[];render(data);const rendered=performance.now();if(bundle.jpeg.byteLength&&data.image.timestamp!==lastCapture){lastCapture=data.image.timestamp;clientFrames.push(rendered);}clientFrames=clientFrames.filter(t=>rendered-t<3000);const fps=clientFrames.length>1?(clientFrames.length-1)*1000/(clientFrames[clientFrames.length-1]-clientFrames[0]):0;data.client_receive_to_render_ms=rendered-bundle.received;$('connection').textContent=data.stale?'Live push · sensor stale':`Live push · ${fmt(fps,1)} FPS · render ${fmt(data.client_receive_to_render_ms,0)} ms · dropped ${clientDrops}`;
}}catch(e){disconnected('Render error: '+e.message);}finally{decoding=false;}}
function disconnected(message){$('connection').textContent='Disconnected';$('connection').className='pill bad';$('warnings').textContent=message;$('cam').hidden=true;$('camera-empty').hidden=false;$('camera-empty').textContent='No live stream. Last map is historical.';}
function connect(){const current=++generation;pendingBundle=null;clientFrames=[];lastCapture=null;if(reconnectTimer)clearTimeout(reconnectTimer);if(socket)socket.close();
const scheme=location.protocol==='https:'?'wss:':'ws:';socket=new WebSocket(scheme+'//'+location.host+'/live?view='+encodeURIComponent($('image-view').value));socket.binaryType='arraybuffer';
socket.onmessage=event=>{if(current!==generation)return;try{const bytes=event.data;if(!(bytes instanceof ArrayBuffer)||bytes.byteLength<4||bytes.byteLength>8*1024*1024)throw Error('invalid stream envelope');const length=new DataView(bytes).getUint32(0);if(length>2*1024*1024||length+4>bytes.byteLength)throw Error('invalid metadata length');const meta=JSON.parse(new TextDecoder().decode(new Uint8Array(bytes,4,length)));const jpeg=bytes.slice(4+length);if(jpeg.byteLength!==meta.image.bytes)throw Error('image/metadata length mismatch');if(meta.surface_cells){surfaceCache=meta.surface_cells;surfaceEpoch=meta.pose.epoch;}if(pendingBundle)clientDrops++;pendingBundle={meta,jpeg,generation:current,received:performance.now()};drain();}catch(e){disconnected(e.message);socket.close();}};
socket.onclose=()=>{if(current!==generation)return;disconnected('Stream closed; reconnecting without queuing old frames.');reconnectTimer=setTimeout(connect,1000);};socket.onerror=()=>{if(current===generation)disconnected('WebSocket unavailable');};}
$('cam').onerror=()=>{$('cam').hidden=true;$('camera-empty').hidden=false;$('camera-empty').textContent='Image decode failed; waiting for a fresh frame.';};
for(const b of document.querySelectorAll('[data-action]'))b.onclick=async()=>{try{const r=await fetch('/mock/'+b.dataset.action,{method:'POST'});if(!r.ok)throw Error(await r.text());}catch(e){$('warnings').textContent=e.message;}};
for(const id of ['view','range'])$(id).onchange=()=>data&&render(data);$('image-view').onchange=connect;window.addEventListener('resize',()=>data&&render(data));window.addEventListener('beforeunload',()=>{generation++;if(socket)socket.close();if(imageURL)URL.revokeObjectURL(imageURL);});connect();
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

        def test_depth_heading_matches_wheel_frame(self):
            matrix = np.array([[1,0,0,0],[0,-.629,.777,0],[0,-.777,-.629,1.62]])
            rotation, yaw = depth_heading_rotation(matrix)
            self.assertAlmostEqual(yaw, -90)
            np.testing.assert_allclose(rotation @ [0,1,.7], [1,0,.7], atol=1e-12)
            np.testing.assert_allclose(rotation @ [1,0,0], [0,-1,0], atol=1e-12)
            identity, _ = depth_heading_rotation(matrix, 0)
            np.testing.assert_allclose(identity, np.eye(3))

        def test_range_image_invalid_and_duplicate_pixels(self):
            points = np.array([[0,0,2], [0,0,1], [np.nan,0,1], [0,0,1], [0,0,0]], float)
            image = range_image(points, np.array([0,0,1,-1,2]), (2,2,3), [0,0,0])
            self.assertEqual(image.shape, (2,2,3))
            self.assertTrue(image[0,0].any())
            self.assertFalse(image[0,1].any())
            self.assertFalse(image[1].any())
            reference = range_image(np.array([[0,0,1.]]), np.array([0]), (2,2,3), [0,0,0])
            np.testing.assert_equal(image, reference)

        def test_raw_camera_available_without_depth(self):
            from unittest.mock import patch
            from types import SimpleNamespace
            raw = np.full((80,160,3), 120, np.uint8)
            fake = SimpleNamespace(raw_jpeg=_encode_jpeg(raw), raw_ts=time.time(), poll=lambda: None, close=lambda: None)
            with patch(__name__+'.CaptureWorker',return_value=fake):
                session = PerceptionSession()
                self.assertIsNone(session.poll())
                encoded, ts = session.streams['raw']
                decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(decoded.shape, (80,80,3))
                self.assertEqual(ts, fake.raw_ts)
                self.assertFalse(session.streams['rect'][0])
                session.close()

        def test_box_localization_rejects_flat_background(self):
            y,x = np.mgrid[:40,:40]
            points = np.column_stack((.5+x.ravel()*.005,y.ravel()*.005,np.zeros(1600)))
            ids = np.arange(1600)
            d = {"bbox":[14,14,26,26],"score":.3,"label":"cardboard_box"}
            flat = localize_box(d,points,ids,(40,40,3),np.array([0,0,1.6]))
            self.assertIsNone(flat['position_base_m'])
            self.assertEqual(flat['depth_status'],'background_or_flat_surface')
            on_box = (x.ravel()>=14)&(x.ravel()<=26)&(y.ravel()>=14)&(y.ravel()<=26)
            points[on_box,2] = .1
            box = localize_box(d,points,ids,(40,40,3),np.array([0,0,1.6]))
            self.assertEqual(box['depth_status'],'surface_supported')
            self.assertAlmostEqual(box['position_base_m'][2],.1)
            self.assertEqual(box['position_kind'],'visible_surface_centroid')
            self.assertIsNone(box['grasp_pose'])

        def test_box_missing_depth_is_still_2d(self):
            d = {"bbox":[2,2,12,12],"score":.3,"label":"cardboard_box"}
            item = localize_box(d,np.empty((0,3)),np.empty(0,dtype=int),(20,20,3),np.zeros(3))
            self.assertIsNone(item['position_base_m'])
            self.assertEqual(item['bbox'],d['bbox'])

        def test_object_identity_across_ego_motion(self):
            tracker = ObjectTracker()
            original = [.7,.1,.1]
            d = {'bbox':[10,10,30,30], 'position_base_m':original, 'score':.3}
            first = tracker.update([d],self.pose,self.t)[0]
            rotated = Pose(.1,0,.6,self.t+.1,'test',True)
            d2 = dict(d,position_base_m=rotated.to_base(original).tolist())
            second = tracker.update([d2],rotated,self.t+.1)[0]
            self.assertEqual(first['id'],second['id'])
            self.assertEqual(second['identity_status'],'tracked')
            export = tracker.snapshot(rotated,None,self.t+.2,self.cfg)[0]
            self.assertEqual(export['zone'],'unassigned')
            self.assertFalse(export['pick_candidate'])
            self.assertEqual(export['position_frame'],'base_at_pose_timestamp')

        def test_object_tracker_one_to_one_and_ambiguity(self):
            tracker = ObjectTracker()
            a = {'bbox':[0,0,10,10],'position_base_m':[.6,-.01,.1]}
            b = {'bbox':[20,0,30,10],'position_base_m':[.6,.01,.1]}
            initial = tracker.update([a,b],self.pose,self.t)
            self.assertEqual(len({d['id'] for d in initial}),2)
            merged = tracker.update([{'bbox':[5,0,25,10],'position_base_m':[.6,0,.1]}],self.pose,self.t+.1)[0]
            self.assertEqual(merged['identity_status'],'ambiguous')
            self.assertNotIn(merged['id'],[d['id'] for d in initial])

        def test_recent_object_beats_stale_nearby_track(self):
            tracker = ObjectTracker()
            d = {'bbox':[10,10,30,30],'position_base_m':[.6,.1,.1]}
            first = tracker.update([d],self.pose,self.t)[0]
            recent = dict(first,id=1001,track_id='box-002',last_seen=self.t+4,
                          bbox=[12,10,32,30],world_position_m=[.65,.1,.1])
            tracker.tracks[1001] = recent
            tracker.next_id = 1002
            result = tracker.update([dict(d,bbox=[13,10,33,30],position_base_m=[.64,.1,.1])],self.pose,self.t+5)[0]
            self.assertEqual(result['id'],1001)
            self.assertNotEqual(result['identity_status'],'ambiguous')

        def test_old_epoch_object_keeps_only_2d_evidence(self):
            tracker = ObjectTracker()
            tracker.update([{'bbox':[10,10,30,30],'position_base_m':[.6,.1,.1]}],self.pose,self.t)
            new_pose = Pose(ts=self.t+.1,source='test',valid=True,epoch=1)
            d = tracker.snapshot(new_pose,None,self.t+.1,self.cfg)[0]
            self.assertIsNone(d['position_base_m'])
            self.assertIsNone(d['world_position_m'])
            self.assertFalse(d['current'])
            self.assertEqual(d['identity_status'],'pose_epoch_changed')

        def test_object_memory_and_epoch_reset(self):
            tracker = ObjectTracker()
            d = {'bbox':[10,10,30,30],'position_base_m':None}
            first = tracker.update([d],self.pose,self.t)[0]
            tracker.update([],self.pose,self.t+1)
            self.assertFalse(tracker.snapshot(self.pose,None,self.t+1,self.cfg)[0]['current'])
            self.assertFalse(tracker.snapshot(self.pose,None,self.t+31,self.cfg))
            new = Pose(ts=self.t+2,source='test',valid=True,epoch=1)
            second = tracker.update([d],new,self.t+2)[0]
            self.assertNotEqual(first['id'],second['id'])

        def test_detector_tile_offsets_and_duplicate_suppression(self):
            detector = BoxDetector.__new__(BoxDetector)
            detector.threshold = .1
            replies = iter([[{'bbox':[160,220,210,260],'score':.8,'label':'cardboard_box'}],
                            [{'bbox':[32,28,82,68],'score':.9,'label':'cardboard_box'}]])
            detector._detect_single = lambda rgb: next(replies)
            result = detector.detect(np.zeros((384,512,3),np.uint8))
            self.assertEqual(len(result),1)
            np.testing.assert_allclose(result[0]['bbox'],[160,220,210,260])

        def test_detector_scores_not_double_sigmoided(self):
            from types import SimpleNamespace
            detector = BoxDetector.__new__(BoxDetector)
            detector.size,detector.threshold,detector.prompts = 512,.1,['cardboard box','chair']
            detector.embeddings = np.zeros((1,2,512),np.float32)
            output = np.array([[[100],[100],[40],[40],[.25],[-1e-7]]],np.float32)
            detector.session = SimpleNamespace(run=lambda *args: [output])
            result = detector._detect_single(np.zeros((384,512,3),np.uint8))
            self.assertEqual(len(result),1)
            self.assertAlmostEqual(result[0]['score'],.25)

        def test_detector_worker_preserves_capture_timestamp(self):
            import queue
            from types import SimpleNamespace
            worker = DetectorWorker.__new__(DetectorWorker)
            worker.jobs,worker.results = queue.Queue(1),queue.Queue(2)
            worker.process = SimpleNamespace(is_alive=lambda: True)
            worker.status = {'state':'ready'}
            worker.tracker = ObjectTracker()
            worker.overlay = LiveOverlay()
            worker.debug_image_requested = True
            worker.completions,worker.result_ages = deque(maxlen=120),deque(maxlen=120)
            worker.pending = (np.zeros((30,30,3),np.uint8),np.empty((0,3)),np.empty(0,dtype=int),self.t,self.pose)
            worker.results.put({'ts':self.t,'inference_s':.5,'detections':[{'bbox':[5,5,20,20],'score':.3,'label':'cardboard_box'}]})
            self.assertTrue(worker.poll(None,np.zeros(3)))
            self.assertEqual(worker.image[1],self.t)
            self.assertEqual(worker.status['localized'],0)
            self.assertEqual(worker.status['detections'],1)
            self.assertIsNone(worker.pending)

        def test_engine_cache_key_covers_runtime_and_vocab(self):
            from unittest.mock import patch
            values = np.zeros((1,len(DETECTOR_PROMPTS),512),np.float32)
            with patch(__name__+'._gpu_identity',return_value={'name':'test','sm':'87','cuda_driver':12060}):
                first = _engine_spec(512,'fp16',values,'10.3.0')
                self.assertEqual(first['image_shape'],[1,3,384,512])
                self.assertNotEqual(_cache_key(first),_cache_key(_engine_spec(512,'fp32',values,'10.3.0')))
                self.assertNotEqual(_cache_key(first),_cache_key(_engine_spec(640,'fp16',values,'10.3.0')))
                changed=values.copy(); changed[0,0,0]=1
                self.assertNotEqual(_cache_key(first),_cache_key(_engine_spec(512,'fp16',changed,'10.3.0')))

        def test_cached_embeddings_do_not_load_text_runtime(self):
            import builtins,tempfile
            from unittest.mock import patch
            values=np.zeros((1,len(DETECTOR_PROMPTS),512),np.float32); values[:,:,0]=1
            key=_cache_key({'assets':[(a[0],a[3]) for a in MODEL_ASSETS[1:]],'prompts':DETECTOR_PROMPTS})
            original=builtins.__import__
            def guarded(name,*args,**kwargs):
                if name in ('onnxruntime','tokenizers'):
                    raise AssertionError('cached embeddings must not reload the text model')
                return original(name,*args,**kwargs)
            with tempfile.TemporaryDirectory() as folder:
                np.save(Path(folder)/f'text-embeddings-{key}.npy',values,allow_pickle=False)
                with patch('builtins.__import__',side_effect=guarded):
                    np.testing.assert_equal(_text_embeddings(folder),values)

        def test_missing_engine_does_not_allocate_gpu_buffers(self):
            import tempfile
            from unittest.mock import patch
            with tempfile.TemporaryDirectory() as folder:
                with patch(__name__+'._tensorrt'),patch(__name__+'._engine_spec',return_value={'test':1}),patch(__name__+'.CudaRuntime') as cuda:
                    with self.assertRaises(FileNotFoundError):
                        TensorRTSession(folder,512,'fp16',np.empty(0))
                    cuda.assert_not_called()

        def test_worker_gpu_request_cannot_silently_fall_back(self):
            import queue
            from unittest.mock import patch
            results=queue.Queue()
            with patch('os.nice'),patch(__name__+'.BoxDetector',side_effect=FileNotFoundError('engine missing')) as detector:
                _detector_process(None,results,Path('/missing'),512,.1,'tensorrt','fp16')
                detector.assert_called_once_with(Path('/missing'),512,.1,'tensorrt','fp16')
            self.assertIn('engine missing',results.get_nowait()['error'])
            self.assertTrue(results.empty())

        def test_capture_worker_owns_readers_and_drops_old_packets(self):
            from types import SimpleNamespace
            ready=threading.Event(); identity={}
            class FakeSource:
                def __init__(self):
                    identity['created']=threading.get_ident()
                    self.raw_jpeg,self.raw_ts,self.sensor_ts=b'',0.0,{}
                    self.odom=SimpleNamespace(pose=Pose(valid=True))
                    self.camera_origin,self.depth_yaw_deg=np.zeros(3),0.0
                    self.count=0
                def poll(self):
                    self.count+=1
                    if self.count>=5:ready.set()
                    return (np.zeros((2,2,3),np.uint8),np.ones((1,3)),np.zeros(1,int),float(self.count),Pose(valid=True))
                def close(self):identity['closed']=threading.get_ident()
            capture=CaptureWorker(source_factory=FakeSource)
            try:
                self.assertTrue(ready.wait(1))
                packet=capture.poll()
                self.assertGreaterEqual(packet[3],5)
                self.assertFalse(packet[0].flags.writeable)
                self.assertGreater(capture.stats()['replaced_packets'],0)
            finally:capture.close()
            self.assertEqual(identity['created'],identity['closed'])
            self.assertNotEqual(identity['created'],threading.get_ident())

        def test_direct_scan_and_json_share_automatic_tracks(self):
            tracker=ObjectTracker()
            d={'bbox':[10,10,30,30],'position_base_m':[.6,.1,.1],'position_kind':'visible_surface_centroid',
               'depth_status':'surface_supported','score':.3}
            tracker.update([d],self.pose,self.t)
            objects=tracker.snapshot(self.pose,None,self.t+.1,self.cfg)
            s=Scan(ts=self.t,pose=self.pose)
            attached=_attach_object_tracks(_attach_object_tracks(s,objects),objects)
            self.assertEqual(len(attached.tracks),1)
            encoded=scan_to_dict(attached)
            self.assertEqual(len(encoded['tracks']),1)
            self.assertEqual(attached.tracks[0]['id'],encoded['tracks'][0]['id'])
            self.assertEqual(attached.tracks[0]['world'],encoded['tracks'][0]['world'])
            self.assertFalse(encoded['tracks'][0]['pick_candidate'])

        def test_visual_overlay_expiry_does_not_refresh_evidence(self):
            overlay=LiveOverlay()
            rgb=np.zeros((80,80,3),np.uint8)
            for y in range(20,60,8):
                for x in range(20,60,8):
                    if (x+y)//8%2:rgb[y:y+5,x:x+5]=255
            obj={'id':1000,'track_id':'box-001','bbox':[18,18,63,63],'score':.4,'identity_status':'tracked'}
            overlay.update(rgb,[obj],self.t,0)
            _,same=overlay.draw(rgb,self.t,0)
            self.assertEqual(same['source'],'detected')
            self.assertEqual(same['count'],1)
            shifted=cv2.warpAffine(rgb,np.array([[1,0,2],[0,1,1]],np.float32),(80,80))
            _,tracked=overlay.draw(shifted,self.t+.1,0)
            self.assertEqual(tracked['source'],'visual_tracking')
            self.assertEqual(tracked['count'],1)
            self.assertEqual(overlay.ts,self.t)
            _,expired=overlay.draw(shifted,self.t+1,0)
            self.assertEqual(expired['count'],0)
            _,reset=overlay.draw(shifted,self.t+.1,1)
            self.assertEqual(reset['count'],0)
            self.assertEqual(obj['bbox'],[18,18,63,63])

        def test_large_sensor_reads_are_paced(self):
            from types import SimpleNamespace
            from unittest.mock import Mock,patch
            source=LiveSource.__new__(LiveSource)
            source.next_head_poll=source.next_depth_poll=0.0
            source.head=SimpleNamespace(ready=Mock(return_value=False))
            source.points=SimpleNamespace(ready=Mock(return_value=False))
            source.imu=SimpleNamespace(ready=Mock(return_value=False))
            source.wheels=SimpleNamespace(ready=Mock(return_value=False))
            source.odom=SimpleNamespace(pose=Pose())
            for i in range(101):
                with patch('time.monotonic',return_value=i/1000):
                    self.assertIsNone(source.poll())
            self.assertEqual(source.head.ready.call_count,2)
            self.assertLessEqual(source.points.ready.call_count,5)
            self.assertEqual(source.wheels.ready.call_count,101)

        def test_cached_map_reprojects_without_rebuilding(self):
            world=WorldModel()
            cloud=np.tile([.61,.11,.2],(20,1))
            first=world.update([],self.pose,self.t,points=cloud)
            pose=Pose(.1,.2,.4,self.t+.1,'test',True)
            second=world.update([],pose,self.t+.1,points=cloud,map_update=False)
            self.assertEqual(len(first.surface_cells),len(second.surface_cells))
            self.assertEqual(first.surface_cells[0]['last_seen'],second.surface_cells[0]['last_seen'])
            np.testing.assert_allclose(second.surface_cells[0]['pos'],pose.to_base(first.surface_cells[0]['world']))

        def test_detector_dispatch_has_matching_depth_and_rejects_stale_input(self):
            import queue
            from types import SimpleNamespace
            worker=DetectorWorker.__new__(DetectorWorker)
            worker.results,worker.jobs=queue.Queue(),queue.Queue(1)
            worker.process=SimpleNamespace(is_alive=lambda:True)
            worker.status={'state':'ready'};worker.pending=None
            rgb=np.zeros((3,3,3),np.uint8);points=np.ones((2,3));indices=np.array([0,1])
            stale=(rgb,points,indices,self.t-10,self.pose)
            worker.poll(stale,np.zeros(3))
            self.assertTrue(worker.jobs.empty())
            self.assertEqual(worker.status['frames_skipped_stale'],1)
            current=(rgb,points,indices,time.time(),self.pose)
            worker.poll(current,np.zeros(3))
            payload=worker.jobs.get_nowait()
            self.assertEqual(payload[0],current[3])
            np.testing.assert_equal(payload[2],points.astype(np.float32))
            np.testing.assert_equal(payload[3],indices)
            self.assertEqual(worker.pending[3],current[3])

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
    ap.add_argument("--depth-yaw-deg", type=float, default=None, help="override depth-to-forward heading rotation")
    ap.add_argument("--box-size", type=float, default=BOX_SIZE)
    ap.add_argument("--build-cols", type=int, default=FOOTPRINT)
    ap.add_argument("--build-rows", type=int, default=FOOTPRINT)
    ap.add_argument("--capture", type=Path, help="save one live debug JPEG; refuses to overwrite")
    ap.add_argument("--image-view", choices=["live", "rect", "range", "raw", "boxes"], default="rect")
    ap.add_argument("--prepare-detector", action="store_true")
    ap.add_argument("--prepare-gpu-detector", action="store_true")
    ap.add_argument("--detector-backend", choices=["cpu","tensorrt"], default="cpu")
    ap.add_argument("--precision", choices=["fp16","fp32"], default="fp16")
    ap.add_argument("--benchmark-iterations", type=int, default=10)
    ap.add_argument("--model-cache", type=Path, default=MODEL_CACHE)
    ap.add_argument("--detector-benchmark", type=Path)
    ap.add_argument("--detector", choices=["aruco", "boxes"], default="aruco")
    ap.add_argument("--detector-size", type=int, default=512)
    ap.add_argument("--detector-threshold", type=float, default=.10)
    ap.add_argument("--benchmark-output", type=Path)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    cfg = Settings(box_size=a.box_size, cell=a.box_size, build_cols=a.build_cols,
                   build_rows=a.build_rows, depth_yaw_deg=a.depth_yaw_deg)
    if a.self_test:
        self_test()
    elif a.prepare_detector:
        prepare_detector(a.model_cache)
    elif a.prepare_gpu_detector:
        prepare_gpu_detector(a.model_cache,a.detector_size,a.precision)
    elif a.detector_benchmark:
        benchmark_detector(a.detector_benchmark,a.model_cache,a.detector_size,
                           a.detector_threshold,a.benchmark_output,a.detector_backend,a.precision,a.benchmark_iterations)
    elif a.viz:
        serve_viz(a.port, a.mock, cfg, a.pose_source, a.host, a.detector=="boxes",
                  a.model_cache,a.detector_size,a.detector_threshold,a.detector_backend,a.precision)
    elif a.capture:
        if a.mock:
            ap.error("capture requires live mode")
        session = PerceptionSession(settings=cfg, pose_source=a.pose_source,
                                    detector=a.detector=="boxes" or a.image_view=="boxes",
                                    model_cache=a.model_cache, detector_size=a.detector_size,
                                    detector_threshold=a.detector_threshold,
                                    detector_backend=a.detector_backend,precision=a.precision)
        try:
            deadline = time.monotonic()+15
            while time.monotonic() < deadline:
                session.poll()
                data, ts = session.streams[a.image_view]
                if data and 0 <= time.time()-ts < (3.0 if a.image_view=="boxes" else cfg.fresh_s):
                    with a.capture.open("xb") as f:
                        f.write(data)
                    print(f"Saved {a.image_view} frame to {a.capture}; acquisition timestamp {ts}")
                    if a.image_view=="boxes":
                        print(json.dumps(session.latest.objects, indent=2))
                    break
                time.sleep(.005)
            else:
                raise TimeoutError("requested live image stream unavailable")
        finally:
            session.close()
    else:
        print(json.dumps(scan_to_dict(scan(a.mock, settings=cfg, pose_source=a.pose_source), cfg, a.mock), indent=2, allow_nan=False))
