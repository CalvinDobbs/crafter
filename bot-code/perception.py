"""perception — cameras + depth -> list[Detection] of cardboard boxes.

Two detection paths, both behind detect_boxes():
  1. ArUco markers taped to the boxes (RECOMMENDED — reliable in bad light).
     cv2.aruco is bundled with opencv. Marker id -> Detection.id.
  2. Color/contour blob segmentation (fallback if no markers).

Position comes from `camera.points`: the depth daemon already publishes
base-frame xyz per pixel, so detection = find pixels of a box -> median of the
pointcloud under that pixel mask. No TF math needed.

Everything hardware-touching here is Reader-only, so this module can run live
on the robot WHILE mc_skills owns the arms — parallel dev is safe.

Offline dev:  python perception.py --mock    (serves fixtures/world_state.json)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from contracts import Detection, detections_from_dict, detections_to_dict

FIXTURE = Path(__file__).parent / "fixtures" / "world_state.json"
ARUCO_DICT = "DICT_4X4_50"   # marker id == box id


def _detect_aruco(rgb: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """-> [(marker_id, pixel_mask)] for each visible marker."""
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT)),
        cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    out = []
    if ids is None:
        return out
    for i, mid in enumerate(ids.flatten()):
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, corners[i][0].astype(np.int32), 1)
        out.append((int(mid), mask.astype(bool)))
    return out


def _mask_positions(mask: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    """Median base-frame xyz of pointcloud samples under a pixel mask.

    camera.points is aligned to camera.rect (the rectified head frame), which
    matches the rgb left image shape — VERIFY on first run, swap rgb half if off.
    """
    pts = points[mask.reshape(points.shape[:2])]
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = pts[np.linalg.norm(pts, axis=1) > 1e-3]
    if len(pts) < 5:
        return None
    return np.median(pts, axis=0)


def detect_boxes(mock=False) -> list[Detection]:
    """One scan -> the boxes currently visible. Call again after /rotate."""
    if mock:
        return detections_from_dict(json.loads(FIXTURE.read_text()))

    from bbos import Reader, Config
    cfg_head = Config("cam_head")
    with Reader("camera.head.rgb") as r_rgb, Reader("camera.points") as r_pts:
        while not (r_rgb.ready() and r_pts.ready()):
            pass
        img = np.asarray(r_rgb.data["rgb"])
        left, _right = cfg_head.split(img)
        pts = np.asarray(r_pts.data["points"])

    dets = []
    for mid, mask in _detect_aruco(left):
        pos = _mask_positions(mask, pts)
        if pos is not None:
            dets.append(Detection(id=mid, pos=[float(v) for v in pos]))
    return dets


def scan_all(skills=None, mock=False, sweeps=1) -> list[Detection]:
    """Detect, optionally /rotate between sweeps, merge by id (latest wins)."""
    found: dict[int, Detection] = {}
    for i in range(sweeps):
        for d in detect_boxes(mock=mock):
            found[d.id] = d
        if skills is not None and i + 1 < sweeps:
            skills.rotate(2 * np.pi / max(sweeps - 1, 1) * -1)  # scan motion
    return list(found.values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--save", default=str(FIXTURE))
    a = ap.parse_args()
    ds = detect_boxes(mock=a.mock)
    print(json.dumps(detections_to_dict(ds), indent=2))
    if a.save:
        Path(a.save).write_text(json.dumps(detections_to_dict(ds), indent=2))
