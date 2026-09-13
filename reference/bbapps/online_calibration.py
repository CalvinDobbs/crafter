#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.10.*"
# dependencies = [
#   "bbos",
#   "numpy",
#   "opencv-python-headless",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Tune stereo rotation from live images while keeping intrinsics and T fixed."""

import argparse
import time
from itertools import product

import cv2
import numpy as np
from bbos import Config, Reader


def capture_frames(count, sample_every):
    frames = []
    seen = 0
    with Reader("camera.head.jpeg", keeptime=False, sync=True) as reader:
        deadline = time.monotonic() + 30.0
        while len(frames) < count and time.monotonic() < deadline:
            ready = reader.ready()
            if ready:
                seen += 1
                if seen % sample_every == 0:
                    n = int(reader.data["jpeg_len"])
                    encoded = np.frombuffer(bytes(reader.data["jpeg"][:n]), np.uint8)
                    stereo = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
                    if stereo is not None:
                        frames.append(stereo)
    if len(frames) < count:
        raise RuntimeError(f"captured only {len(frames)}/{count} frames")
    return frames


def load_calibration(path):
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise RuntimeError(f"cannot read {path}")
    names = ("mtx_l", "dist_l", "mtx_r", "dist_r", "R", "T", "R1", "R2", "P1", "P2", "Q")
    matrices = {name: storage.getNode(name).mat() for name in names}
    storage.release()
    return matrices


def find_matches(frames, eye_width):
    orb = cv2.ORB_create(6000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    left_points = []
    right_points = []
    for stereo in frames:
        left = np.ascontiguousarray(stereo[:, :eye_width])
        right = np.ascontiguousarray(stereo[:, eye_width:])
        keys_l, desc_l = orb.detectAndCompute(left, None)
        keys_r, desc_r = orb.detectAndCompute(right, None)
        if desc_l is not None and desc_r is not None:
            pairs = matcher.knnMatch(desc_l, desc_r, k=2)
            good = [
                match
                for pair in pairs
                if len(pair) == 2
                for match, other in [pair]
                if match.distance < 0.75 * other.distance
            ]
            left_points.extend(keys_l[m.queryIdx].pt for m in good)
            right_points.extend(keys_r[m.trainIdx].pt for m in good)
    if len(left_points) < 100:
        raise RuntimeError(f"only {len(left_points)} stereo matches")
    return (
        np.asarray(left_points, np.float64).reshape(-1, 1, 2),
        np.asarray(right_points, np.float64).reshape(-1, 1, 2),
    )


def rectify(K1, D1, K2, D2, size, R, T):
    return cv2.fisheye.stereoRectify(
        K1,
        D1,
        K2,
        D2,
        size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        balance=0.0,
        fov_scale=1.0,
    )


def vertical_errors(points_l, points_r, K1, D1, K2, D2, R1, R2, P1, P2):
    rect_l = cv2.fisheye.undistortPoints(points_l, K1, D1, R=R1, P=P1)
    rect_r = cv2.fisheye.undistortPoints(points_r, K2, D2, R=R2, P=P2)
    return np.abs(rect_l[:, 0, 1] - rect_r[:, 0, 1])


def score(errors):
    return float(np.median(errors) + 0.25 * np.percentile(errors, 80))


def search_rotation(points_l, points_r, K1, D1, K2, D2, size, R, T):
    center_deg = np.zeros(3, np.float64)
    stages = ((1.5, 0.5), (0.45, 0.15), (0.12, 0.04))
    for radius, step in stages:
        offsets = np.arange(-radius, radius + step / 2, step)
        best = None
        for offset in product(offsets, repeat=3):
            delta_deg = center_deg + np.asarray(offset)
            delta_R, _ = cv2.Rodrigues(np.radians(delta_deg).reshape(3, 1))
            candidate_R = delta_R @ R
            R1, R2, P1, P2, Q = rectify(K1, D1, K2, D2, size, candidate_R, T)
            errors = vertical_errors(points_l, points_r, K1, D1, K2, D2, R1, R2, P1, P2)
            result = (score(errors), delta_deg, candidate_R, R1, R2, P1, P2, Q)
            if best is None or result[0] < best[0]:
                best = result
        center_deg = best[1]
        print(
            f"grid radius={radius:.2f}deg step={step:.2f}deg "
            f"delta={center_deg} score={best[0]:.4f}px",
            flush=True,
        )
    return best


def epi_errors(frames, eye_width, K1, D1, K2, D2, R1, R2, P1, P2):
    size = (eye_width, frames[0].shape[0])
    map1x, map1y = cv2.fisheye.initUndistortRectifyMap(
        K1, D1, R1, P1, size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.fisheye.initUndistortRectifyMap(
        K2, D2, R2, P2, size, cv2.CV_32FC1
    )
    orb = cv2.ORB_create(6000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    errors = []
    for stereo in frames:
        left = cv2.remap(stereo[:, :eye_width], map1x, map1y, cv2.INTER_LINEAR)
        right = cv2.remap(stereo[:, eye_width:], map2x, map2y, cv2.INTER_LINEAR)
        keys_l, desc_l = orb.detectAndCompute(left, None)
        keys_r, desc_r = orb.detectAndCompute(right, None)
        if desc_l is not None and desc_r is not None:
            pairs = matcher.knnMatch(desc_l, desc_r, k=2)
            good = [
                match
                for pair in pairs
                if len(pair) == 2
                for match, other in [pair]
                if match.distance < 0.75 * other.distance
            ]
            errors.extend(
                abs(keys_l[m.queryIdx].pt[1] - keys_r[m.trainIdx].pt[1]) for m in good
            )
    return np.asarray(errors)


def print_stats(label, errors):
    print(
        f"{label}: n={len(errors)} median|dy|={np.median(errors):.4f}px "
        f"p90={np.percentile(errors, 90):.4f}px "
        f"|dy|<1={(errors < 1).mean() * 100:.2f}% "
        f"|dy|<2={(errors < 2).mean() * 100:.2f}%"
    )


def write_calibration(path, matrices):
    storage = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise RuntimeError(f"cannot write {path}")
    for name, matrix in matrices.items():
        storage.write(name, matrix)
    storage.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--sample-every", type=int, default=6)
    parser.add_argument("--output")
    args = parser.parse_args()

    cam = Config("cam_head")
    depth = Config("depth")
    calibration = load_calibration(depth.calib_path)
    K1, D1 = calibration["mtx_l"], calibration["dist_l"]
    K2, D2 = calibration["mtx_r"], calibration["dist_r"]
    R, T = calibration["R"], calibration["T"]
    R1, R2 = calibration["R1"], calibration["R2"]
    P1, P2 = calibration["P1"], calibration["P2"]
    size = (cam.width // 2, cam.height)

    print(f"calibration={depth.calib_path}")
    print(f"baseline={np.linalg.norm(T):.6f}mm (fixed); intrinsics/distortion fixed")
    frames = capture_frames(args.frames, args.sample_every)
    split = max(1, int(len(frames) * 2 / 3))
    train_frames, test_frames = frames[:split], frames[split:]
    train_l, train_r = find_matches(train_frames, size[0])
    test_l, test_r = find_matches(test_frames, size[0])
    print(f"frames train={len(train_frames)} test={len(test_frames)} matches train={len(train_l)} test={len(test_l)}")

    current_train = vertical_errors(train_l, train_r, K1, D1, K2, D2, R1, R2, P1, P2)
    current_test = vertical_errors(test_l, test_r, K1, D1, K2, D2, R1, R2, P1, P2)
    print_stats("current fixed-match train", current_train)
    print_stats("current fixed-match test", current_test)

    best = search_rotation(train_l, train_r, K1, D1, K2, D2, size, R, T)
    _, delta_deg, candidate_R, new_R1, new_R2, new_P1, new_P2, new_Q = best
    candidate_test = vertical_errors(
        test_l, test_r, K1, D1, K2, D2, new_R1, new_R2, new_P1, new_P2
    )
    current_epi = epi_errors(test_frames, size[0], K1, D1, K2, D2, R1, R2, P1, P2)
    candidate_epi = epi_errors(
        test_frames, size[0], K1, D1, K2, D2, new_R1, new_R2, new_P1, new_P2
    )
    print(f"rotation_delta_deg={delta_deg}")
    print(f"candidate_R=\n{candidate_R}")
    print(f"T_mm={T.ravel()} (unchanged)")
    print_stats("candidate fixed-match test", candidate_test)
    print_stats("current epi_test-style test", current_epi)
    print_stats("candidate epi_test-style test", candidate_epi)

    accepted = score(candidate_test) < score(current_test) and score(candidate_epi) < score(current_epi)
    print(f"accepted={accepted}")
    if args.output and accepted:
        write_calibration(
            args.output,
            {
                "mtx_l": K1,
                "dist_l": D1,
                "mtx_r": K2,
                "dist_r": D2,
                "R": candidate_R,
                "T": T,
                "R1": new_R1,
                "R2": new_R2,
                "P1": new_P1,
                "P2": new_P2,
                "Q": new_Q,
            },
        )
        print(f"wrote={args.output}")
    elif args.output:
        print("candidate rejected; output not written")


if __name__ == "__main__":
    main()
