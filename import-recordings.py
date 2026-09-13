import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit


API_BASE = "https://api.bracketbot.com"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "recordings"
ARM_CHANNELS = {"left": "arm_left_state", "right": "arm_right_state"}


def request_json(url, *, api_key=None, body=None, expected_size=None):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Downloads require an HTTPS URL without embedded credentials.")
    if api_key is not None and parsed.netloc != urlsplit(API_BASE).netloc:
        raise ValueError("API credentials must only be sent to the Bracket Bot API.")
    config = [
        "url = " + json.dumps(url),
        'proto = "=https"',
        'header = "Accept: application/json"',
        "silent", "show-error", "connect-timeout = 10", "max-time = 60",
        'write-out = "\\n%{http_code}"',
    ]
    if api_key is not None:
        if not api_key or any(char.isspace() for char in api_key):
            raise ValueError("BB_API_KEY must be nonempty and contain no whitespace.")
        config.append("header = " + json.dumps("Authorization: Bearer " + api_key))
    if body is not None:
        config.extend([
            'header = "Content-Type: application/json"',
            "data = " + json.dumps(json.dumps(body)),
        ])
    try:
        response = subprocess.run(
            ["curl", "--disable", "--config", "-"], input="\n".join(config).encode(),
            capture_output=True, timeout=75,
        )
    except FileNotFoundError:
        raise RuntimeError("curl is required; install it and retry.") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError("HTTPS request timed out.") from None
    if response.returncode:
        raise RuntimeError(f"HTTPS request failed (curl exit {response.returncode}).")
    payload, separator, status = response.stdout.rpartition(b"\n")
    if not separator or not status.isdigit():
        raise RuntimeError("Missing HTTP response status.")
    code = int(status)
    if code != 200:
        detail = " Check the API key and its workspace access." if code in (401, 403) else ""
        raise RuntimeError(f"HTTPS request returned HTTP {code}.{detail}")
    if expected_size is not None and len(payload) != expected_size:
        raise ValueError("Downloaded JSON size does not match the manifest.")
    try:
        return json.loads(payload)
    except (ValueError, UnicodeError):
        raise ValueError("Server returned invalid JSON.") from None


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def position_series(sensors, channel):
    if not isinstance(sensors, dict) or not isinstance(sensors.get("series"), list):
        raise ValueError("Expected sensors.json with a series array.")
    matches = [row for row in sensors["series"] if isinstance(row, dict)
               and row.get("channel") == channel and row.get("field") == "pos"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {channel}.pos series; found {len(matches)}.")
    row = matches[0]
    times, values = row.get("t"), row.get("v")
    if not isinstance(times, list) or not times or not all(finite_number(t) for t in times):
        raise ValueError(f"{channel}.pos needs nonempty, finite timestamps.")
    if any(current <= previous for previous, current in zip(times, times[1:])):
        raise ValueError(f"{channel}.pos timestamps must be strictly increasing.")
    if row.get("dim") != 8 or not isinstance(values, list) or len(values) != 8:
        raise ValueError(f"{channel}.pos must contain eight joint arrays.")
    if any(not isinstance(axis, list) or len(axis) != len(times)
           or not all(finite_number(value) for value in axis) for axis in values):
        raise ValueError(f"{channel}.pos must have eight finite positions for every timestamp.")
    return times, values


def build_recording(sensors, name, *, saved_at=None):
    arms = {side: position_series(sensors, channel) for side, channel in ARM_CHANNELS.items()}
    start = max(0.0, *(times[0] for times, _ in arms.values()))
    timeline = sorted({t for times, _ in arms.values() for t in times if t >= start})
    if not timeline:
        raise ValueError("No nonnegative timestamps with measurements for both arms.")
    indices = {side: 0 for side in arms}
    frames = []
    for timestamp in timeline:
        frame = {"t": float(timestamp)}
        for side, (times, values) in arms.items():
            index = indices[side]
            while index + 1 < len(times) and times[index + 1] <= timestamp:
                index += 1
            indices[side] = index
            frame[side] = [float(axis[index]) for axis in values]
        frames.append(frame)
    return {"name": name, "saved_at": time.time() if saved_at is None else saved_at, "frames": frames}


def sensor_files(dataset, manifest):
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ValueError("Expected a download manifest with a files array.")
    prefix = f"{dataset}/_derived/v1/episodes/"
    suffix = "/analysis/sensors.json"
    episodes = {}
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("key"), str):
            raise ValueError("Invalid file entry in download manifest.")
        key = entry["key"]
        if not key.startswith(prefix) or not key.endswith(suffix):
            continue
        episode = key[len(prefix):-len(suffix)]
        if not episode or "/" in episode or episode in episodes:
            raise ValueError("Invalid or duplicate episode in download manifest.")
        if not isinstance(entry.get("url"), str) or type(entry.get("size")) is not int or entry["size"] < 0:
            raise ValueError("Sensor download entry requires a URL and byte size.")
        episodes[episode] = {**entry, "episode": episode}
    if not episodes:
        raise ValueError(f"No processed episode sensors.json files in {dataset}; cloud processing may be pending.")
    return [episodes[episode] for episode in sorted(episodes)]


def recording_name(dataset, episode):
    identity = f"{dataset}/{episode}"
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", f"{dataset}__{episode}").strip("_")[:120]
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    return f"{label or 'recording'}_{digest}"


def write_recording(recording, output_dir):
    name = recording["name"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Recording name must be a safe filename without an extension.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{name}.json"
    fd, temporary = tempfile.mkstemp(prefix=".import-", suffix=".tmp", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(recording, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink()
    return destination


def import_dataset(dataset, api_key, output_dir):
    manifest = request_json(API_BASE + "/v1/datasets/download", api_key=api_key, body={"dataset": dataset})
    failed = False
    for entry in sensor_files(dataset, manifest):
        name = recording_name(dataset, entry["episode"])
        destination = output_dir / f"{name}.json"
        if os.path.lexists(destination):
            print(f"Skipped existing: {destination}")
            continue
        try:
            sensors = request_json(entry["url"], expected_size=entry["size"])
            recording = build_recording(sensors, name)
            path = write_recording(recording, output_dir)
            print(f"Imported: {path} ({len(recording['frames'])} frames, {recording['frames'][-1]['t']:.3f}s)")
        except FileExistsError:
            print(f"Skipped existing: {destination}")
        except (OSError, ValueError, RuntimeError) as error:
            print(f"Error importing {dataset}/{entry['episode']}: {error}", file=sys.stderr)
            failed = True
    return not failed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Import only arm_left_state.pos and arm_right_state.pos from Bracket Bot Cloud into Mimic JSON.",
        epilog="Requires Python 3.10+, curl, and BB_API_KEY in the environment. The key selects the workspace. "
               "Frames use episode-relative seconds and the latest measured eight motor-turn positions from each arm. "
               "No interpolation, normalization, cloud deletion, or overwriting of existing files.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="list datasets without downloading recordings")
    mode.add_argument("--dataset", action="append", metavar="NAME", help="import a dataset; repeat to select several")
    mode.add_argument("--all", action="store_true", help="import all datasets accessible to the key")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="destination directory (default: recordings beside this script)")
    args = parser.parse_args(argv)
    api_key = os.environ.get("BB_API_KEY", "").strip()
    if not api_key:
        print("Error: set BB_API_KEY in your environment; do not put the key in this script.", file=sys.stderr)
        return 1
    try:
        response = request_json(API_BASE + "/v1/datasets", api_key=api_key)
        if not isinstance(response, dict) or not isinstance(response.get("datasets"), list):
            raise ValueError("Expected a datasets array from the API.")
        datasets = response["datasets"]
        if any(not isinstance(item, dict) or not isinstance(item.get("name"), str) for item in datasets):
            raise ValueError("Invalid dataset entry from the API.")
        names = sorted({item["name"] for item in datasets})
        if args.list:
            for item in sorted(datasets, key=lambda item: item["name"]):
                print(f"{item['name']}\t{item.get('files', '?')} files\t{item.get('totalSize', '?')} bytes")
            if not names:
                print("No datasets in this key's workspace.")
            return 0
        selected = names if args.all else list(dict.fromkeys(args.dataset))
        unknown = set(selected) - set(names)
        if unknown:
            raise ValueError("Dataset not available to this key: " + ", ".join(sorted(unknown)))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if not selected:
            print("No datasets in this key's workspace.")
        failed = False
        for dataset in selected:
            try:
                if not import_dataset(dataset, api_key, args.output_dir):
                    failed = True
            except (OSError, ValueError, RuntimeError) as error:
                print(f"Error importing {dataset}: {error}", file=sys.stderr)
                failed = True
        return int(failed)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Import interrupted.", file=sys.stderr)
        raise SystemExit(130)
