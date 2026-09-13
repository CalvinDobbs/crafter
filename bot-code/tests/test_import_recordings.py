import contextlib
import copy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("import_recordings", ROOT / "import-recordings.py")
importer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(importer)


def series(channel, times, offset=0.0, field="pos"):
    return {
        "id": f"{channel}.{field}", "channel": channel, "field": field,
        "dim": 8, "n": len(times), "t": times,
        "v": [[offset + joint + sample / 10 for sample in range(len(times))] for joint in range(8)],
    }


def sensors():
    return {"t0_ns": 1789271899744008486, "series": [
        series("arm_left_ctrl", [0.0, 0.1], 100.0),
        series("arm_left_state", [0.02, 0.05, 0.08]),
        series("arm_right_state", [0.03, 0.06, 0.08], -10.0),
        series("arm_right_state", [0.03, 0.06, 0.08], 200.0, field="vel"),
    ]}


class RecordingConversionTests(unittest.TestCase):
    def test_matches_mimic_schema_and_keeps_measured_positions(self):
        source = sensors()
        original = copy.deepcopy(source)
        result = importer.build_recording(source, "example", saved_at=1234.5)
        self.assertEqual(list(result), ["name", "saved_at", "frames"])
        self.assertEqual(result["name"], "example")
        self.assertEqual(result["saved_at"], 1234.5)
        self.assertEqual([frame["t"] for frame in result["frames"]], [0.03, 0.05, 0.06, 0.08])
        for frame in result["frames"]:
            self.assertEqual(list(frame), ["t", "left", "right"])
            self.assertEqual(len(frame["left"]), 8)
            self.assertEqual(len(frame["right"]), 8)
        self.assertEqual(result["frames"][0]["left"], [float(j) for j in range(8)])
        self.assertEqual(result["frames"][0]["right"], [float(j - 10) for j in range(8)])
        self.assertEqual(result["frames"][1]["right"], result["frames"][0]["right"])
        self.assertEqual(result["frames"][2]["left"], result["frames"][1]["left"])
        self.assertEqual(result["frames"][-1]["left"], [j + 0.2 for j in range(8)])
        self.assertEqual(source, original)
        json.dumps(result, allow_nan=False)

    def test_transposes_even_when_sample_count_equals_joint_count(self):
        times = [float(i) for i in range(8)]
        data = {"series": [series("arm_left_state", times), series("arm_right_state", times)]}
        frames = importer.build_recording(data, "eight")["frames"]
        self.assertEqual(frames[2]["left"], [j + 0.2 for j in range(8)])

    def test_carries_last_sample_to_final_update_without_fabricating_positions(self):
        data = {"series": [series("arm_left_state", [0.1]), series("arm_right_state", [0.1, 0.2])]}
        frames = importer.build_recording(data, "end")["frames"]
        self.assertEqual([f["t"] for f in frames], [0.1, 0.2])
        self.assertEqual(frames[0]["left"], frames[1]["left"])

    def test_pre_episode_samples_can_seed_but_never_emit_negative_time(self):
        data = {"series": [series("arm_left_state", [-0.1, 0.1]), series("arm_right_state", [-0.2, 0.2])]}
        frames = importer.build_recording(data, "start")["frames"]
        self.assertEqual([frame["t"] for frame in frames], [0.1, 0.2])
        self.assertEqual(frames[0]["right"], [float(j) for j in range(8)])

    def test_missing_or_duplicate_position_channel_is_rejected(self):
        data = sensors()
        del data["series"][1]
        with self.assertRaisesRegex(ValueError, "arm_left_state"):
            importer.build_recording(data, "missing")
        data = sensors()
        data["series"].append(copy.deepcopy(data["series"][1]))
        with self.assertRaisesRegex(ValueError, "arm_left_state"):
            importer.build_recording(data, "duplicate")

    def test_invalid_positions_and_timestamps_are_rejected(self):
        mutations = [
            lambda row: row.update(dim=7),
            lambda row: row.update(t=[]),
            lambda row: row.update(t=[0.02, 0.02, 0.08]),
            lambda row: row.update(t=[0.05, 0.02, 0.08]),
            lambda row: row.update(t=[0.02, math.nan, 0.08]),
            lambda row: row.update(t=[0.02, True, 0.08]),
            lambda row: row["v"].pop(),
            lambda row: row["v"][0].pop(),
            lambda row: row["v"][0].__setitem__(0, None),
            lambda row: row["v"][0].__setitem__(0, math.inf),
            lambda row: row["v"][0].__setitem__(0, "0.2"),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                data = sensors()
                mutate(data["series"][1])
                with self.assertRaises(ValueError):
                    importer.build_recording(data, "invalid")

    def test_no_nonnegative_frames_is_rejected(self):
        data = {"series": [series("arm_left_state", [-0.1]), series("arm_right_state", [-0.2])]}
        with self.assertRaises(ValueError):
            importer.build_recording(data, "empty")


class RecordingImportTests(unittest.TestCase):
    def manifest(self):
        prefix = "category/run/_derived/v1/episodes"
        return {"files": [
            {"key": f"{prefix}/ep000001_ab/analysis/sensors.json", "url": "https://storage.example/one", "size": 1},
            {"key": f"{prefix}/ep000000_cd/analysis/sensors.json", "url": "https://storage.example/zero", "size": 2},
            {"key": f"{prefix}/ep000000_cd/analysis/features.json", "url": "https://storage.example/features", "size": 3},
            {"key": f"{prefix}/ep000000_cd/video/head.mp4", "url": "https://storage.example/video", "size": 4},
            {"key": "other/run/_derived/v1/episodes/ep/analysis/sensors.json", "url": "https://storage.example/other", "size": 5},
        ]}

    def test_selects_only_episode_sensor_files_for_requested_dataset(self):
        files = importer.sensor_files("category/run", self.manifest())
        self.assertEqual([item["episode"] for item in files], ["ep000000_cd", "ep000001_ab"])
        with self.assertRaises(ValueError):
            importer.sensor_files("missing", self.manifest())

    def test_output_names_are_safe_and_distinguish_collisions(self):
        names = [importer.recording_name(dataset, episode) for dataset, episode in [
            ("a/b", "ep"), ("a_b", "ep"), ("../../escape", "../ep"), ("x" * 500, "ep"),
        ]]
        self.assertEqual(len(set(names)), len(names))
        for name in names:
            self.assertRegex(name, r"^[A-Za-z0-9_-]+$")
            self.assertLess(len(name), 200)
        self.assertEqual(names[0], importer.recording_name("a/b", "ep"))

    def test_write_matches_reference_serialization_and_refuses_overwrite(self):
        recording = importer.build_recording(sensors(), "test", saved_at=1.0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "recordings"
            path = importer.write_recording(recording, output)
            self.assertEqual(path.parent, output)
            self.assertEqual(path.read_text(), json.dumps(recording))
            with self.assertRaises(FileExistsError):
                importer.write_recording(recording, output)
            self.assertEqual(json.loads(path.read_text()), recording)
            self.assertEqual(list(output.iterdir()), [path])

    def test_http_keeps_key_off_command_line_and_checks_download_size(self):
        response = subprocess.CompletedProcess([], 0, b'{"ok": true}\n200', b'')
        with patch.object(importer.subprocess, "run", return_value=response) as run:
            self.assertEqual(importer.request_json(importer.API_BASE + "/v1/datasets", api_key="test-key"), {"ok": True})
            args, kwargs = run.call_args
            self.assertNotIn("test-key", str(args))
            self.assertIn(b"Authorization: Bearer test-key", kwargs["input"])
            self.assertEqual(args[0][1], "--disable")
            self.assertGreater(kwargs["timeout"], 0)
        with patch.object(importer.subprocess, "run", return_value=response):
            with self.assertRaisesRegex(ValueError, "size"):
                importer.request_json("https://storage.example/data", expected_size=500)

    def test_rejects_credential_forwarding_and_insecure_urls(self):
        for url, key in [("http://storage.example/data", None), ("https://storage.example/data", "test-key")]:
            with self.subTest(url=url), patch.object(importer.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    importer.request_json(url, api_key=key)
                run.assert_not_called()

    def test_http_failure_does_not_expose_response_or_secrets(self):
        for status in (302, 401, 403, 500):
            response = subprocess.CompletedProcess([], 0, b'secret-content\n' + str(status).encode(), b'signed-url-secret')
            with self.subTest(status=status), patch.object(importer.subprocess, "run", return_value=response):
                with self.assertRaises(RuntimeError) as error:
                    importer.request_json(importer.API_BASE + "/v1/datasets", api_key="test-key")
                self.assertIn(str(status), str(error.exception))
                self.assertNotIn("secret", str(error.exception))
                self.assertNotIn("test-key", str(error.exception))

    def test_main_imports_all_episodes_and_skips_existing_files(self):
        calls = []

        def request(url, **kwargs):
            calls.append((url, kwargs))
            if url.endswith("/v1/datasets"):
                return {"datasets": [{"name": "category/run", "files": 5, "totalSize": 100}]}
            if url.endswith("/v1/datasets/download"):
                self.assertEqual(kwargs["body"], {"dataset": "category/run"})
                return self.manifest()
            self.assertNotIn("api_key", kwargs)
            return sensors()

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"BB_API_KEY": "test-key"}), \
                patch.object(importer, "request_json", side_effect=request), contextlib.redirect_stdout(io.StringIO()):
            args = ["--all", "--output-dir", directory]
            self.assertEqual(importer.main(args), 0)
            paths = list(Path(directory).glob("*.json"))
            self.assertEqual(len(paths), 2)
            before = {path.name: path.read_bytes() for path in paths}
            self.assertEqual(importer.main(args), 0)
            self.assertEqual(before, {path.name: path.read_bytes() for path in paths})
        downloads = [url for url, _ in calls if url.startswith("https://storage.example/")]
        self.assertEqual(downloads, ["https://storage.example/zero", "https://storage.example/one"])
        self.assertFalse(any("delete" in url for url, _ in calls))

    def test_list_does_not_download(self):
        response = {"datasets": [{"name": "category/run", "files": 1, "totalSize": 10}]}
        with patch.dict(os.environ, {"BB_API_KEY": "test-key"}), \
                patch.object(importer, "request_json", return_value=response) as request, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(importer.main(["--list"]), 0)
        request.assert_called_once()
        self.assertIn("category/run", output.getvalue())

    def test_missing_key_fails_without_network(self):
        with patch.dict(os.environ, {"BB_API_KEY": ""}), patch.object(importer, "request_json") as request, \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertNotEqual(importer.main(["--all"]), 0)
        request.assert_not_called()
        self.assertIn("BB_API_KEY", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
