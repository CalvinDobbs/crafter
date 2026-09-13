import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main


class CliTests(unittest.TestCase):
    def test_default_mock_runs_agent_without_openai_or_hardware_imports(self):
        process = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "main.py"), "--mock"],
                                 cwd=tempfile.gettempdir(), capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
        result = json.loads(process.stdout.splitlines()[-1])
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["placed"], 4)

    def test_mock_auto_ignores_api_credentials(self):
        output = io.StringIO()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "not-a-real-key"}), contextlib.redirect_stdout(output):
            with patch("agent_backend.OpenAIReasoner", side_effect=AssertionError("unexpected API client")):
                self.assertEqual(main.main(["--mock"]), 0)

    def test_live_missing_provider_fails_without_hardware(self):
        process = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "main.py"),
                                  "--box-size", ".3", "--planner", "deterministic"],
                                 cwd=tempfile.gettempdir(), capture_output=True, text=True, timeout=10)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("provider", process.stderr)
        self.assertNotIn("homing", process.stdout)

    def test_mock_llm_requires_explicit_network_opt_in(self):
        process = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "main.py"), "--mock", "--planner", "llm"],
                                 capture_output=True, text=True, timeout=10)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("allow-api-with-mock", process.stderr)

    def test_help_and_incomplete_build(self):
        help_result = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "main.py"), "--help"],
                                     capture_output=True, text=True, timeout=10)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--provider", help_result.stdout)
        failed = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "main.py"), "--mock", "--max-steps", "1"],
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(failed.returncode, 0)
        self.assertNotIn('"status": "COMPLETED"', failed.stdout)


class PerceptionCliTests(unittest.TestCase):
    def snapshot(self):
        import time
        from agent_types import ObservationSnapshot, SceneImage
        now = time.time()
        return ObservationSnapshot(
            "live-1", now, now, 3, frame_id="session-world", valid=True, pose_valid=True,
            base_position=(1.0, 2.0, 0.0), base_yaw=0.0,
            images=(SceneImage("rect", "data:image/jpeg;base64,/9j/4A==", now, 3,
                               frame_id="session-world"),),
            world_model_json=json.dumps({"tracks": [{"id": 7, "classification": "protected"}],
                                         "map_semantics": "blank cells UNKNOWN"}))

    def test_perception_inspection_needs_no_action_provider_or_model(self):
        from agent_types import PerceptionCapabilities
        output = io.StringIO()
        snapshot = self.snapshot()
        with patch("agent_adapters.PerceptionObservations") as factory:
            provider = factory.return_value.__enter__.return_value
            provider.observe.return_value = snapshot
            provider.capabilities.return_value = PerceptionCapabilities(inventory=True, images=True)
            with patch.dict(os.environ, {"OPENAI_API_KEY": "not-a-real-key"}), contextlib.redirect_stdout(output):
                with patch("agent_backend.OpenAIReasoner", side_effect=AssertionError("unexpected model")), \
                     patch("agent_adapters.load_providers", side_effect=AssertionError("unexpected actions")):
                    self.assertEqual(main.main(["--observe-perception", "http://127.0.0.1:8007"]), 0)
            factory.assert_called_once_with("http://127.0.0.1:8007")
            factory.return_value.__exit__.assert_called_once()
        result = json.loads(output.getvalue())
        self.assertFalse(result["execution_enabled"])
        self.assertEqual(result["observation"]["world_model"]["tracks"][0]["classification"], "protected")
        self.assertIsNone(result["decision"])
        self.assertNotIn(snapshot.images[0].data_url, output.getvalue())
        self.assertFalse(result["observation"]["images"][0]["simulated"])

    def test_explicit_reasoning_receives_the_real_world_and_images_with_read_only_choices(self):
        from agent_types import PerceptionCapabilities, Step
        snapshot = self.snapshot()
        with patch("agent_adapters.PerceptionObservations") as factory, \
             patch("agent_backend.OpenAIReasoner") as reasoner, \
             patch("agent_backend.load_api_key", return_value="not-a-real-key"), \
             contextlib.redirect_stdout(io.StringIO()):
            provider = factory.return_value.__enter__.return_value
            provider.observe.return_value = snapshot
            provider.capabilities.return_value = PerceptionCapabilities(inventory=True, images=True)
            reasoner.return_value.decide.return_value = Step("stop", "Actions are not connected")
            self.assertEqual(main.main(["--observe-perception", "http://127.0.0.1:8007", "--planner", "llm"]), 0)
            context, choices = reasoner.return_value.decide.call_args.args
            self.assertEqual(context["world_model"], snapshot.world_model)
            self.assertEqual(context["images"][0]["data_url"], snapshot.images[0].data_url)
            self.assertEqual({step.operation for step in choices}, {"observe", "stop"})
            self.assertFalse(context["execution_enabled"])

    def test_inspection_waits_for_on_demand_camera_before_output_or_reasoning(self):
        from agent_types import PerceptionCapabilities, Step
        for planner in ("auto", "llm"):
            with self.subTest(planner=planner), patch("agent_adapters.PerceptionObservations") as factory, \
                 patch("agent_backend.OpenAIReasoner") as reasoner, \
                 patch("agent_backend.load_api_key", return_value="not-a-real-key"), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                snapshot = self.snapshot()
                provider = factory.return_value.__enter__.return_value
                provider.observe.side_effect = [replace(snapshot, images=()), snapshot]
                provider.capabilities.return_value = PerceptionCapabilities(inventory=True, images=True)
                reasoner.return_value.decide.return_value = Step("stop", "Read-only test")
                self.assertEqual(main.main(["--observe-perception", "http://127.0.0.1:8007", "--planner", planner]), 0)
                self.assertEqual(provider.observe.call_count, 2)
                self.assertEqual(len(json.loads(output.getvalue())["observation"]["images"]), 1)
                if planner == "llm":
                    context, _ = reasoner.return_value.decide.call_args.args
                    self.assertEqual(context["images"][0]["data_url"], snapshot.images[0].data_url)
                else:
                    reasoner.assert_not_called()

    def test_missing_camera_never_spends_a_model_request(self):
        from agent_types import PerceptionCapabilities, Step
        with patch("agent_adapters.PerceptionObservations") as factory, \
             patch("agent_backend.OpenAIReasoner") as reasoner, \
             patch("agent_backend.load_api_key", return_value="not-a-real-key") as key_loader, \
             patch("main.time.monotonic", side_effect=[0.0, 5.0]), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as error:
            provider = factory.return_value.__enter__.return_value
            provider.observe.return_value = replace(self.snapshot(), images=())
            provider.capabilities.return_value = PerceptionCapabilities(inventory=True, images=True)
            reasoner.return_value.decide.return_value = Step("stop")
            self.assertEqual(main.main(["--observe-perception", "http://127.0.0.1:8007", "--planner", "llm"]), 2)
            reasoner.assert_not_called()
            key_loader.assert_not_called()
            self.assertIn("image unavailable", error.getvalue())

    def test_data_only_inspection_still_reports_world_when_camera_wait_expires(self):
        from agent_types import PerceptionCapabilities
        with patch("agent_adapters.PerceptionObservations") as factory, \
             patch("main.time.monotonic", side_effect=[0.0, 5.0]), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            provider = factory.return_value.__enter__.return_value
            provider.observe.return_value = replace(self.snapshot(), images=())
            provider.capabilities.return_value = PerceptionCapabilities(inventory=True, images=True)
            self.assertEqual(main.main(["--observe-perception", "http://127.0.0.1:8007"]), 0)
            result = json.loads(output.getvalue())
            self.assertTrue(result["observation"]["world_model"])
            self.assertEqual(result["observation"]["images"], [])
            self.assertFalse(result["execution_enabled"])

    def test_perception_inspection_cannot_select_mock_ui_or_action_execution(self):
        for args in (["--mock"], ["--ui"], ["--provider", "other:create"], ["--mode", "oneshot"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main.main(["--observe-perception", "http://127.0.0.1:8007", *args])


if __name__ == "__main__":
    unittest.main()
