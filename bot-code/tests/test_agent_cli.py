import contextlib
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main


class CliTests(unittest.TestCase):
    def test_default_mock_runs_agent_without_openai_or_hardware_imports(self):
        process = subprocess.run([sys.executable, "-S", str(ROOT / "main.py"), "--mock"],
                                 cwd="/tmp", capture_output=True, text=True, timeout=10)
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
        process = subprocess.run([sys.executable, "-S", str(ROOT / "main.py"), "--box-size", ".3"],
                                 cwd="/tmp", capture_output=True, text=True, timeout=10)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("provider", process.stderr)
        self.assertNotIn("homing", process.stdout)

    def test_mock_llm_requires_explicit_network_opt_in(self):
        process = subprocess.run([sys.executable, "-S", str(ROOT / "main.py"), "--mock", "--planner", "llm"],
                                 capture_output=True, text=True, timeout=10)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("allow-api-with-mock", process.stderr)

    def test_help_and_incomplete_build(self):
        help_result = subprocess.run([sys.executable, "-S", str(ROOT / "main.py"), "--help"],
                                     capture_output=True, text=True, timeout=10)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("--provider", help_result.stdout)
        failed = subprocess.run([sys.executable, "-S", str(ROOT / "main.py"), "--mock", "--max-steps", "1"],
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(failed.returncode, 0)
        self.assertNotIn('"status": "COMPLETED"', failed.stdout)


if __name__ == "__main__":
    unittest.main()
