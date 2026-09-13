import json
import os
import stat
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_backend import OpenAIReasoner, STEP_SCHEMA, load_api_key, parse_step, save_api_key_from_env
from agent_types import Step


class BackendTests(unittest.TestCase):
    def client(self, content, finish_reason="stop", refusal=None):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(finish_reason=finish_reason,
                            message=SimpleNamespace(content=content, refusal=refusal))])
        return client

    def test_strict_schema_and_bounded_request(self):
        step = Step("pickup", "Use the approached material", box_id=7)
        client = self.client(json.dumps(asdict(step)))
        backend = OpenAIReasoner(client=client)
        self.assertEqual(backend.decide({"goal": {}}, (step,)), step)
        request = client.chat.completions.create.call_args.kwargs
        schema = request["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertEqual(schema["schema"], STEP_SCHEMA)
        self.assertLessEqual(request["max_tokens"], 512)
        self.assertEqual(request["timeout"], 15.0)

    def test_json_only_is_explicit_and_still_validated(self):
        client = self.client(json.dumps(asdict(Step("done"))))
        backend = OpenAIReasoner(client=client, json_only=True)
        backend.decide({}, (Step("done"),))
        self.assertEqual(client.chat.completions.create.call_args.kwargs["response_format"],
                         {"type": "json_object"})

    def test_invalid_shapes_types_and_operation_fields(self):
        base = asdict(Step("pickup", box_id=2))
        invalid = [dict(base, box_id=True), dict(base, box_id=2.0),
                   dict(base, joint=1), dict(base, site_id="invented"),
                   dict(base, operation="drive"), dict(base, reason="x"*513),
                   dict(base, cell=[0, 0, 0]), {"operation": "done"}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_step(json.dumps(value))
        for text in ('[]', '{"operation":"done","operation":"stop"}',
                     '```json\n{}\n```', '{"x": NaN}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_step(text)

    def test_response_refusal_truncation_empty_and_invented_ids(self):
        for client in (self.client(None), self.client("{}", refusal="refused"),
                       self.client("{}", finish_reason="length"),
                       self.client(json.dumps(asdict(Step("pickup", box_id=999))))):
            with self.subTest(client=client), self.assertRaises(ValueError):
                OpenAIReasoner(client=client).decide({}, (Step("pickup", box_id=2),))

    def test_transport_failure_propagates_to_agent_boundary(self):
        client = self.client("{}")
        client.chat.completions.create.side_effect = TimeoutError("offline")
        with self.assertRaises(TimeoutError):
            OpenAIReasoner(client=client).decide({}, (Step("done"),))

    def test_prompt_and_reply_size_limits(self):
        client = self.client(" "*20000)
        with self.assertRaises(ValueError):
            OpenAIReasoner(client=client).decide({}, (Step("done"),))
        client.reset_mock()
        with self.assertRaises(ValueError):
            OpenAIReasoner(client=client).decide({"huge": "x"*100000}, (Step("done"),))
        client.chat.completions.create.assert_not_called()


class SavedKeyTests(unittest.TestCase):
    def test_saved_key_is_private_and_loaded_without_environment(self):
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {"HOME": home, "OPENAI_API_KEY": "sk-test-persistent-placeholder"}):
            path = save_api_key_from_env()
            self.assertEqual(path, Path(home) / ".config" / "crafter" / "openai_api_key")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
            os.environ["OPENAI_API_KEY"] = ""
            self.assertEqual(load_api_key(), "sk-test-persistent-placeholder")
            os.environ["OPENAI_API_KEY"] = "sk-test-environment-override"
            self.assertEqual(load_api_key(), "sk-test-environment-override")
            with self.assertRaises(FileExistsError):
                save_api_key_from_env()
            os.environ["OPENAI_API_KEY"] = ""
            self.assertEqual(load_api_key(), "sk-test-persistent-placeholder")

    def test_missing_key_is_not_saved_as_an_empty_file(self):
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {"HOME": home, "OPENAI_API_KEY": ""}):
            self.assertIsNone(load_api_key())
            with self.assertRaises(ValueError):
                save_api_key_from_env()
            self.assertFalse((Path(home) / ".config" / "crafter" / "openai_api_key").exists())

    def test_insecure_or_symlinked_file_is_not_loaded(self):
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {"HOME": home, "OPENAI_API_KEY": "sk-test-private-placeholder"}):
            path = save_api_key_from_env()
            os.environ["OPENAI_API_KEY"] = ""
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                load_api_key()
            path.unlink()
            target = Path(home) / "unrelated"
            target.write_text("sk-test-unrelated-placeholder")
            target.chmod(0o600)
            path.symlink_to(target)
            with self.assertRaises(OSError):
                load_api_key()


if __name__ == "__main__":
    unittest.main()
