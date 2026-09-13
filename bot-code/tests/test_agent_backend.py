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
from agent_types import MAX_IMAGE_BYTES, Step
from mock_agent_world import MockAgentWorld


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

    def test_mock_scene_is_sent_as_image_content_not_base64_prompt_text(self):
        image = asdict(MockAgentWorld(1).observe().images[0])
        client = self.client(json.dumps(asdict(Step("observe"))))
        backend = OpenAIReasoner(client=client)
        backend.decide({"images": [image]}, (Step("observe"),))
        content = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertEqual([part["type"] for part in content], ["text", "image_url"])
        self.assertEqual(content[1]["image_url"]["url"], image["data_url"])
        metadata = json.loads(content[0]["text"])["images"][0]
        self.assertTrue(metadata["simulated"])
        self.assertEqual(metadata["view"], "overview")
        self.assertNotIn("data_url", metadata)
        self.assertNotIn(image["data_url"], content[0]["text"])

    def test_remote_invalid_or_excessive_images_never_reach_client(self):
        image = asdict(MockAgentWorld(1).observe().images[0])
        invalid = [[dict(image, data_url="https://example.invalid/frame.png")],
                   [dict(image, data_url="file:///tmp/frame.png")],
                   [dict(image, data_url="data:image/png;base64,not-base64")],
                   [dict(image, data_url="data:image/png;base64," + "A"*(MAX_IMAGE_BYTES*2))],
                   [image]*4]
        for images in invalid:
            with self.subTest(images=len(images)):
                client = self.client(json.dumps(asdict(Step("observe"))))
                with self.assertRaises(ValueError):
                    OpenAIReasoner(client=client).decide({"images": images}, (Step("observe"),))
                client.chat.completions.create.assert_not_called()

    def test_images_do_not_authorize_an_ineligible_decision(self):
        image = asdict(MockAgentWorld(1).observe().images[0])
        client = self.client(json.dumps(asdict(Step("pickup", box_id=999))))
        with self.assertRaisesRegex(ValueError, "eligible"):
            OpenAIReasoner(client=client).decide({"images": [image]}, (Step("observe"),))

    def test_fake_multimodal_client_completes_a_mock_build(self):
        from agent import Agent
        from contracts import Block, Structure
        client = self.client("{}")
        def respond(**kwargs):
            content = kwargs["messages"][1]["content"]
            self.assertTrue(any(part["type"] == "image_url" for part in content))
            context = json.loads(content[0]["text"])
            decision = context["allowed_choices"][0]
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content=json.dumps(decision), refusal=None))])
        client.chat.completions.create.side_effect = respond
        world = MockAgentWorld(3)
        result = Agent(world.actions, world.observations, reasoner=OpenAIReasoner(client=client),
                       backend="llm", clock=world.clock, sleep=world.sleep).run(
                           Structure([Block(0, 0, 0), Block(1, 0, 0), Block(0, 1, 0)]))
        self.assertTrue(result.success, result)
        self.assertEqual(result.placed, 3)
        self.assertTrue(world.monitor_calls)


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
