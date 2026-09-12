import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_backend import OpenAIReasoner, STEP_SCHEMA, parse_step
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


if __name__ == "__main__":
    unittest.main()
