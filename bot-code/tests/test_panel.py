import asyncio
import importlib.util
import json
import re
import sys
import threading
import unittest
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panel import PanelSession, StructureReceiver, parse_design


def payload():
    return {"origin": [0, 1, -1], "size": [1, 2, 3], "count": 4,
            "palette": ["minecraft:oak_planks"],
            "blocks": [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 2, 0]]}


class FirstChoice:
    def decide(self, context, choices):
        return choices[0]


class PanelTests(unittest.TestCase):
    def session(self, reasoner=FirstChoice):
        session = PanelSession(reasoner_factory=reasoner, tool_delay=0)
        self.addCleanup(session.close)
        return session

    def test_wire_parsing_and_unsupported_preview(self):
        design = parse_design(payload())
        self.assertEqual(len(design.structure.blocks), 4)
        self.assertTrue(design.buildable)
        unsupported = payload()
        unsupported["count"] = 1
        unsupported["blocks"] = [[0, 1, 0, 0]]
        design = parse_design(unsupported)
        self.assertFalse(design.buildable)
        self.assertIn("unsupported", design.error)

    def test_malformed_payloads_are_rejected(self):
        invalid = []
        for changes in ({"count": 3}, {"count": True}, {"size": [0, 2, 3]},
                        {"palette": []}, {"blocks": [[0, 0, 0, 4]]},
                        {"blocks": [[True, 0, 0, 0]]}, {"origin": [0, 0]}):
            invalid.append(dict(payload(), **changes))
        duplicate = payload()
        duplicate["blocks"][1] = duplicate["blocks"][0]
        invalid.append(duplicate)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_design(value)

    def test_only_latest_received_design_is_retained(self):
        session = self.session()
        old, new = session.next_sequence(), session.next_sequence()
        session.receive(payload(), sequence=new)
        self.assertFalse(session.receive(payload(), sequence=old))
        self.assertEqual(session.snapshot()["design"]["id"], str(new))

    def test_waits_for_design_and_key(self):
        session = PanelSession(tool_delay=0)
        self.addCleanup(session.close)
        self.assertIsNone(session.snapshot()["design"])
        self.assertEqual(session.snapshot()["view"], "main")
        with self.assertRaises(ValueError):
            session.start("missing")
        session.receive(payload())
        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            session.start(session.snapshot()["design"]["id"])

    def test_start_freezes_design_and_completes_with_tool_events(self):
        session = self.session()
        session.receive(payload())
        initial = session.snapshot()["design"]
        session.start(initial["id"])
        self.assertTrue(session.wait(5))
        state = session.snapshot()
        self.assertEqual(state["view"], "complete", state)
        self.assertEqual(state["job"]["status"], "completed")
        self.assertEqual(len(state["job"]["placed"]), 4)
        self.assertGreater(state["job"]["llm_calls"], 0)
        types = {event["type"] for event in state["job"]["events"]}
        self.assertTrue({"llm_start", "llm_result", "tool_start", "tool_result"}.issubset(types))
        session.back()
        self.assertEqual(session.snapshot()["view"], "main")

    def test_cancel_during_llm_returns_main_and_ignores_late_completion(self):
        started, release = threading.Event(), threading.Event()
        class Blocking:
            calls = 0
            def decide(self, context, choices):
                self.calls += 1
                started.set()
                release.wait(3)
                return choices[0]
        model = Blocking()
        session = self.session(lambda: model)
        self.addCleanup(release.set)
        session.receive(payload())
        original_id = session.snapshot()["design"]["id"]
        session.start(original_id)
        self.assertTrue(started.wait(2))
        session.receive(payload())
        self.assertNotEqual(original_id, session.snapshot()["design"]["id"])
        self.assertEqual(original_id, session.snapshot()["job"]["design"]["id"])
        with self.assertRaisesRegex(ValueError, "busy"):
            session.start(session.snapshot()["design"]["id"])
        session.cancel()
        self.assertEqual(session.snapshot()["view"], "main")
        self.assertTrue(session.snapshot()["worker_busy"])
        release.set()
        self.assertTrue(session.wait(3))
        state = session.snapshot()
        self.assertEqual(state["view"], "main")
        self.assertEqual(state["job"]["status"], "cancelled")
        self.assertFalse(state["worker_busy"])
        self.assertEqual(model.calls, 1)
        self.assertEqual(state["job"]["tool_calls"], 0)

    def test_stale_start_id_and_api_error_do_not_report_completion(self):
        class Failed:
            def decide(self, context, choices):
                raise RuntimeError("sensitive provider error text")
        session = self.session(Failed)
        session.receive(payload())
        old = session.snapshot()["design"]["id"]
        session.receive(payload())
        with self.assertRaisesRegex(ValueError, "changed"):
            session.start(old)
        session.start(session.snapshot()["design"]["id"])
        self.assertTrue(session.wait(3))
        state = session.snapshot()
        self.assertEqual(state["job"]["status"], "failed")
        self.assertNotEqual(state["view"], "complete")
        self.assertNotIn("sensitive provider error text", json.dumps(state))

    def test_key_is_session_only_and_never_in_public_state(self):
        session = PanelSession()
        self.addCleanup(session.close)
        key = "sk-test-placeholder-not-a-real-key"
        session.set_api_key(key)
        self.assertTrue(session.snapshot()["llm_ready"])
        self.assertNotIn(key, json.dumps(session.snapshot()))
        self.assertEqual(session.reasoner_factory().api_key, key)
        self.assertIsNone(session.reasoner_factory().client)
        self.assertFalse(PanelSession().snapshot()["llm_ready"])
        session.close()
        self.assertFalse(session.snapshot()["llm_ready"])

    def test_invalid_key_and_mid_build_key_change_are_rejected(self):
        session = PanelSession()
        self.addCleanup(session.close)
        for key in (None, "short", "x"*513, "sk-not a valid key here"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                session.set_api_key(key)
        session.worker_busy = True
        with self.assertRaises(ValueError):
            session.set_api_key("sk-test-placeholder-not-a-real-key")
        self.assertFalse(session.snapshot()["llm_ready"])
        session.worker_busy = False

    def test_returned_state_cannot_mutate_job_or_design(self):
        session = self.session()
        session.receive(payload())
        state = session.snapshot()
        state["design"]["blocks"].clear()
        self.assertEqual(len(session.snapshot()["design"]["blocks"]), 4)


class ReceiverTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.session = PanelSession()
        self.receiver = StructureReceiver(self.session, host="127.0.0.1", port=0, timeout=.2)
        await self.receiver.start()

    async def asyncTearDown(self):
        await self.receiver.close()
        self.session.close()

    async def send(self, data, split=False):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.receiver.port)
        raw = json.dumps(data).encode() if not isinstance(data, bytes) else data
        if split:
            writer.write(raw[:12])
            await writer.drain()
            await asyncio.sleep(.01)
            self.assertIsNone(self.session.snapshot()["design"])
            writer.write(raw[12:])
        else:
            writer.write(raw)
        await writer.drain()
        writer.write_eof()
        await reader.read()
        writer.close()
        await writer.wait_closed()

    async def test_fragmented_tcp_message_waits_for_eof(self):
        await self.send(payload(), split=True)
        self.assertEqual(self.session.snapshot()["design"]["count"], 4)

    async def test_invalid_message_preserves_latest_valid_design(self):
        await self.send(payload())
        latest = self.session.snapshot()["design"]["id"]
        await self.send(b"not-json")
        self.assertEqual(self.session.snapshot()["design"]["id"], latest)
        self.assertTrue(self.session.snapshot()["notice"])

    async def test_slow_old_connection_does_not_replace_newer_design(self):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.receiver.port)
        writer.write(json.dumps(payload()).encode())
        await writer.drain()
        await self.send(payload())
        latest = self.session.snapshot()["design"]["id"]
        writer.write_eof()
        await reader.read()
        writer.close()
        await writer.wait_closed()
        self.assertEqual(self.session.snapshot()["design"]["id"], latest)


class PanelAssetsTests(unittest.TestCase):
    def test_script_dom_references_exist(self):
        class Elements(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids = []
            def handle_starttag(self, tag, attrs):
                values = dict(attrs)
                if "id" in values:
                    self.ids.append(values["id"])
        root = Path(__file__).resolve().parents[1]
        parser = Elements()
        parser.feed((root / "panel.html").read_text())
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        script = (root / "panel.js").read_text()
        referenced = set(re.findall(r"\b(?:byId|text)\('([^']+)'", script))
        self.assertFalse(referenced-set(parser.ids))
        self.assertNotIn("innerHTML", script)
        self.assertNotIn("https://", script)

    @unittest.skipUnless(importlib.util.find_spec("quickjs"), "optional JavaScript engine not installed")
    def test_javascript_syntax(self):
        import quickjs
        source = (Path(__file__).resolve().parents[1] / "panel.js").read_text()
        quickjs.Context().eval("new Function("+json.dumps(source)+")")

    @unittest.skipUnless(importlib.util.find_spec("quickjs"), "optional JavaScript engine not installed")
    def test_3d_preview_draws_visible_top_and_rotates(self):
        import quickjs
        source = (Path(__file__).resolve().parents[1] / "panel.js").read_text().split("let state = null")[0]
        context = quickjs.Context()
        context.eval("""
            const window = {devicePixelRatio: 1};
            class ResizeObserver { observe() {} }
            function requestAnimationFrame(fn) { fn(); }
            const ctx = new Proxy({}, {get: (target, key) => target[key] || (() => {})});
            const canvas = {getContext: () => ctx, parentElement: {}, addEventListener: () => {},
                            getBoundingClientRect: () => ({width: 600, height: 400})};
        """)
        context.eval(source)
        context.eval("const view = new VoxelView(canvas); view.set({id: 'one', size: [1,1,1], blocks: [{x:0,y:0,z:0}]}, 'design', [], null);")
        faces = json.loads(context.eval("JSON.stringify(view.faces.map(f => f.face))"))
        self.assertEqual(len(faces), 3)
        self.assertIn(3, faces)
        self.assertNotIn(2, faces)
        before = context.eval("JSON.stringify(view.project([0, 0, 0]))")
        context.eval("view.yaw += .4; view.draw();")
        self.assertNotEqual(before, context.eval("JSON.stringify(view.project([0, 0, 0]))"))


@unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"),
                     "web dependencies available through the uv app environment")
class PanelHTTPTests(unittest.TestCase):
    def test_key_entry_requires_csrf_and_is_never_echoed(self):
        from fastapi.testclient import TestClient
        from panel import create_app
        session = PanelSession()
        key = "sk-test-placeholder-not-a-real-key"
        with TestClient(create_app(session, receiver_host="127.0.0.1", receiver_port=0),
                        base_url="https://testserver") as client:
            self.assertEqual(client.post("/api/key", json={"api_key": key}).status_code, 403)
            token = client.get("/api/state").json()["csrf"]
            headers = {"X-Crafter-Token": token}
            response = client.post("/api/key", json={"api_key": key}, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertNotIn(key, response.text)
            state = client.get("/api/state")
            self.assertTrue(state.json()["llm_ready"])
            self.assertNotIn(key, state.text)
            self.assertIsNone(session.reasoner_factory().client)
            self.assertEqual(client.post("/api/key", content="x"*5000, headers=headers).status_code, 413)

    def test_remote_unencrypted_key_entry_is_rejected(self):
        from fastapi.testclient import TestClient
        from panel import create_app
        session = PanelSession()
        with TestClient(create_app(session, receiver_host="127.0.0.1", receiver_port=0)) as client:
            headers = {"X-Crafter-Token": client.get("/api/state").json()["csrf"]}
            response = client.post("/api/key", json={"api_key": "sk-test-placeholder-not-a-real-key"}, headers=headers)
            self.assertEqual(response.status_code, 403)
            self.assertFalse(session.snapshot()["llm_ready"])

    def test_http_assets_state_protection_and_complete_flow(self):
        from fastapi.testclient import TestClient
        from panel import create_app
        session = PanelSession(FirstChoice, tool_delay=0)
        with TestClient(create_app(session, receiver_host="127.0.0.1", receiver_port=0)) as client:
            for path in ("/", "/panel.js", "/panel.css"):
                response = client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("Content-Security-Policy", response.headers)
            state = client.get("/api/state").json()
            self.assertIsNone(state["design"])
            self.assertEqual(client.post("/api/example").status_code, 403)
            headers = {"X-Crafter-Token": state["csrf"]}
            self.assertEqual(client.post("/api/example", headers=headers).status_code, 200)
            design_id = client.get("/api/state").json()["design"]["id"]
            started = client.post("/api/builds", json={"design_id": design_id}, headers=headers)
            self.assertEqual(started.status_code, 200)
            self.assertTrue(session.wait(5))
            self.assertEqual(client.get("/api/state").json()["view"], "complete")
            self.assertEqual(client.post("/api/main", headers=headers).status_code, 200)
            self.assertEqual(client.get("/api/state").json()["view"], "main")
            self.assertEqual(client.post("/api/cancel", json={"job_id": "old-job"}, headers=headers).status_code, 409)


if __name__ == "__main__":
    unittest.main()
