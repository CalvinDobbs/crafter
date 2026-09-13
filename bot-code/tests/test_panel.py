import asyncio
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import unittest
from functools import partial
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
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

    def test_clear_removes_design_and_preserves_configuration(self):
        session = self.session()
        session.receive(payload())
        session.receiver_notice("Previous scan rejected")
        before = session.snapshot()
        session.clear_design(before["design"]["id"])
        state = session.snapshot()
        self.assertIsNone(state["design"])
        self.assertIsNone(state["job"])
        self.assertEqual(state["notice"], "")
        self.assertEqual(state["view"], "main")
        self.assertTrue(state["llm_ready"])
        self.assertEqual(state["receiver"], before["receiver"])
        self.assertGreater(state["revision"], before["revision"])
        with self.assertRaisesRegex(ValueError, "wait for a Minecraft design"):
            session.start(before["design"]["id"])
        self.assertTrue(session.receive(payload()))
        self.assertNotEqual(session.snapshot()["design"]["id"], before["design"]["id"])

    def test_stale_clear_cannot_discard_a_newer_design(self):
        session = self.session()
        session.receive(payload())
        old_id = session.snapshot()["design"]["id"]
        session.receive(payload())
        before = session.snapshot()
        for design_id in (old_id, None):
            with self.subTest(design_id=design_id), self.assertRaisesRegex(ValueError, "design changed"):
                session.clear_design(design_id)
        self.assertEqual(session.snapshot(), before)

    def test_clear_preserves_active_build_and_its_completion(self):
        started, release = threading.Event(), threading.Event()
        class Blocking:
            def decide(self, context, choices):
                started.set()
                release.wait(3)
                return choices[0]
        session = self.session(Blocking)
        self.addCleanup(release.set)
        session.receive(payload())
        design_id = session.snapshot()["design"]["id"]
        session.start(design_id)
        self.assertTrue(started.wait(2))
        before = session.snapshot()
        session.clear_design(design_id)
        state = session.snapshot()
        self.assertIsNone(state["design"])
        self.assertEqual(state["job"], before["job"])
        self.assertEqual(state["view"], "build")
        self.assertTrue(state["worker_busy"])
        self.assertFalse(session.cancel_event.is_set())
        release.set()
        self.assertTrue(session.wait(5))
        state = session.snapshot()
        self.assertEqual(state["job"]["status"], "completed")
        self.assertEqual(len(state["job"]["placed"]), 4)
        self.assertIsNone(state["design"])

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

    async def test_clear_ignores_inflight_scan_and_accepts_next_scan(self):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.receiver.port)
        try:
            writer.write(json.dumps(payload()).encode())
            await writer.drain()
            await self.send(payload())
            self.session.clear_design(self.session.snapshot()["design"]["id"])
            writer.write_eof()
            await reader.read()
            self.assertIsNone(self.session.snapshot()["design"])
        finally:
            writer.close()
            await writer.wait_closed()
        await self.send(payload())
        self.assertEqual(self.session.snapshot()["design"]["count"], 4)


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

    def test_clear_button_is_on_main_screen(self):
        html = (Path(__file__).resolve().parents[1] / "panel.html").read_text()
        main = html.split('<section id="main-screen"', 1)[1].split("</section>", 1)[0]
        self.assertRegex(main, r'<button\b[^>]*\bid="clear-blueprint"[^>]*\bdisabled')
        self.assertIn("Clear blueprint", main)

    @unittest.skipUnless(os.environ.get("CRAFTER_LAYOUT_CDP") and shutil.which("node"),
                         "optional layout check requires Node 22+ and an isolated Chromium CDP endpoint")
    def test_screens_fit_the_browser_viewport(self):
        class Assets(SimpleHTTPRequestHandler):
            def log_message(self, *_):
                pass
        root = Path(__file__).resolve().parents[1]
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Assets, directory=str(root)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        script = r"""
            const cdp = process.env.CRAFTER_LAYOUT_CDP;
            const target = await fetch(cdp+'/json/new?about:blank', {method:'PUT'}).then(r=>r.json());
            const ws = new WebSocket(target.webSocketDebuggerUrl);
            await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
            let sequence=0;
            const pending=new Map();
            ws.addEventListener('message', event=>{
                const message=JSON.parse(event.data), call=pending.get(message.id);
                if(!call)return;
                pending.delete(message.id);clearTimeout(call.timer);
                message.error?call.reject(new Error(JSON.stringify(message.error))):call.resolve(message.result);
            });
            const send=(method,params={})=>new Promise((resolve,reject)=>{
                const id=++sequence,timer=setTimeout(()=>{pending.delete(id);reject(new Error('CDP timeout: '+method));},5000);
                pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));
            });
            const evaluate=async expression=>{
                const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
                if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));
                return r.result.value;
            };
            const now=Date.now()/1000;
            const design={id:'1',received_at:now,count:4,blocks:[{x:0,y:0,z:0},{x:1,y:0,z:0},{x:0,y:0,z:1},{x:0,y:1,z:0}],size:[2,2,2],source:'Example',buildable:true};
            const base={revision:1,view:'main',design,job:null,notice:'',receiver:{listening:true,port:5005},worker_busy:false,llm_ready:true,model:'offline-layout-test',csrf:'layout-only',simulation:true};
            const job={id:'layout-job',design,status:'running',phase:'BUILD',reasoning:'Inspect the supported target before placing the next box. '.repeat(9),placed:[],started_at:now,finished_at:null,llm_calls:20,tool_calls:30,current_tool:'place',current_cell:[0,0,0],error:null,events:Array.from({length:40},(_,i)=>({id:i+1,type:'placement_confirmed',at:now+i,data:{box_id:i,cell:[0,0,0]}}))};
            const failures=[];
            const viewports=[[1600,900],[1280,720],[1280,600],[1024,640],[1024,576],[800,600],[640,480],[390,844],[320,568],[844,390]];
            const modes=['waiting','main','configuration','notice','unsupported','build','failure','complete'];
            try {
                await send('Page.enable');
                await send('Network.enable');
                await send('Network.setCacheDisabled',{cacheDisabled:true});
                await send('Page.addScriptToEvaluateOnNewDocument',{source:`window.__layoutState=${JSON.stringify(base)};window.fetch=async()=>new Response(JSON.stringify(window.__layoutState),{headers:{'Content-Type':'application/json'}});`});
                await send('Page.navigate',{url:process.env.CRAFTER_LAYOUT_URL+'/panel.html'});
                for(let i=0;i<60;i++){
                    if(await evaluate("document.readyState==='complete' && typeof render==='function'"))break;
                    await new Promise(r=>setTimeout(r,25));
                }
                for(const [width,height] of viewports) {
                    await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
                    for(const mode of modes) {
                        const screen=['waiting','configuration','notice','unsupported'].includes(mode)?'main':mode==='failure'?'build':mode;
                        const s={...base,revision:++sequence,view:screen,llm_ready:mode!=='configuration',
                            design:mode==='waiting'?null:mode==='unsupported'?{...design,buildable:false,error:'Target contains an unsupported block at (0, 1, 0).'}:design,
                            notice:mode==='notice'?'Minecraft design rejected: invalid, incomplete, or timed-out message. Please scan the structure again.':'',
                            job:screen==='main'?null:{...job,status:mode==='complete'?'completed':mode==='failure'?'failed':'running',error:mode==='failure'?'Verification unavailable. '.repeat(20):null,finished_at:mode==='complete'?now+20:null}};
                        await evaluate(`window.__layoutState=${JSON.stringify(s)};state=window.__layoutState;render(state);new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))`);
                        const result=await evaluate(`(()=>{
                            const problems=[],root=document.documentElement;
                            if(root.scrollHeight>innerHeight+1||root.scrollWidth>innerWidth+1)problems.push('document overflows: '+root.scrollWidth+'x'+root.scrollHeight);
                            const controls=${JSON.stringify(screen==='main'?['#start-build','#clear-blueprint','#load-example','#main-canvas']:screen==='build'?[mode==='failure'?'#failed-back':'#cancel-build','#build-canvas','#activity-feed']:['#back-main','#complete-canvas'])};
                            for(const selector of controls){
                                const element=document.querySelector(selector),r=element.getBoundingClientRect();
                                if(!r.width||!r.height||r.top<-.5||r.left<-.5||r.bottom>innerHeight+.5||r.right>innerWidth+.5)problems.push(selector+' outside viewport');
                                for(let parent=element.parentElement;parent;parent=parent.parentElement){
                                    const style=getComputedStyle(parent),p=parent.getBoundingClientRect();
                                    if(/auto|scroll|hidden|clip/.test(style.overflowY)&&(r.top<p.top-1||r.bottom>p.bottom+1))problems.push(selector+' clipped by '+parent.className);
                                }
                            }
                            const canvas=document.querySelector('[data-screen]:not([hidden]) canvas'),r=canvas.getBoundingClientRect();
                            if(r.width<1||r.height<1)problems.push('preview collapsed');
                            if(${JSON.stringify(screen)}==='build'){
                                const feed=document.querySelector('#activity-feed'),r=feed.getBoundingClientRect();
                                if(r.bottom>innerHeight+1||r.height<1||getComputedStyle(feed).overflowY!=='auto')problems.push('activity feed is not contained');
                            }
                            return problems;
                        })()`);
                        if(result.length)failures.push({viewport:[width,height],mode,problems:result});
                    }
                }
                console.log(JSON.stringify({cases:viewports.length*modes.length,failures},null,2));
                if(failures.length)process.exitCode=1;
            } finally {
                for(const call of pending.values())clearTimeout(call.timer);
                ws.close();
                await fetch(cdp+'/json/close/'+target.id);
            }
        """
        try:
            result = subprocess.run([shutil.which("node"), "--input-type=module", "-"], input=script,
                                    capture_output=True, text=True, timeout=60,
                                    env=dict(os.environ, CRAFTER_LAYOUT_URL=f"http://127.0.0.1:{server.server_port}"))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

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
        context.eval("view.selected = '0,0,0'; view.set(null, 'design', [], null);")
        self.assertEqual(context.eval("view.blocks.length"), 0)
        self.assertEqual(context.eval("view.faces.length"), 0)
        self.assertIsNone(context.eval("view.selected"))


@unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"),
                     "web dependencies available through the uv app environment")
class PanelHTTPTests(unittest.TestCase):
    def test_clear_requires_csrf_and_current_design(self):
        from fastapi.testclient import TestClient
        from panel import create_app
        session = PanelSession(FirstChoice, tool_delay=0)
        with TestClient(create_app(session, receiver_host="127.0.0.1", receiver_port=0)) as client:
            headers = {"X-Crafter-Token": client.get("/api/state").json()["csrf"]}
            self.assertEqual(client.post("/api/example", headers=headers).status_code, 200)
            before = client.get("/api/state").json()
            body = {"design_id": before["design"]["id"]}
            self.assertEqual(client.post("/api/clear", json=body).status_code, 403)
            self.assertEqual(client.post("/api/clear", json={"design_id": "stale"}, headers=headers).status_code, 409)
            self.assertEqual(client.get("/api/state").json()["design"], before["design"])
            response = client.post("/api/clear", json=body, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"ok": True})
            state = client.get("/api/state").json()
            self.assertIsNone(state["design"])
            self.assertEqual(state["view"], "main")
            self.assertTrue(state["llm_ready"])
            self.assertEqual(client.post("/api/builds", json=body, headers=headers).status_code, 409)
            self.assertEqual(client.post("/api/example", headers=headers).status_code, 200)
            self.assertNotEqual(client.get("/api/state").json()["design"]["id"], body["design_id"])

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
