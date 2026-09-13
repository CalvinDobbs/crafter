import asyncio
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import unittest
from functools import partial
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panel import PANEL_DIR, PanelSession, StructureReceiver, parse_design


def payload():
    return {"origin": [0, 1, -1], "size": [1, 2, 3], "count": 4,
            "palette": ["minecraft:oak_planks"],
            "blocks": [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 2, 0]]}


def debug_console(box_count=4, action_duration=.05):
    """A DebugConsole over MockAgentWorld. The console itself ships no simulator; tests supply one."""
    from debug_console import DebugConsole
    from test_debug_console import real_time_world
    voxel = (.3, .3, .3)
    world = real_time_world(box_count, voxel, action_duration)
    return DebugConsole(lambda: (world.observations, None),
                        lambda observations: (world.actions, None),
                        kind="test double", voxel_size=voxel)


class FirstChoice:
    def decide(self, context, choices):
        return choices[0]


class PanelTests(unittest.TestCase):
    def session(self, reasoner=FirstChoice):
        session = PanelSession(reasoner_factory=reasoner, pace=0)
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

    def test_a_build_waits_for_a_design_but_not_for_a_key(self):
        """Builds take the agent's own first legal step, so no model and no key are involved."""
        session = PanelSession(pace=0)
        self.addCleanup(session.close)
        self.assertIsNone(session.snapshot()["design"])
        self.assertEqual(session.snapshot()["view"], "main")
        self.assertIsNone(session.reasoner_factory)
        with self.assertRaises(ValueError):
            session.start("missing")
        session.receive(payload())
        session.start(session.snapshot()["design"]["id"])
        self.assertTrue(session.wait(5))
        state = session.snapshot()
        self.assertEqual(state["job"]["status"], "completed")
        self.assertEqual(len(state["job"]["placed"]), 4)

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
        session.close()
        self.assertIsNone(session.reasoner_factory)

    def test_invalid_key_and_mid_build_key_change_are_rejected(self):
        session = PanelSession()
        self.addCleanup(session.close)
        for key in (None, "short", "x"*513, "sk-not a valid key here"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                session.set_api_key(key)
        session.worker_busy = True
        with self.assertRaises(ValueError):
            session.set_api_key("sk-test-placeholder-not-a-real-key")
        self.assertIsNone(session.reasoner_factory)
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
    def test_light_theme_has_red_accent_and_readable_contrast(self):
        html = (PANEL_DIR / "panel.html").read_text()
        css = (PANEL_DIR / "panel.css").read_text()
        self.assertIn('<meta name="color-scheme" content="light">', html)
        self.assertIn("color-scheme:light", css)
        colors = dict(re.findall(r"(--[\w-]+):#([0-9a-f]{6})\b", css))
        for name in ("--bg", "--surface", "--raised"):
            self.assertGreaterEqual(min(int(colors[name][i:i+2], 16) for i in (0, 2, 4)), 230)
        red, green, blue = (int(colors["--accent"][i:i+2], 16) for i in (0, 2, 4))
        self.assertGreater(red, max(green, blue)+50)
        for foreground, background in ((colors["--text"], colors["--bg"]), ("ffffff", colors["--accent"])):
            luminance = []
            for color in (foreground, background):
                channels = [int(color[i:i+2], 16)/255 for i in (0, 2, 4)]
                linear = [v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in channels]
                luminance.append(sum(v*weight for v, weight in zip(linear, (.2126, .7152, .0722))))
            self.assertGreaterEqual((max(luminance)+.05)/(min(luminance)+.05), 4.5)

    def test_brand_uses_packaged_svg(self):
        self.assertEqual(PANEL_DIR.name, "web")
        html = (PANEL_DIR / "panel.html").read_text()
        self.assertRegex(html, r'<img\b[^>]*class="brand-logo"[^>]*src="/assets/Crafter-transparent\.svg"[^>]*alt="Crafter"')
        self.assertTrue((PANEL_DIR / "assets" / "Crafter-transparent.svg").is_file())
        self.assertNotIn('class="brand-mark"', html)
        self.assertNotIn('class="brand-sub"', html)
        for asset in ("panel.html", "panel.css", "panel.js", "Crafter-transparent.svg"):
            with self.subTest(asset=asset):
                self.assertFalse((PANEL_DIR.parent / asset).exists())

    def test_script_dom_references_exist(self):
        class Elements(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids = []
            def handle_starttag(self, tag, attrs):
                values = dict(attrs)
                if "id" in values:
                    self.ids.append(values["id"])
        root = PANEL_DIR
        parser = Elements()
        parser.feed((root / "panel.html").read_text())
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        for name in ("panel.js", "debug.js"):
            with self.subTest(script=name):
                script = (root / name).read_text()
                referenced = set(re.findall(r"\b(?:byId|text)\('([^']+)'", script))
                self.assertTrue(referenced)
                self.assertFalse(referenced-set(parser.ids))
                self.assertNotIn("innerHTML", script)
                self.assertNotIn("https://", script)

    def test_clear_button_is_on_main_screen(self):
        html = (PANEL_DIR / "panel.html").read_text()
        main = html.split('<section id="main-screen"', 1)[1].split("</section>", 1)[0]
        self.assertRegex(main, r'<button\b[^>]*\bid="clear-blueprint"[^>]*\bdisabled')
        self.assertIn("Clear blueprint", main)

    def test_main_screen_omits_design_and_runtime_cards(self):
        html = (PANEL_DIR / "panel.html").read_text()
        main = html.split('<section id="main-screen"', 1)[1].split("</section>", 1)[0]
        self.assertNotIn("<h2>Latest design</h2>", main)
        self.assertNotIn("<h2>The demo setup</h2>", main)
        self.assertNotRegex(main, r'id="(?:design-source|box-count|layer-count|design-size|received-at|design-id|model-name|model-status)"')
        for element in ("main-canvas", "start-build", "clear-blueprint", "configuration-warning", "design-warning"):
            with self.subTest(element=element):
                self.assertIn(f'id="{element}"', main)

    @unittest.skipUnless(os.environ.get("CRAFTER_LAYOUT_CDP") and shutil.which("node"),
                         "optional layout check requires Node 22+ and an isolated Chromium CDP endpoint")
    def test_screens_fit_the_browser_viewport(self):
        class Assets(SimpleHTTPRequestHandler):
            def log_message(self, *_):
                pass
        root = PANEL_DIR
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
                            if(getComputedStyle(root).colorScheme!=='light')problems.push('native controls are not in light mode');
                            const background=getComputedStyle(document.body).backgroundColor.match(/[0-9]+/g).slice(0,3).map(Number);
                            if(Math.min(...background)<230)problems.push('page background is not light');
                            if(root.scrollHeight>innerHeight+1||root.scrollWidth>innerWidth+1)problems.push('document overflows: '+root.scrollWidth+'x'+root.scrollHeight);
                            const controls=${JSON.stringify(screen==='main'?['#start-build','#clear-blueprint','#load-example','#main-canvas']:screen==='build'?[mode==='failure'?'#failed-back':'#cancel-build','#build-canvas','#activity-feed']:['#back-main','#complete-canvas'])};
                            controls.push('.brand-logo','.header-status');
                            const logo=document.querySelector('.brand-logo'),brand=logo.getBoundingClientRect(),status=document.querySelector('.header-status').getBoundingClientRect();
                            if(!logo.complete||!logo.naturalWidth)problems.push('brand logo did not load');
                            if(brand.right+8>status.left)problems.push('brand logo overlaps header status');
                            if(${JSON.stringify(mode)}==='configuration')controls.push('#api-key','#save-api-key');
                            if(${JSON.stringify(screen)}==='main'){
                                const layout=document.querySelector('.design-layout').getBoundingClientRect(),preview=document.querySelector('.design-layout>.scene-card').getBoundingClientRect();
                                if(Math.abs(preview.width-layout.width)>2)problems.push('preview does not use the full width');
                            }
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
                            const pixel=canvas.getContext('2d').getImageData(0,0,1,1).data;
                            if(Math.min(pixel[0],pixel[1],pixel[2])<150)problems.push('preview background is not light');
                            if(${JSON.stringify(mode)}==='configuration'){
                                const input=getComputedStyle(document.querySelector('#api-key')).backgroundColor.match(/[0-9]+/g).slice(0,3).map(Number);
                                if(Math.min(...input)<230)problems.push('key input is not light');
                            }
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

    @unittest.skipUnless(os.environ.get("CRAFTER_LAYOUT_CDP") and shutil.which("node"),
                         "optional layout check requires Node 22+ and an isolated Chromium CDP endpoint")
    def test_debug_screen_scrolls_its_panes_inside_the_fixed_shell(self):
        """The shell never scrolls, so the debug panes must; a clipped pane hides tools entirely."""
        console = debug_console(6)
        self.addCleanup(console.close)
        console.observe()
        console.find_sites()
        console.select_site("floor-a")
        console.observe("floor-a")
        console.verify()
        payload = json.dumps(console.state())

        class Assets(SimpleHTTPRequestHandler):
            def log_message(self, *_):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Assets, directory=str(PANEL_DIR)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        script = r"""
            const cdp = process.env.CRAFTER_LAYOUT_CDP;
            const target = await fetch(cdp+'/json/new?about:blank', {method:'PUT'}).then(r=>r.json());
            const ws = new WebSocket(target.webSocketDebuggerUrl);
            await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
            let sequence=0;
            const pending=new Map(), errors=[];
            ws.addEventListener('message', event=>{
                const message=JSON.parse(event.data), call=pending.get(message.id);
                if(message.method==='Runtime.exceptionThrown')errors.push(message.params.exceptionDetails.text);
                if(!call)return;
                pending.delete(message.id);clearTimeout(call.timer);
                message.error?call.reject(new Error(JSON.stringify(message.error))):call.resolve(message.result);
            });
            const send=(method,params={})=>new Promise((resolve,reject)=>{
                const id=++sequence,timer=setTimeout(()=>{pending.delete(id);reject(new Error('CDP timeout: '+method));},8000);
                pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));
            });
            const evaluate=async expression=>{
                const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
                if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));
                return r.result.value;
            };
            const failures=[];
            try {
                await send('Page.enable');
                await send('Runtime.enable');
                await send('Network.setCacheDisabled',{cacheDisabled:true});
                const panel={revision:1,view:'debug',design:null,job:null,notice:'',receiver:{listening:true,port:5005},
                             worker_busy:false,llm_ready:true,model:'offline-layout-test',csrf:'layout-only',simulation:true,debug_kind:'test double'};
                await send('Page.addScriptToEvaluateOnNewDocument',{source:
                    `window.__panel=${JSON.stringify(panel)};window.__console=${process.env.CRAFTER_DEBUG_STATE};`+
                    `window.fetch=async url=>new Response(JSON.stringify(String(url).includes('/api/debug')?window.__console:window.__panel),{headers:{'Content-Type':'application/json'}});`});
                for(const [width,height] of [[1600,900],[1280,720],[1024,640],[430,900]]) {
                    await send('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
                    await send('Page.navigate',{url:process.env.CRAFTER_LAYOUT_URL+'/panel.html'});
                    for(let i=0;i<80;i++){
                        if(await evaluate("document.readyState==='complete' && typeof render==='function'"))break;
                        await new Promise(r=>setTimeout(r,25));
                    }
                    await new Promise(r=>setTimeout(r,1400));
                    const problems=await evaluate(`(()=>{
                        const problems=[],root=document.documentElement;
                        if(document.getElementById('debug-screen').hidden)problems.push('debug screen did not open');
                        if(!document.getElementById('workflow').hidden)problems.push('build workflow still shown');
                        if(root.scrollHeight>innerHeight+1||root.scrollWidth>innerWidth+1)problems.push('shell scrolls: '+root.scrollWidth+'x'+root.scrollHeight);
                        const scrollers=[...document.querySelectorAll('#debug-screen .debug-column, #debug-screen .debug-layout')]
                            .filter(node=>/auto|scroll/.test(getComputedStyle(node).overflowY));
                        if(!scrollers.length)problems.push('no scrolling pane');
                        for(const card of document.querySelectorAll('#debug-screen .panel-card')){
                            if(card.hidden)continue;     // empty cards hide rather than stand there saying nothing
                            const r=card.getBoundingClientRect(),name=(card.querySelector('h2,summary')||card).textContent.slice(0,28);
                            if(!r.width||!r.height)problems.push(name+' collapsed');
                            if(r.right>innerWidth+.5||r.left<-.5)problems.push(name+' outside the viewport width');
                            const pane=scrollers.find(node=>node.contains(card));
                            if(!pane)problems.push(name+' is not inside a scrolling pane');
                        }
                        for(const control of document.querySelectorAll('#debug-screen button:not([hidden]), #debug-screen select')){
                            const r=control.getBoundingClientRect();
                            if(control.offsetParent===null)continue;
                            if(r.right>innerWidth+.5||r.left<-.5)problems.push((control.id||control.textContent.trim()).slice(0,24)+' control outside width');
                        }
                        if(!document.getElementById('debug-banner').hidden)problems.push('an alert is standing on an unarmed console');
                        if(!document.getElementById('debug-monitor-card').hidden)problems.push('the monitor card shows with no monitor configured');
                        if(!document.querySelectorAll('#debug-tools .tool-row').length)problems.push('no tool rows rendered');
                        if(!document.querySelectorAll('#debug-box-rows tr').length)problems.push('no observed boxes rendered');
                        if(!document.querySelectorAll('#debug-log article').length)problems.push('no log entries rendered');
                        return problems;
                    })()`);
                    if(problems.length)failures.push({viewport:[width,height],problems});
                }
                console.log(JSON.stringify({failures,errors},null,2));
                if(failures.length||errors.length)process.exitCode=1;
            } finally {
                for(const call of pending.values())clearTimeout(call.timer);
                ws.close();
                await fetch(cdp+'/json/close/'+target.id);
            }
        """
        try:
            result = subprocess.run([shutil.which("node"), "--input-type=module", "-"], input=script,
                                    capture_output=True, text=True, timeout=90,
                                    env=dict(os.environ, CRAFTER_DEBUG_STATE=payload,
                                             CRAFTER_LAYOUT_URL=f"http://127.0.0.1:{server.server_port}"))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_debug_screen_is_separate_and_offers_every_agent_operation(self):
        from agent_types import MOTION_OPS
        from debug_console import OPERATIONS
        html = (PANEL_DIR / "panel.html").read_text()
        self.assertIn('<script defer src="/debug.js"></script>', html)
        self.assertIn('id="open-debug"', html.split('<main', 1)[0])
        self.assertRegex(html, r'<section id="debug-screen"[^>]*\bdata-screen="debug"[^>]*\bhidden>')
        debug = html.split('<section id="debug-screen"', 1)[1].split("</section>", 1)[0]
        for element in ("debug-tools", "debug-health", "debug-image", "debug-occupancy",
                        "debug-warnings", "debug-json", "debug-viz", "debug-log", "debug-stop",
                        "debug-arm", "debug-executor", "debug-capabilities"):
            with self.subTest(element=element):
                self.assertIn(f'id="{element}"', debug)
        for screen in ("main-screen", "build-screen", "complete-screen"):
            with self.subTest(screen=screen):
                section = html.split(f'<section id="{screen}"', 1)[1].split("</section>", 1)[0]
                self.assertNotIn("debug-", section)
        script = (PANEL_DIR / "debug.js").read_text()
        offered = set(re.findall(r"\{id: '([a-z_]+)'", script))
        self.assertTrue(set(OPERATIONS) <= offered, set(OPERATIONS)-offered)
        for operation in MOTION_OPS:
            with self.subTest(operation=operation):
                self.assertRegex(script, rf"\{{id: '{operation}'[^}}]*motion: true")

    @unittest.skipUnless(importlib.util.find_spec("quickjs"), "optional JavaScript engine not installed")
    def test_javascript_syntax(self):
        import quickjs
        for name in ("panel.js", "debug.js"):
            with self.subTest(script=name):
                source = (PANEL_DIR / name).read_text()
                quickjs.Context().eval("new Function("+json.dumps(source)+")")

    @unittest.skipUnless(importlib.util.find_spec("quickjs"), "optional JavaScript engine not installed")
    def test_3d_preview_draws_visible_top_and_rotates(self):
        import quickjs
        source = (PANEL_DIR / "panel.js").read_text().split("let state = null")[0]
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
        session = PanelSession(FirstChoice, pace=0)
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
            self.assertIsNone(session.reasoner_factory)

    def test_debug_console_drives_the_agents_tools_by_hand(self):
        from fastapi.testclient import TestClient
        from panel import create_app
        console = debug_console()
        session = PanelSession(FirstChoice, pace=0, debug=console)
        with TestClient(create_app(session, receiver_host="127.0.0.1", receiver_port=0)) as client:
            state = client.get("/api/state").json()
            headers = {"X-Crafter-Token": state["csrf"]}
            self.assertEqual(state["debug_kind"], "test double")
            self.assertEqual(client.get("/debug.js").status_code, 200)
            self.assertEqual(client.post("/api/debug/observe", json={}).status_code, 403)

            self.assertEqual(client.post("/api/debug/open", headers=headers).status_code, 200)
            self.assertEqual(client.get("/api/state").json()["view"], "debug")
            self.assertEqual(client.get("/api/debug/image?view=rect").status_code, 503)

            snapshot = client.post("/api/debug/observe", json={}, headers=headers).json()["snapshot"]
            self.assertTrue(snapshot["valid"])
            self.assertEqual(len(snapshot["boxes"]), 4)
            image = client.get("/api/debug/image?view=" + snapshot["images"][0]["view"])
            self.assertEqual(image.status_code, 200)
            self.assertTrue(image.headers["Content-Type"].startswith("image/"))

            sites = client.post("/api/debug/sites", headers=headers).json()["sites"]
            self.assertEqual([site["id"] for site in sites], ["obstructed", "floor-a"])
            selected = client.post("/api/debug/select", json={"site_id": "floor-a"}, headers=headers)
            self.assertEqual(selected.json()["site"]["id"], "floor-a")
            client.post("/api/debug/observe", json={"site_id": "floor-a"}, headers=headers)

            for body in ({"operation": "done"}, {"operation": "approach_box", "box_id": -1},
                         {"operation": "approach_box", "cell": [0, 0]}, {"operation": "look_around", "search": "elsewhere"}):
                with self.subTest(body=body):
                    self.assertEqual(client.post("/api/debug/action", json=body, headers=headers).status_code, 400)

            action = {"operation": "approach_box", "box_id": 0, "site_id": "floor-a", "cell": [0, 0, 0]}
            unarmed = client.post("/api/debug/action", json=action, headers=headers)
            self.assertEqual(unarmed.status_code, 409)
            self.assertIn("arm actions", unarmed.json()["detail"])
            self.assertFalse(client.get("/api/debug").json()["actions_open"])

            self.assertEqual(client.post("/api/debug/arm", json={"armed": "yes"}, headers=headers).status_code, 400)
            armed = client.post("/api/debug/arm", json={"armed": True}, headers=headers)
            self.assertEqual(armed.status_code, 200)
            self.assertTrue(armed.json()["armed"] and armed.json()["actions_open"])

            started = client.post("/api/debug/action", json=action, headers=headers)
            self.assertEqual(started.status_code, 200)
            self.assertEqual(started.json()["active"]["operation"], "approach_box")
            deadline = time.monotonic()+6
            while client.get("/api/debug").json()["active"] and time.monotonic() < deadline:
                time.sleep(.02)
            finished = client.get("/api/debug").json()
            self.assertIsNone(finished["active"])
            self.assertEqual(finished["outcome"]["status"], "succeeded")
            self.assertTrue(any(entry["kind"] == "action" for entry in finished["log"]))

            self.assertEqual(client.post("/api/debug/verify", headers=headers).status_code, 200)
            self.assertEqual(client.post("/api/debug/stop", headers=headers).status_code, 200)
            self.assertEqual(client.post("/api/main", headers=headers).status_code, 200)
            self.assertEqual(client.get("/api/state").json()["view"], "main")

    def test_a_panel_without_a_console_exposes_no_debug_surface(self):
        from fastapi.testclient import TestClient
        from panel import create_app
        session = PanelSession(FirstChoice, pace=0)
        with TestClient(create_app(session, receiver_host="127.0.0.1", receiver_port=0)) as client:
            headers = {"X-Crafter-Token": client.get("/api/state").json()["csrf"]}
            self.assertIsNone(client.get("/api/state").json()["debug_kind"])
            for path in ("/api/debug", "/api/debug/image?view=rect"):
                with self.subTest(path=path):
                    self.assertEqual(client.get(path).status_code, 404)
            for path in ("/api/debug/open", "/api/debug/observe", "/api/debug/sites", "/api/debug/stop"):
                with self.subTest(path=path):
                    self.assertEqual(client.post(path, json={}, headers=headers).status_code, 404)

    def test_a_running_build_keeps_the_manual_tools_out_of_the_way(self):
        from fastapi.testclient import TestClient
        from panel import create_app
        release = threading.Event()
        self.addCleanup(release.set)

        class Blocking:
            def decide(self, context, choices):
                release.wait(5)
                return choices[0]

        console = debug_console()
        session = PanelSession(Blocking, pace=0, debug=console)
        with TestClient(create_app(session, receiver_host="127.0.0.1", receiver_port=0)) as client:
            headers = {"X-Crafter-Token": client.get("/api/state").json()["csrf"]}
            client.post("/api/example", headers=headers)
            design_id = client.get("/api/state").json()["design"]["id"]
            self.assertEqual(client.post("/api/builds", json={"design_id": design_id}, headers=headers).status_code, 200)
            client.post("/api/debug/open", headers=headers)
            self.assertEqual(client.get("/api/state").json()["view"], "debug")
            self.assertEqual(client.post("/api/debug/observe", json={}, headers=headers).status_code, 200)
            action = {"operation": "look_around", "search": "materials"}
            refused = client.post("/api/debug/action", json=action, headers=headers)
            self.assertEqual(refused.status_code, 409)
            self.assertIn("build is running", refused.json()["detail"])
            self.assertEqual(client.post("/api/main", headers=headers).status_code, 200)
            self.assertEqual(client.get("/api/state").json()["view"], "build")
            release.set()
            session.cancel()
            self.assertTrue(session.wait(5))

    def test_http_assets_state_protection_and_complete_flow(self):
        from fastapi.testclient import TestClient
        from panel import create_app
        session = PanelSession(FirstChoice, pace=0)
        with TestClient(create_app(session, receiver_host="127.0.0.1", receiver_port=0)) as client:
            for path, media_type in (("/", "text/html"), ("/panel.js", "text/javascript"),
                                     ("/panel.css", "text/css"), ("/assets/Crafter-transparent.svg", "image/svg+xml")):
                response = client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.headers["Content-Type"].startswith(media_type))
                self.assertIn("Content-Security-Policy", response.headers)
            self.assertEqual(response.content, (PANEL_DIR / "assets" / "Crafter-transparent.svg").read_bytes())
            self.assertEqual(client.get("/assets/main.py").status_code, 404)
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
