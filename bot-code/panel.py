import asyncio
import copy
import json
import math
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from agent import Agent, JobCancelled, validate_job
from agent_backend import OpenAIReasoner
from agent_types import MOTION_OPS, AgentConfig, cell_valid
from contracts import Block, Structure
from debug_console import DEFAULT_VOXEL, DebugConsole, perception_sources, provider_sources, robot_sources
from mock_agent_world import MockAgentWorld

PANEL_DIR = Path(__file__).parent / "web"
PANEL_CONFIG = AgentConfig(max_blocks=64, max_actions=384, max_steps=512, max_no_progress=12)
MAX_MESSAGE = 512 * 1024
EXAMPLE = {"origin": [0, 1, -1], "size": [1, 2, 3], "count": 4,
           "palette": ["minecraft:oak_planks"],
           "blocks": [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 2, 0]]}


@dataclass
class ParsedDesign:
    structure: Structure
    origin: list
    size: list
    palette: list
    buildable: bool
    error: str | None


def parse_design(data, config=PANEL_CONFIG):
    if not isinstance(data, dict):
        raise ValueError("expected a Minecraft structure object")
    origin, size = data.get("origin"), data.get("size")
    if not cell_valid(origin) or any(abs(v) > 30_000_000 for v in origin):
        raise ValueError("origin must contain three Minecraft integer coordinates")
    if not cell_valid(size) or any(not 1 <= v <= 1024 for v in size):
        raise ValueError("size must contain three positive bounded extents")
    count, blocks, palette = data.get("count"), data.get("blocks"), data.get("palette")
    if type(count) is not int or not 1 <= count <= 512:
        raise ValueError("design must contain between 1 and 512 blocks")
    if not isinstance(blocks, list) or len(blocks) != count:
        raise ValueError("block count does not match the received blocks")
    if (not isinstance(palette, list) or not 1 <= len(palette) <= 512
            or any(not isinstance(p, str) or not 1 <= len(p) <= 128 for p in palette)):
        raise ValueError("palette must contain bounded block names")
    seen, converted = set(), []
    for entry in blocks:
        if not isinstance(entry, list) or len(entry) != 4 or any(type(v) is not int for v in entry):
            raise ValueError("each block must contain four integers")
        x, y, z, index = entry
        cell = (x, y, z)
        if any(not 0 <= n < extent for n, extent in zip(cell, size)) or not 0 <= index < len(palette):
            raise ValueError("block is outside its bounds or palette")
        if cell in seen:
            raise ValueError("duplicate block coordinates")
        seen.add(cell)
        converted.append(Block(x, y, z, palette[index]))
    structure = Structure(converted)
    error = None
    try:
        validate_job(structure, "preview", config)
    except ValueError as exc:
        error = str(exc)
    return ParsedDesign(structure, list(origin), list(size), list(palette), error is None, error)


class StructureReceiver:
    def __init__(self, session, host="0.0.0.0", port=5005, timeout=5.0):
        self.session, self.host, self.port, self.timeout = session, host, port, timeout
        self.server = None
        self.clients = set()

    async def start(self):
        self.server = await asyncio.start_server(self._client, self.host, self.port, backlog=16)
        self.port = self.server.sockets[0].getsockname()[1]
        self.session.receiver_status(True, self.port)
        return self

    async def _client(self, reader, writer):
        task = asyncio.current_task()
        if len(self.clients) >= 16:
            writer.close()
            return
        self.clients.add(task)
        sequence = self.session.next_sequence()
        try:
            raw = await asyncio.wait_for(self._read_message(reader), self.timeout)
            self.session.receive(json.loads(raw.decode("utf-8")), sequence=sequence)
        except (ValueError, UnicodeError, TimeoutError, asyncio.TimeoutError, ConnectionError) as exc:
            message = str(exc) if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError) else "invalid, incomplete, or timed-out message"
            self.session.receiver_notice(f"Minecraft design rejected: {message[:180]}")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            self.clients.discard(task)

    async def _read_message(self, reader):
        chunks, total = [], 0
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                return b"".join(chunks)
            total += len(chunk)
            if total > MAX_MESSAGE:
                raise ValueError("message exceeds the 512 KiB limit")
            chunks.append(chunk)

    async def close(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        tasks = list(self.clients)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.session.receiver_status(False, self.port)


ACTION_SECONDS = {"look_around": 9.0, "approach_box": 6.0, "pickup": 8.0,
                  "move_to_build": 6.0, "place": 8.0}
OTHER_ACTION_SECONDS = 6.0


class PanelWorld(MockAgentWorld):
    """The simulated world, paced so a build reads as physical work rather than a progress bar.

    The mock runs on virtual time, which the agent advances by sleeping. Stretching those sleeps
    against the wall clock is what makes an action take real seconds, and doing it per operation
    keeps the proportions the robot's own routines have -- a survey is long, a drive is short.
    Cancellation waits on the same event, so a slow action still stops the moment it is asked to.
    """

    def __init__(self, count, cancel, emit, pace):
        super().__init__(count)
        self.cancel_event, self.emit, self.pace = cancel, emit, pace
        self.reported = set()

    def submit(self, request):
        if self.cancel_event.is_set():
            raise JobCancelled("cancelled")
        receipt = super().submit(request)
        self.emit("tool_start", operation=request.step.operation, request_id=request.request_id,
                  arguments=asdict(request.step))
        return receipt

    def status(self, action_id):
        outcome = super().status(action_id)
        if outcome.status != "running" and action_id not in self.reported:
            self.reported.add(action_id)
            request = self._pending[action_id]["request"]
            self.emit("tool_result", operation=request.step.operation, request_id=request.request_id,
                      result="success" if outcome.status == "succeeded" else outcome.status)
        return outcome

    def sleep(self, seconds):
        self.cancel_event.wait(seconds*self.pace*self._seconds()/self.action_duration)
        super().sleep(seconds)

    def _seconds(self):
        """Wall-clock budget for whatever is running now; nothing between actions has to drag."""
        if self.active is None:
            return 0.0
        operation = self._pending[self.active]["request"].step.operation
        return ACTION_SECONDS.get(operation, OTHER_ACTION_SECONDS)


class DeterministicChoice:
    """The agent's own first legal step, taken without a model.

    Agent.choices() already returns only steps whose preconditions hold, ordered so the first is
    the one a correct reasoner would pick, so this drives a whole build with no API call, no key
    and no per-run cost.
    """

    def decide(self, context, choices):
        return choices[0]


class PanelReasoner:
    def __init__(self, reasoner, cancel, emit):
        self.reasoner, self.cancel, self.emit = reasoner, cancel, emit

    def decide(self, context, choices):
        if self.cancel.is_set():
            raise JobCancelled("cancelled before model call")
        call_id = uuid.uuid4().hex
        self.emit("llm_start", call_id=call_id, phase=context["phase"],
                  input={"inventory": len(context["inventory"]), "placed": len(context["confirmed"]),
                         "choices": [asdict(c) for c in choices]})
        started = time.monotonic()
        try:
            decision = self.reasoner.decide(context, choices)
        except Exception as exc:
            self.emit("llm_error", call_id=call_id, error=type(exc).__name__)
            raise
        if self.cancel.is_set():
            raise JobCancelled("cancelled during model call")
        self.emit("llm_result", call_id=call_id, duration=round(time.monotonic()-started, 2),
                  decision=asdict(decision))
        return decision


class PanelSession:
    def __init__(self, reasoner_factory=None, model="gpt-4o-mini", pace=1.0,
                 base_url=None, model_timeout=15.0, json_only=False, debug=None):
        if not math.isfinite(pace) or not 0 <= pace <= 10:
            raise ValueError("build pace must be a multiplier between 0 and 10")
        self.reasoner_factory, self.model, self.pace = reasoner_factory, model, pace
        self.base_url, self.model_timeout, self.json_only = base_url, model_timeout, json_only
        self.debug = debug
        self.lock = threading.RLock()
        self.csrf = secrets.token_urlsafe(32)
        self.design = None
        self.job = None
        self.view = "main"
        self.notice = ""
        self.sequence = 0
        self.design_sequence = 0
        self.revision = 0
        self.receiver = {"listening": False, "port": 5005, "error": None}
        self.worker = None
        self.worker_busy = False
        self.cancel_event = threading.Event()
        self.agent = None
        self.closed = False

    def set_api_key(self, key):
        if not isinstance(key, str):
            raise ValueError("Enter an API key in the masked field")
        key = key.strip()
        if not 16 <= len(key) <= 512 or any(not 33 <= ord(c) <= 126 for c in key):
            raise ValueError("API key must be 16–512 characters without whitespace")
        with self.lock:
            if self.closed or self.worker_busy:
                raise ValueError("Wait for the current build to stop before changing the API key")
            self.reasoner_factory = lambda: OpenAIReasoner(
                model=self.model, base_url=self.base_url, api_key=key,
                timeout=self.model_timeout, json_only=self.json_only)
            self.revision += 1

    def next_sequence(self):
        with self.lock:
            self.sequence += 1
            return self.sequence

    def receive(self, data, sequence=None, source="Minecraft"):
        sequence = sequence if sequence is not None else self.next_sequence()
        parsed = parse_design(data)
        dx = min(b.x for b in parsed.structure.blocks)
        dz = min(b.z for b in parsed.structure.blocks)
        blocks = [{"x": b.x-dx, "y": b.y, "z": b.z-dz, "kind": b.kind} for b in parsed.structure.blocks]
        design = {"id": str(sequence), "received_at": time.time(), "source": source,
                  "count": len(blocks), "blocks": blocks, "origin": parsed.origin,
                  "size": [max(b[axis] for b in blocks)+1 for axis in ("x", "y", "z")],
                  "palette": parsed.palette, "buildable": parsed.buildable, "error": parsed.error}
        with self.lock:
            if sequence <= self.design_sequence:
                return False
            self.design, self.design_sequence = design, sequence
            self.notice = ""
            self.revision += 1
            return True

    def clear_design(self, design_id):
        with self.lock:
            if not self.design or self.design["id"] != design_id:
                raise ValueError("design changed; review the latest preview before clearing")
            self.design = None
            self.design_sequence = max(self.design_sequence, self.sequence)
            self.notice = ""
            self.revision += 1

    def receiver_status(self, listening, port, error=None):
        with self.lock:
            self.receiver = {"listening": listening, "port": port, "error": error}
            self.revision += 1

    def receiver_notice(self, message):
        with self.lock:
            self.notice = message
            self.revision += 1

    def snapshot(self):
        with self.lock:
            return copy.deepcopy({"revision": self.revision, "view": self.view, "design": self.design,
                                  "job": self.job, "notice": self.notice, "receiver": self.receiver,
                                  "worker_busy": self.worker_busy, "llm_ready": True,
                                  "model": self.model, "csrf": self.csrf,
                                  "debug_kind": self.debug.kind if self.debug else None})

    def start(self, design_id):
        with self.lock:
            if self.closed:
                raise ValueError("panel is shutting down")
            if self.worker_busy:
                raise ValueError("busy: previous build or model request is still finishing")
            if not self.design:
                raise ValueError("wait for a Minecraft design first")
            if self.design["id"] != design_id:
                raise ValueError("design changed; review the latest preview before starting")
            if not self.design["buildable"]:
                raise ValueError(self.design["error"])
            job_id = uuid.uuid4().hex
            self.cancel_event = threading.Event()
            self.job = {"id": job_id, "design": copy.deepcopy(self.design), "status": "running",
                        "started_at": time.time(), "finished_at": None, "phase": "INVENTORY",
                        "reasoning": "Waiting for the first model decision.", "events": [], "placed": [],
                        "llm_calls": 0, "tool_calls": 0, "current_tool": None, "current_cell": None,
                        "error": None, "result": None}
            self.view, self.worker_busy = "build", True
            self.revision += 1
            self.worker = threading.Thread(target=self._run, args=(job_id, self.cancel_event),
                                           name="panel-build", daemon=True)
            self.worker.start()
            return job_id

    def _emit(self, job_id, event_type, **data):
        with self.lock:
            if not self.job or self.job["id"] != job_id or self.job["status"] != "running":
                return
            job = self.job
            self.revision += 1
            event = {"id": self.revision, "type": event_type, "at": time.time(), "data": data}
            job["events"].append(event)
            del job["events"][:-200]
            if event_type == "llm_start":
                job["llm_calls"] += 1
                job["phase"] = data["phase"]
            elif event_type == "llm_result":
                job["reasoning"] = data["decision"]["reason"]
            elif event_type == "tool_start":
                job["tool_calls"] += 1
                job["current_tool"] = data["operation"]
                job["current_cell"] = data["arguments"].get("cell")
            elif event_type == "tool_result":
                job["current_tool"] = None
            elif event_type == "placement_confirmed":
                job["placed"].append({"cell": list(data["cell"]), "box_id": data["box_id"]})
                job["current_cell"] = None

    def _run(self, job_id, cancel):
        emit = lambda event_type, **data: self._emit(job_id, event_type, **data)
        try:
            with self.lock:
                blocks = copy.deepcopy(self.job["design"]["blocks"])
            world = PanelWorld(len(blocks), cancel, emit, self.pace)
            reasoner = PanelReasoner((self.reasoner_factory or DeterministicChoice)(), cancel, emit)
            agent = Agent(world.actions, world.observations, config=PANEL_CONFIG, backend="llm", reasoner=reasoner,
                          clock=world.clock, sleep=world.sleep,
                          event_sink=lambda event: emit(event["event"], **{k: v for k, v in event.items() if k != "event"}))
            with self.lock:
                self.agent = agent
            result = agent.run(Structure([Block(**b) for b in blocks]), job_id=job_id)
            with self.lock:
                if not cancel.is_set() and self.job["id"] == job_id:
                    self.job["result"] = {"status": result.status, "reason": result.reason,
                                          "placed": result.placed, "required": result.required}
                    self.job["status"] = "completed" if result.success else "failed"
                    self.job["error"] = None if result.success else result.reason
                    self.job["finished_at"] = time.time()
                    self.view = "complete" if result.success else "build"
        except Exception as exc:
            with self.lock:
                if not cancel.is_set() and self.job["id"] == job_id:
                    self.job["status"] = "failed"
                    self.job["error"] = f"Build could not run ({type(exc).__name__}). Check the server configuration."
                    self.job["finished_at"] = time.time()
        finally:
            with self.lock:
                if self.job and self.job["id"] == job_id:
                    self.job["current_tool"] = None
                    self.worker_busy = False
                    self.agent = None
                    self.revision += 1

    def cancel(self, job_id=None):
        with self.lock:
            if job_id is not None and (not self.job or self.job["id"] != job_id):
                raise ValueError("the active build changed; refresh before cancelling")
            if self.job and self.job["status"] == "running":
                self.cancel_event.set()
                if self.agent:
                    self.agent.cancel()
                self.job["status"] = "cancelled"
                self.job["finished_at"] = time.time()
            self.view = "main"
            self.revision += 1

    def back(self):
        with self.lock:
            if self.job and self.job["status"] == "running":
                if self.view != "debug":
                    raise ValueError("cancel the active build before returning")
                self.view = "build"      # the debug screen is a detour, not an exit from the build
            else:
                self.view = "main"
            self.revision += 1

    def open_debug(self):
        with self.lock:
            if self.debug is None:
                raise ValueError("the debug console is not available in this panel")
            self.view = "debug"
            self.revision += 1

    def debug_requirements(self, source):
        """Requirements the manual tools admit actions against: the live design, or one cell."""
        if self.debug is None:
            raise ValueError("the debug console is not available in this panel")
        if source == "design":
            with self.lock:
                design = copy.deepcopy(self.design)
            if not design:
                raise ValueError("no Minecraft design has been received yet")
            config = AgentConfig(voxel_size=self.debug.voxel_size, max_blocks=PANEL_CONFIG.max_blocks)
            job = validate_job(Structure([Block(**b) for b in design["blocks"]]),
                               self.debug.job_id, config)
            return self.debug.set_requirements(job.requirements, f"Minecraft design {design['id']}")
        if source == "cell":
            config = AgentConfig(voxel_size=self.debug.voxel_size)
            job = validate_job(Structure([Block(0, 0, 0)]), self.debug.job_id, config)
            return self.debug.set_requirements(job.requirements, "single cell at the site origin")
        raise ValueError("requirements come from the current design or a single cell")

    def wait(self, timeout):
        thread = self.worker
        if thread:
            thread.join(timeout)
        return not self.worker_busy

    def close(self):
        with self.lock:
            self.closed = True
            debug = self.debug
        self.cancel()
        self.wait(.1)
        if debug is not None:
            debug.close()
        with self.lock:
            self.reasoner_factory = None


def build_debug_console(provider=None, perception=None, box_size=None):
    """The real providers behind the manual tool bench. There is no simulated option by design.

    Default is the deployed shape: perception in this process, actions from the interface-v2
    provider once armed. The two overrides exist for development against a perception server that
    is already running, or a different provider factory.
    """
    voxel = (box_size,)*3 if box_size else DEFAULT_VOXEL
    if provider:
        observations, actions = provider_sources(provider)
        kind = f"live providers from {provider}"
    elif perception:
        observations, actions = perception_sources(perception)
        kind = "live perception, no motion owner"
    else:
        observations, actions = robot_sources(voxel)
        kind = "this robot, in process"
    return DebugConsole(observations, actions, kind=kind, voxel_size=voxel,
                        perception_url=perception)


def create_app(session=None, receiver_host="0.0.0.0", receiver_port=5005, model="gpt-4o-mini",
               base_url=None, model_timeout=15.0, json_only=False, pace=1.0,
               debug_provider=None, debug_perception=None, box_size=None):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, Response

    if session is None:
        # Builds run on the agent's own legal-step ordering, so the panel starts with no reasoner
        # and no key. set_api_key installs one for anyone who wants model decisions back.
        session = PanelSession(None, model, pace, base_url, model_timeout, json_only,
                               debug=build_debug_console(debug_provider, debug_perception, box_size))
    receiver = StructureReceiver(session, receiver_host, receiver_port)
    origin = urlsplit(debug_perception) if debug_perception else None
    frame_src = f"{origin.scheme}://{origin.netloc}" if origin else "'none'"

    @asynccontextmanager
    async def lifespan(app):
        try:
            await receiver.start()
        except OSError:
            session.receiver_status(False, receiver_port, "TCP port unavailable; another receiver may be running")
        try:
            yield
        finally:
            await receiver.close()
            session.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.panel = session

    @app.middleware("http")
    async def protect(request, call_next):
        if request.method == "POST":
            supplied = request.headers.get("x-crafter-token", "")
            if not secrets.compare_digest(supplied.encode(), session.csrf.encode()):
                return JSONResponse({"detail": "refresh the panel before sending a command"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                                                       "img-src 'self' data:; connect-src 'self'; base-uri 'none'; object-src 'none'; "
                                                       f"frame-src {frame_src}")
        return response

    @app.get("/")
    def index():
        return FileResponse(PANEL_DIR / "panel.html", media_type="text/html")

    @app.get("/panel.js")
    def script():
        return FileResponse(PANEL_DIR / "panel.js", media_type="text/javascript")

    @app.get("/panel.css")
    def stylesheet():
        return FileResponse(PANEL_DIR / "panel.css", media_type="text/css")

    @app.get("/debug.js")
    def debug_script():
        return FileResponse(PANEL_DIR / "debug.js", media_type="text/javascript")

    @app.get("/assets/Crafter-transparent.svg")
    def logo():
        return FileResponse(PANEL_DIR / "assets" / "Crafter-transparent.svg", media_type="image/svg+xml")

    @app.get("/api/state")
    def state():
        return session.snapshot()

    @app.post("/api/key")
    async def configure_key(request: Request):
        if request.url.scheme != "https" and (not request.client or request.client.host not in {"127.0.0.1", "::1"}):
            raise HTTPException(403, "Use a local/forwarded connection or HTTPS to enter the API key")
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 4096:
                raise HTTPException(413, "Key request is too large")
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeError):
            raise HTTPException(400, "Invalid key request") from None
        if not isinstance(body, dict) or set(body) != {"api_key"}:
            raise HTTPException(400, "Invalid key request")
        try:
            session.set_api_key(body["api_key"])
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"ok": True}

    @app.post("/api/builds")
    def start(body: dict):
        try:
            return {"job_id": session.start(body.get("design_id"))}
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/cancel")
    def cancel(body: dict):
        if not isinstance(body.get("job_id"), str):
            raise HTTPException(400, "job_id is required")
        try:
            session.cancel(body["job_id"])
            return {"ok": True}
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/main")
    def back():
        try:
            session.back()
            return {"ok": True}
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/clear")
    def clear(body: dict):
        try:
            session.clear_design(body.get("design_id"))
            return {"ok": True}
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/example")
    def example():
        session.receive(copy.deepcopy(EXAMPLE), source="Example")
        return {"ok": True}

    def console():
        if session.debug is None:
            raise HTTPException(404, "the debug console is not available in this panel")
        return session.debug

    def run(call, *args, **kwargs):
        """Every manual tool answers with the console's whole state, so one call refreshes the page."""
        console()
        try:
            call(*args, **kwargs)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return console().state()

    @app.get("/api/debug")
    def debug_state():
        return console().state()

    @app.get("/api/debug/image")
    def debug_image(view: str = "rect"):
        if not isinstance(view, str) or not 0 < len(view) <= 64:
            raise HTTPException(400, "unknown image view")
        image = console().image(view)
        if image is None:
            raise HTTPException(503, "no image of that view in the latest observation")
        media_type, data = image
        return Response(data, media_type=media_type, headers={"Cache-Control": "no-store"})

    @app.post("/api/debug/open")
    def debug_open():
        return run(session.open_debug)

    @app.post("/api/debug/arm")
    def debug_arm(body: dict):
        if type(body.get("armed")) is not bool:
            raise HTTPException(400, "armed must be true or false")
        return run(console().arm, body["armed"])

    @app.post("/api/debug/requirements")
    def debug_requirements(body: dict):
        return run(session.debug_requirements, body.get("source"))

    @app.post("/api/debug/observe")
    def debug_observe(body: dict):
        site_id = body.get("site_id")
        if site_id is not None and (not isinstance(site_id, str) or not 0 < len(site_id) <= 128):
            raise HTTPException(400, "site_id must be a bounded string")
        return run(console().observe, site_id)

    @app.post("/api/debug/sites")
    def debug_sites():
        return run(console().find_sites)

    @app.post("/api/debug/select")
    def debug_select(body: dict):
        return run(console().select_site, body.get("site_id"))

    @app.post("/api/debug/verify")
    def debug_verify():
        return run(console().verify)

    @app.post("/api/debug/action")
    def debug_action(body: dict):
        operation, cell = body.get("operation"), body.get("cell")
        box_id, site_id, search = body.get("box_id"), body.get("site_id"), body.get("search")
        if operation not in MOTION_OPS:
            raise HTTPException(400, "operation must be one of the agent's motion operations")
        if box_id is not None and (type(box_id) is not int or box_id < 0):
            raise HTTPException(400, "box_id must be a nonnegative integer")
        if site_id is not None and (not isinstance(site_id, str) or not 0 < len(site_id) <= 128):
            raise HTTPException(400, "site_id must be a bounded string")
        if cell is not None and not cell_valid(cell):
            raise HTTPException(400, "cell must contain three integers")
        if search is not None and search not in {"materials", "sites"}:
            raise HTTPException(400, "search must be materials or sites")
        if session.snapshot()["worker_busy"]:
            raise HTTPException(409, "a build is running; cancel it before driving tools by hand")
        return run(console().submit, operation, box_id=box_id, site_id=site_id,
                   cell=cell, search=search)

    @app.post("/api/debug/cancel")
    def debug_cancel():
        return run(console().cancel)

    @app.post("/api/debug/stop")
    def debug_stop():
        return run(console().stop)

    return app


def serve_panel(host="127.0.0.1", port=8005, receiver_host="0.0.0.0", receiver_port=5005, **kwargs):
    import uvicorn
    app = create_app(receiver_host=receiver_host, receiver_port=receiver_port, **kwargs)
    console = app.state.panel.debug
    print(f"[panel] http://{host}:{port} | Minecraft TCP {receiver_port}", flush=True)
    if console is not None:
        print(f'[panel] debug screen: open the panel, then "Debug console" | manual tools against {console.kind} providers'
              + (f" | perception {console.perception_url}" if console.perception_url else ""), flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")
