import base64
import copy
import json
import sys
import threading
import time
import unittest
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import CapabilityError, preflight
from agent_adapters import (FunctionActions, FunctionResult, ObservationWorker,
                            normalize_scan, normalize_skill_result, load_providers)
from agent_types import (INTERFACE_VERSION, ActionRequest, AgentConfig, BoxObservation,
                         BuildRequirements, Capabilities, ExecutorState, Holding, PerceptionCapabilities, Step)
from mock_agent_world import MockAgentWorld


def scan():
    now = time.time()
    pose = SimpleNamespace(valid=True, epoch=3, x=0.0, y=0.0, yaw=0.0)
    return SimpleNamespace(ts=now, pose=pose, warnings=[], boxes=[], protected=[], unknown=[],
                           tracks=[{"id": 7, "world": [1.0, 2.0, .15], "current": True,
                                    "classification": "protected", "size": .3, "last_seen": now},
                                   {"id": 8, "world": [-1.0, 2.0, .15], "current": False,
                                    "classification": "loose", "size": .3, "last_seen": now-10}])


class AdapterTests(unittest.TestCase):
    def request(self, operation="look_around", request_id="one", now=None):
        now = time.time() if now is None else now
        box = BoxObservation(0, (1.0, 0.0, .15), (.3, .3, .3), now, eligible=True)
        step = Step("pickup", box_id=0) if operation == "pickup" else Step("look_around", search="materials")
        requirements = BuildRequirements(((0, 0, 0),), (.3, .3, .3), (1, 1, 1))
        return ActionRequest(request_id, "job", step, "snapshot", 0, "world", now,
                             requirements=requirements, box=box if operation == "pickup" else None,
                             expires_at=now+2)

    def test_inventory_includes_protected_but_does_not_authorize_picking(self):
        snapshot = normalize_scan(scan(), time.time())
        self.assertEqual({box.id for box in snapshot.boxes}, {7, 8})
        self.assertTrue(all(box.eligible is None for box in snapshot.boxes))
        self.assertFalse(snapshot.boxes[1].current)
        self.assertIsNone(snapshot.holding)
        self.assertFalse(snapshot.occupancy_complete)
        self.assertEqual(snapshot.epoch, 3)

    def test_eligibility_requires_explicit_provider_mapping(self):
        snapshot = normalize_scan(scan(), time.time(), eligibility=lambda track: track["id"] == 7)
        self.assertTrue(snapshot.boxes[0].eligible)
        self.assertFalse(snapshot.boxes[1].eligible)

    def test_nested_skill_failure_is_not_success_or_possession(self):
        self.assertEqual(normalize_skill_result({"ok": True, "result": [False, "approach IK failed"]}), "failed")
        self.assertEqual(normalize_skill_result({"ok": True, "result": [True, "ok"]}), "succeeded")
        self.assertEqual(normalize_skill_result({"ok": True}), "unknown")
        self.assertEqual(normalize_skill_result({"ok": True, "result": [1, "ok"]}), "unknown")
        self.assertEqual(normalize_skill_result({"ok": False, "error": "busy"}), "failed")

    def test_worker_creates_polls_and_closes_on_same_thread(self):
        calls = []
        class Session:
            def __init__(self):
                calls.append(("create", threading.get_ident()))
            def poll(self):
                calls.append(("poll", threading.get_ident()))
                return scan()
            def close(self):
                calls.append(("close", threading.get_ident()))
        with ObservationWorker(Session, interval=.001, timeout=.5) as worker:
            first = worker.observe()
            time.sleep(.02)
            second = worker.observe()
            self.assertNotEqual(first.revision, second.revision)
            self.assertEqual(first.boxes[0].position, (1.0, 2.0, .15))
            with self.assertRaises(CapabilityError):
                worker.find_build_sites(None)
        self.assertEqual(calls[-1][0], "close")
        self.assertEqual(len({tid for _, tid in calls}), 1)
        self.assertNotEqual(calls[0][1], threading.get_ident())

    def test_worker_timeout_error_and_close_are_bounded(self):
        class NoFrames:
            def poll(self):
                return None
            def close(self):
                pass
        with ObservationWorker(NoFrames, interval=.001, timeout=.01) as worker:
            with self.assertRaises(TimeoutError):
                worker.observe()
        class Broken:
            def __init__(self):
                raise RuntimeError("sensor unavailable")
        with ObservationWorker(Broken, interval=.001, timeout=.1) as worker:
            with self.assertRaises(RuntimeError):
                worker.observe()

    def test_missing_geometry_capabilities_fail_preflight(self):
        world = MockAgentWorld(3)
        worker = ObservationWorker(lambda: None)
        req = BuildRequirements(((0, 0, 0),), AgentConfig().voxel_size, (1, 1, 1))
        with self.assertRaisesRegex(CapabilityError, "sites"):
            preflight(world.actions, worker, req)
        self.assertFalse(world.requests)

    def test_live_provider_must_be_explicit(self):
        with self.assertRaisesRegex(CapabilityError, "provider"):
            load_providers(None)
        with self.assertRaises(ValueError):
            load_providers("missing-colon")

    def test_plain_action_function_runs_once_for_repeated_request(self):
        calls = []
        def action(step):
            calls.append(step)
            return FunctionResult(True)
        def state():
            return ExecutorState(Holding("empty", ts=time.time(), source="test"), time.time(),
                                 ready=True, motion="stopped")
        adapter = FunctionActions({"look_around": action}, state, lambda: None,
                                  max_box_size=(.3, .3, .3), max_height=1.0)
        request = self.request()
        receipt = adapter.submit(request)
        self.assertEqual(adapter.submit(request), receipt)
        deadline = time.monotonic()+1
        while adapter.status(receipt.action_id).status == "running" and time.monotonic() < deadline:
            time.sleep(.001)
        self.assertEqual(adapter.status(receipt.action_id).status, "succeeded")
        self.assertEqual(calls, [request])
        self.assertEqual(adapter.lookup(request.request_id), receipt)
        self.assertIsNone(adapter.lookup("unknown-request"))
        self.assertIsNone(adapter.state().active_action)

    def test_plain_function_exception_is_unknown_not_safe_failure(self):
        stopped = threading.Event()
        def action(step):
            raise TimeoutError("may have moved")
        def state():
            return ExecutorState(Holding("empty", ts=time.time(), source="test"), time.time(),
                                 ready=True, motion="stopped")
        adapter = FunctionActions({"pickup": action}, state, stopped.set,
                                  max_box_size=(.3, .3, .3), max_height=1.0)
        request = self.request("pickup", "two")
        receipt = adapter.submit(request)
        deadline = time.monotonic()+1
        while adapter.status(receipt.action_id).status == "running" and time.monotonic() < deadline:
            time.sleep(.001)
        outcome = adapter.status(receipt.action_id)
        self.assertEqual(outcome.status, "unknown")
        self.assertEqual(outcome.effects_started, "unknown")
        adapter.cancel(receipt.action_id)
        self.assertTrue(stopped.is_set())

    def test_function_heartbeat_does_not_rewrite_transition_or_completion_time(self):
        now = [1000.0]
        release = threading.Event()
        self.addCleanup(release.set)
        def action(request):
            release.wait(2)
            return FunctionResult(True)
        def state():
            return ExecutorState(Holding("empty", ts=now[0], source="test"), now[0],
                                 ready=True, motion="stopped", phase="surveying")
        adapter = FunctionActions({"look_around": action}, state, release.set,
                                  max_box_size=(.3, .3, .3), max_height=1, clock=lambda: now[0])
        receipt = adapter.submit(self.request(now=now[0]))
        first = adapter.status(receipt.action_id)
        now[0] += 5
        running = adapter.status(receipt.action_id)
        self.assertEqual(running.status, "running")
        self.assertEqual(running.ts, first.ts)
        self.assertEqual(running.observed_at, now[0])
        adapter.cancel(receipt.action_id)
        deadline = time.monotonic()+1
        while adapter.status(receipt.action_id).status == "running" and time.monotonic() < deadline:
            time.sleep(.001)
        terminal = adapter.status(receipt.action_id)
        self.assertEqual(terminal.status, "succeeded")
        now[0] += 5
        repeated = adapter.status(receipt.action_id)
        self.assertEqual(repeated.ts, terminal.ts)
        self.assertGreater(repeated.observed_at, terminal.observed_at)

    def test_cancel_acknowledgement_cannot_hide_a_running_function(self):
        release = threading.Event()
        self.addCleanup(release.set)
        def action(request):
            release.wait(2)
            return FunctionResult(True)
        def state():
            return ExecutorState(Holding("empty", ts=time.time(), source="test"), time.time(),
                                 ready=True, motion="stopped")
        adapter = FunctionActions({"look_around": action}, state, lambda: None,
                                  max_box_size=(.3, .3, .3), max_height=1)
        receipt = adapter.submit(self.request())
        acknowledgement = adapter.cancel(receipt.action_id)
        self.assertTrue(acknowledgement.acknowledged)
        self.assertFalse(acknowledgement.stopped)
        self.assertEqual(adapter.state().active_action, receipt.action_id)
        with self.assertRaises(RuntimeError):
            adapter.submit(self.request(request_id="other"))
        release.set()

    def test_changed_duplicate_and_expired_requests_are_not_executed(self):
        calls = []
        def action(request):
            calls.append(request)
            return FunctionResult(True)
        def state():
            return ExecutorState(Holding("empty", ts=time.time(), source="test"), time.time(),
                                 ready=True, motion="stopped")
        adapter = FunctionActions({"look_around": action}, state, lambda: None,
                                  max_box_size=(.3, .3, .3), max_height=1)
        request = self.request()
        receipt = adapter.submit(request)
        with self.assertRaisesRegex(ValueError, "different payload"):
            adapter.submit(replace(request, step=Step("look_around", "changed", search="materials")))
        self.assertEqual(adapter.submit(request), receipt)
        with self.assertRaisesRegex(ValueError, "expired"):
            adapter.submit(replace(self.request(request_id="expired"), expires_at=0.0))
        deadline = time.monotonic()+1
        while adapter.status(receipt.action_id).status == "running" and time.monotonic() < deadline:
            time.sleep(.001)
        self.assertEqual(calls, [request])

    def test_mock_providers_have_separate_versioned_capabilities(self):
        world = MockAgentWorld(3)
        actions, perception = world.actions.capabilities(), world.observations.capabilities()
        self.assertIs(type(actions), Capabilities)
        self.assertIs(type(perception), PerceptionCapabilities)
        self.assertEqual(actions.api_version, INTERFACE_VERSION)
        self.assertEqual(perception.api_version, INTERFACE_VERSION)
        self.assertTrue(perception.monitoring and perception.images)
        self.assertTrue(actions.emergency_stop and actions.idempotency)
        self.assertFalse(hasattr(world.observations, "submit"))
        self.assertFalse(hasattr(world.actions, "observe"))
        req = self.request().requirements
        self.assertEqual(preflight(world.actions, world.observations, req), actions)
        world.observations.capabilities = lambda: replace(perception, api_version=1)
        with self.assertRaisesRegex(CapabilityError, "protocol v2"):
            preflight(world.actions, world.observations, req)
        world.observations.capabilities = lambda: replace(perception, monitoring=False)
        with self.assertRaisesRegex(CapabilityError, "monitoring"):
            preflight(world.actions, world.observations, req)
        world.observations.capabilities = lambda: perception
        world.observations.monitor = None
        with self.assertRaisesRegex(CapabilityError, "monitor method"):
            preflight(world.actions, world.observations, req)
        self.assertFalse(world.requests)

    def test_site_cell_center_uses_schematic_dimensions_and_world_axes(self):
        site = MockAgentWorld()._site("floor-a")
        position = site.cell_center((1, 2, 3), (.2, .3, .4))
        for actual, expected in zip(position, (.8, -.8, .75)):
            self.assertAlmostEqual(actual, expected)

    def test_mock_observation_is_read_only_and_snapshots_do_not_mutate(self):
        world = MockAgentWorld(3)
        position, holding = world.base_position, world.holding
        first = world.observations.observe()
        second = world.observations.observe()
        self.assertNotEqual(first.revision, second.revision)
        self.assertLess(first.captured_at, second.captured_at)
        self.assertEqual(world.base_position, position)
        self.assertEqual(world.holding, holding)
        self.assertFalse(world.requests)
        world.positions[0] = (99.0, 99.0, 0.0)
        self.assertNotEqual(first.boxes[0].position, world.positions[0])

    def test_mock_idempotency_and_terminal_timestamps(self):
        world = MockAgentWorld()
        request = self.request(now=world.now)
        receipt = world.actions.submit(request)
        self.assertEqual(world.actions.submit(request), receipt)
        self.assertEqual(world.actions.lookup(request.request_id), receipt)
        with self.assertRaisesRegex(ValueError, "different payload"):
            world.actions.submit(replace(request, job_id="other-job"))
        world.sleep(world.action_duration+.001)
        terminal = world.actions.status(receipt.action_id)
        self.assertEqual(terminal.status, "succeeded")
        world.sleep(10)
        repeated = world.actions.status(receipt.action_id)
        self.assertEqual(terminal.ts, repeated.ts)
        self.assertGreater(repeated.observed_at, terminal.observed_at)
        self.assertEqual(world.actions.submit(request), receipt)
        self.assertEqual(len(world.requests), 1)

    def test_mock_cleanup_never_releases_a_load(self):
        world = MockAgentWorld(1)
        world.holding = Holding("holding", 0, world.now, "mock-gripper")
        before = world.holding
        world.close()
        self.assertEqual(world.holding, before)
        self.assertEqual(world.stop_calls, 0)
        self.assertFalse(world.requests)


class PerceptionTests(unittest.TestCase):
    def payload(self, now=1000.0):
        return {
            "schema_version": 2, "mock": False, "ts": now, "stale": False,
            "frame": "base_at_capture",
            "pose": {"x": 1.0, "y": 2.0, "yaw": 1.5707963267948966,
                     "ts": now, "source": "wheel-imu", "valid": True, "epoch": 3, "warning": ""},
            "build": {"origin": [1.0, 0.0, 0.0], "col": [1.0, 0.0, 0.0],
                      "row": [0.0, 1.0, 0.0], "marker": [.8, -.2, 0.0],
                      "ts": now-.1, "valid": True, "age": .1, "cells": [[1.0, 0.0, .15]]},
            "anchor_seen": True,
            "tracks": [
                {"id": 7, "world": [1.0, 3.0, .15], "pos": [1.0, 0.0, .15],
                 "classification": "protected", "size": .3, "last_seen": now,
                 "current": True, "pick_candidate": False},
                {"id": 8, "world": [2.0, 3.0, .15], "pos": [1.0, -1.0, .15],
                 "classification": "loose", "size": .3, "last_seen": now-10,
                 "current": False, "pick_candidate": False},
                {"id": 1000, "world": [1.0, 2.5, .2], "pos": [.5, 0.0, .2],
                 "classification": "unknown", "size": 0.0, "last_seen": now-.3,
                 "current": True, "pick_candidate": False, "position_kind": "visible_surface_centroid"}],
            "objects": [{"id": 1000, "track_id": "box-001", "tracker_session": "session-a",
                         "label": "cardboard_box", "bbox": [10.0, 20.0, 40.0, 60.0], "score": .8,
                         "position_base_m": None, "world_position_m": None, "last_seen": now-.3,
                         "pose_epoch": 3, "identity_status": "ambiguous", "current": True,
                         "position_kind": "visible_surface_centroid", "depth_status": "missing",
                         "pick_candidate": False, "grasp_pose": None}],
            "surface_cells": [{"world": [1.05, 2.05, .4], "pos": [.05, -.05, .4],
                               "z_min": .01, "z_max": .4, "age": .6}],
            "settings": {"box_size": .3, "cell": .3, "build_cols": 3, "build_rows": 3,
                         "resolution": .1, "radius": 2.0, "fresh_s": .8, "anchor_ttl": 15.0},
            "diagnostics": {"depth_yaw_deg": -90.0, "point_count": 1000, "marker_ids": [7, 49]},
            "streams": {"rect": {"ts": now, "age_s": 0.0, "fresh": True}},
            "telemetry": {"ages": {"wheel": .01, "imu": .02}},
            "detector": "YOLO-World", "detector_status": {"enabled": True, "state": "ready"},
            "warnings": ["odometry is not SLAM"],
            "map_semantics": "observed surfaces only; blank cells UNKNOWN, not free; not a navigation map"}

    def server(self, payload=None, frame_status=200, frame_ts=None, scan_status=200):
        state = {"payload": payload or self.payload(time.time()), "frame_status": frame_status,
                 "scan_status": scan_status, "frame_ts": frame_ts, "calls": [], "scans": 0,
                 "jpeg": b"\xff\xd8\xff\xe0offline-camera-test"}
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_GET(self):
                state["calls"].append(self.path)
                if self.path == "/scan":
                    state["scans"] += 1
                    value = state["payload"]
                    if callable(value):
                        value = value(state["scans"])
                    data, status = json.dumps(value).encode(), state["scan_status"]
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                else:
                    data, status = state["jpeg"], state["frame_status"]
                    self.send_response(status)
                    self.send_header("Content-Type", "image/jpeg")
                    ts = state["frame_ts"]
                    if ts is None:
                        value = state["payload"]
                        ts = (value(state["scans"]) if callable(value) else value)["ts"]
                    self.send_header("X-Frame-Timestamp", str(ts))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}", state

    def observe(self, provider):
        deadline = time.monotonic()+2
        while True:
            try:
                return provider.observe()
            except TimeoutError:
                if time.monotonic() >= deadline:
                    raise

    def test_world_model_preserves_memory_geometry_and_uncertainty(self):
        from agent_adapters import normalize_perception
        payload = self.payload()
        snapshot = normalize_perception(payload, 1000.2)
        self.assertEqual(snapshot.epoch, 3)
        self.assertEqual(snapshot.base_position, (1.0, 2.0, 0.0))
        self.assertEqual({b.id for b in snapshot.boxes}, {7, 8})
        self.assertTrue(all(b.eligible is None for b in snapshot.boxes))
        self.assertFalse(snapshot.boxes[1].current)
        self.assertIsNone(snapshot.holding)
        self.assertIsNone(snapshot.site_id)
        self.assertFalse(snapshot.occupancy_complete)
        world = snapshot.world_model
        self.assertEqual(world["tracks"][0]["classification"], "protected")
        self.assertEqual(world["objects"][0]["identity_status"], "ambiguous")
        self.assertIsNone(world["objects"][0]["world_position_m"])
        self.assertFalse(world["objects"][0]["pick_candidate"])
        self.assertEqual(world["objects"][0]["tracker_session"], "session-a")
        for actual, expected in zip(world["build_world"]["origin"], [1.0, 3.0, 0.0]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(world["build_world"]["col"], [0.0, 1.0, 0.0]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(world["surface_cells"][0]["last_seen"], 999.4)
        self.assertAlmostEqual(world["surface_cells"][0]["age"], .8)
        self.assertEqual(world["settings"]["resolution"], .1)
        self.assertEqual(world["diagnostics"]["point_count"], 1000)
        self.assertEqual(world["telemetry"]["ages"]["wheel"], .01)
        self.assertIn("UNKNOWN", world["map_semantics"])
        payload["tracks"][0]["world"][0] = 99
        world["objects"].clear()
        self.assertEqual(snapshot.world_model["tracks"][0]["world"][0], 1.0)
        self.assertEqual(len(snapshot.world_model["objects"]), 1)

    def test_cached_world_evidence_is_not_refreshed_or_promoted(self):
        from agent_adapters import normalize_perception
        payload = self.payload()
        for change in ({}, {"stale": True}, {"pose": dict(payload["pose"], valid=False)}):
            with self.subTest(change=change):
                snapshot = normalize_perception(dict(payload, **change), 1005.0)
                self.assertFalse(snapshot.valid)
                self.assertFalse(any(b.current for b in snapshot.boxes))
                self.assertEqual(snapshot.captured_at, 1000.0)
                self.assertEqual(snapshot.boxes[1].last_seen, 990.0)
                self.assertFalse(snapshot.world_model["build"]["valid"])
                self.assertFalse(any(t["current"] for t in snapshot.world_model["tracks"]))

    def test_pose_loss_preserves_recent_visual_detections_without_trusting_geometry(self):
        from agent_adapters import normalize_perception
        payload = self.payload()
        payload["pose"]["valid"] = False
        snapshot = normalize_perception(payload, 1000.2)
        world = snapshot.world_model
        self.assertFalse(snapshot.valid)
        self.assertFalse(world["pose_valid"])
        self.assertFalse(world["stale"])
        self.assertTrue(world["objects"][0]["observed_current"])
        self.assertFalse(world["objects"][0]["current"])
        self.assertEqual(world["objects"][0]["bbox"], payload["objects"][0]["bbox"])
        self.assertFalse(world["objects"][0]["pick_candidate"])

    def test_world_model_budget_reports_omitted_map_cells(self):
        from agent_adapters import normalize_perception
        from agent_types import MAX_WORLD_MODEL_BYTES
        payload = self.payload()
        payload["surface_cells"] = [dict(payload["surface_cells"][0], world=[i*.1, 2.05, .4])
                                    for i in range(3000)]
        snapshot = normalize_perception(payload, 1000.2)
        self.assertLessEqual(len(snapshot.world_model_json.encode()), MAX_WORLD_MODEL_BYTES)
        world = snapshot.world_model
        self.assertGreater(len(world["surface_cells"]), 0)
        self.assertEqual(world["coverage"]["surface_cells"]["total"], 3000)
        self.assertEqual(world["coverage"]["surface_cells"]["omitted"], 3000-len(world["surface_cells"]))
        self.assertIn("UNKNOWN", world["map_semantics"])
        self.assertEqual(world["surface_summary"]["max_height"], .4)

    def test_wire_schema_mock_and_malformed_pose_are_rejected(self):
        from agent_adapters import normalize_perception
        payload = self.payload()
        for change in ({"schema_version": 1}, {"mock": True}, {"mock": None},
                       {"frame": "unknown"}, {"ts": float("nan")},
                       {"pose": dict(payload["pose"], epoch=True)},
                       {"pose": dict(payload["pose"], x=float("inf"))}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                normalize_perception(dict(payload, **change), 1000.2)

    def test_http_provider_returns_real_images_and_world_model_without_hardware(self):
        from agent_adapters import PerceptionObservations
        url, state = self.server()
        with PerceptionObservations(url, interval=.01) as provider:
            snapshot = self.observe(provider)
            caps = provider.capabilities()
            self.assertTrue(caps.inventory and caps.images)
            self.assertFalse(caps.sites or caps.occupancy or caps.monitoring or caps.possession)
            self.assertEqual(len(snapshot.images), 1)
            image = snapshot.images[0]
            self.assertFalse(image.simulated)
            self.assertEqual(image.captured_at, state["payload"]["ts"])
            self.assertEqual((image.epoch, image.frame_id), (snapshot.epoch, snapshot.frame_id))
            self.assertEqual(base64.b64decode(image.data_url.split(",", 1)[1]), state["jpeg"])
            self.assertTrue(snapshot.world_model["detector_status"]["enabled"])
            with self.assertRaises(CapabilityError):
                provider.find_build_sites(None)
            with self.assertRaises(CapabilityError):
                provider.monitor(None, None)
        self.assertEqual(state["calls"][:3], ["/scan", "/frame?view=rect", "/scan"])
        self.assertNotIn("perception", sys.modules)

    def test_missing_stale_or_epoch_mismatched_images_are_not_fabricated(self):
        from agent_adapters import PerceptionObservations
        now = time.time()
        changed = self.payload(now)
        def payload(count):
            result = copy.deepcopy(changed)
            result["pose"]["epoch"] = count
            return result
        for options in ({"frame_status": 503}, {"frame_ts": now-10}, {"payload": payload}):
            with self.subTest(options=options):
                url, _ = self.server(**options)
                with PerceptionObservations(url, interval=.01) as provider:
                    snapshot = self.observe(provider)
                    self.assertFalse(snapshot.images)
                    self.assertTrue(any("image" in warning for warning in snapshot.warnings))

    def test_http_failure_is_not_a_fresh_cached_scan_and_recovers(self):
        from agent_adapters import PerceptionObservations
        url, state = self.server(scan_status=503)
        with PerceptionObservations(url, interval=.01) as provider:
            with self.assertRaises(RuntimeError):
                self.observe(provider)
            state["scan_status"] = 200
            deadline = time.monotonic()+2
            while True:
                try:
                    snapshot = self.observe(provider)
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.01)
            self.assertEqual(snapshot.captured_at, state["payload"]["ts"])

    def test_live_url_validation_happens_before_network_access(self):
        from agent_adapters import PerceptionObservations
        for url in ("file:///tmp/scan", "http://user:password@localhost:8007", "http://localhost:8007/scan",
                    "http://localhost:8007?target=other", "http://localhost:8007#frame", "http://"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                PerceptionObservations(url)

    def test_detector_epoch_changes_and_missing_anchor_stay_unknown(self):
        from agent_adapters import normalize_perception
        payload = self.payload()
        payload["build"] = None
        payload["objects"][0].update(pose_epoch=2, world_position_m=[1.0, 2.0, .2])
        snapshot = normalize_perception(payload, 1000.2)
        world = snapshot.world_model
        self.assertIsNone(world["build"])
        self.assertNotIn("build_world", world)
        self.assertEqual(world["site_feasibility"], "unknown")
        obj = world["objects"][0]
        self.assertEqual(obj["identity_status"], "pose_epoch_changed")
        self.assertIsNone(obj["world_position_m"])
        self.assertFalse(obj["current"])
        self.assertEqual(obj["bbox"], [10.0, 20.0, 40.0, 60.0])

    def test_agent_context_includes_the_world_model(self):
        from agent import Agent
        from agent_adapters import normalize_perception
        from contracts import Block, Structure
        world = MockAgentWorld(1)
        agent = Agent(world.actions, world.observations, clock=world.clock, sleep=world.sleep)
        agent.state = agent.manager.submit(Structure([Block(0, 0, 0)]))
        agent.capabilities = world.actions.capabilities()
        agent._refresh()
        live = normalize_perception(self.payload(), 1000.2)
        agent.state.snapshot = replace(agent.state.snapshot, world_model_json=live.world_model_json)
        context = agent._context((Step("observe"),))
        self.assertEqual(context["world_model"], live.world_model)
        self.assertNotIn("world_model_json", context)

    def test_world_awareness_does_not_bypass_full_agent_preflight(self):
        from agent import Agent
        from agent_adapters import PerceptionObservations
        from contracts import Block, Structure
        world = MockAgentWorld(1)
        provider = PerceptionObservations()
        with patch.object(provider, "start", side_effect=AssertionError("no sensing before preflight")):
            result = Agent(world.actions, provider, clock=world.clock, sleep=world.sleep).run(
                Structure([Block(0, 0, 0)]))
        self.assertFalse(result.success)
        for capability in ("sites", "occupancy", "monitoring"):
            self.assertIn(capability, result.reason)
        self.assertFalse(world.requests)

    def test_http_world_model_and_image_reach_the_existing_vision_backend(self):
        from agent_adapters import PerceptionObservations
        from agent_backend import OpenAIReasoner
        payload = self.payload()
        payload["surface_cells"] = [dict(payload["surface_cells"][0], world=[i*.1, 2.05, .4])
                                    for i in range(3000)]
        url, state = self.server(payload)
        with PerceptionObservations(url, timeout=.5, clock=lambda: 1000.2) as provider:
            snapshot = self.observe(provider)
        requests = []
        def respond(**request):
            requests.append(request)
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content=json.dumps(asdict(Step("observe"))), refusal=None))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=respond)))
        reasoner = OpenAIReasoner(client=client)
        context = {"world_model": snapshot.world_model, "images": [asdict(i) for i in snapshot.images],
                   "inventory": [asdict(b) for b in snapshot.boxes], "execution_enabled": False}
        self.assertEqual(reasoner.decide(context, (Step("observe"), Step("stop"))), Step("observe"))
        content = requests[0]["messages"][1]["content"]
        self.assertEqual([part["type"] for part in content], ["text", "image_url"])
        model_context = json.loads(content[0]["text"])
        self.assertEqual(model_context["world_model"]["surface_summary"]["count"], 3000)
        self.assertEqual(model_context["world_model"]["objects"][0]["depth_status"], "missing")
        self.assertFalse(model_context["images"][0]["simulated"])
        self.assertEqual(base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1]), state["jpeg"])
        self.assertNotIn("data:image/", content[0]["text"])

    def test_world_model_snapshot_budget_and_json_are_enforced(self):
        from agent_types import MAX_WORLD_MODEL_BYTES, ObservationSnapshot
        for value in ("[]", "{bad", '{"height":NaN}', json.dumps({"large": "x"*MAX_WORLD_MODEL_BYTES})):
            with self.subTest(value=value[:24]), self.assertRaises(ValueError):
                ObservationSnapshot("test", 1000.0, 1000.0, 0, world_model_json=value)

    def test_observation_reads_are_bounded_even_with_a_slow_source(self):
        from agent_adapters import PerceptionObservations
        release = threading.Event()
        class Session:
            def poll(self):
                release.wait(2)
            def close(self):
                pass
        provider = PerceptionObservations(timeout=.03)
        provider.factory = Session
        try:
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                provider.observe()
            self.assertLess(time.monotonic()-started, .25)
        finally:
            release.set()
            provider.close()


if __name__ == "__main__":
    unittest.main()
