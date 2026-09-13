import sys
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
