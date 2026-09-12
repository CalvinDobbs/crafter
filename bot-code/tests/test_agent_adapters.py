import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import CapabilityError, preflight
from agent_adapters import (FunctionActions, FunctionResult, ObservationWorker,
                            normalize_scan, normalize_skill_result, load_providers)
from agent_types import (ActionRequest, AgentConfig, BuildRequirements, ExecutorState, Holding, Step)
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
            preflight(world, worker, req)
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
        request = ActionRequest("one", "job", Step("look_around", search="materials"),
                                "snapshot", 0, "world", time.time())
        receipt = adapter.submit(request)
        self.assertEqual(adapter.submit(request), receipt)
        deadline = time.monotonic()+1
        while adapter.status(receipt.action_id).status == "running" and time.monotonic() < deadline:
            time.sleep(.001)
        self.assertEqual(adapter.status(receipt.action_id).status, "succeeded")
        self.assertEqual(len(calls), 1)
        self.assertIsNone(adapter.state().active_action)

    def test_plain_function_exception_is_unknown_not_safe_failure(self):
        def action(step):
            raise TimeoutError("may have moved")
        def state():
            return ExecutorState(Holding("empty", ts=time.time(), source="test"), time.time(),
                                 ready=True, motion="stopped")
        adapter = FunctionActions({"pickup": action}, state, lambda: None,
                                  max_box_size=(.3, .3, .3), max_height=1.0)
        request = ActionRequest("two", "job", Step("pickup", box_id=0), "snapshot", 0, "world", time.time())
        receipt = adapter.submit(request)
        deadline = time.monotonic()+1
        while adapter.status(receipt.action_id).status == "running" and time.monotonic() < deadline:
            time.sleep(.001)
        outcome = adapter.status(receipt.action_id)
        self.assertEqual(outcome.status, "unknown")
        self.assertEqual(outcome.effects_started, "unknown")


if __name__ == "__main__":
    unittest.main()
