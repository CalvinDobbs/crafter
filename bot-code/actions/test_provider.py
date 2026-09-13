import importlib.util
import io
import sys
import threading
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # agent_adapters, agent_types


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    bbos = ModuleType("bbos")
    for attr in ("Reader", "Writer", "Type", "Config"):
        setattr(bbos, attr, Mock(side_effect=AssertionError("Hardware access in offline test")))
    with patch.dict(sys.modules, {"bbos": bbos, name: module}):
        spec.loader.exec_module(module)
    return module


armctl = _load("armctl", "armctl.py")
sys.modules.setdefault("armctl", armctl)
provider = _load("provider_under_test", "provider.py")


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def time(self):
        return 1_700_000_000.0 + self.now

    def sleep(self, seconds):
        self.now += seconds
        if self.now > 120:
            raise AssertionError("Motion did not finish within virtual time limit")


class FakeArm(armctl.Arm):
    def __init__(self, side):
        self.side = side
        self.sign = 1 if side == "left" else -1
        self.dof = 8
        self.lo = np.full(self.dof, -4.0)
        self.hi = np.full(self.dof, 4.0)
        self.top, self.bottom = 0.0, 2.0
        self.elbow_90 = self.sign * 0.25
        self.grip_open, self.grip_closed = self.sign * 0.3, 0.0
        self.cmd = np.zeros(self.dof)
        self.cmd[armctl.ELBOW] = self.elbow_90 + self.sign * 0.03   # cradled, as pickup leaves it
        self.cmd[armctl.J0] = 0.2
        self.cfg = SimpleNamespace(q2urdf=lambda q: q, ik=SimpleNamespace(fk=self.fk),
                                   wheel_radius=0.0465)
        self.published = []
        self.torque = []
        self.closed = False

    def fk(self, q):
        return np.array([0.4, self.sign * 0.5 + q[armctl.SWING], 0.2]), None

    def live(self, cancel=None, timeout=2.0):
        return self.cmd.copy()

    def publish(self):
        self.published.append(self.cmd.copy())

    def set_torque(self, on):
        self.torque.append(on)


class FakeRig(armctl.Rig):
    def __init__(self, sides=("left", "right")):
        self.log = lambda *a: None
        self.arms = [FakeArm(s) for s in sides]
        self.by_side = {a.side: a for a in self.arms}
        self.twists = []
        self._twist = np.zeros(2)
        self._twist_lock = threading.Lock()
        self._running = threading.Event()
        self._threads = []
        self.closed = False

    def set_twist(self, v, w):
        self.twists.append((v, w))
        with self._twist_lock:
            self._twist[:] = (v, w)


class PlaceTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        for patcher in (patch.object(armctl, "time", self.clock),
                        patch.object(provider, "time", self.clock),
                        patch("sys.stdout", new_callable=io.StringIO)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.rig = FakeRig()
        self.executor = provider.Executor(self.rig, log=lambda *a: None)
        self.phases = []
        real_phase = self.executor.phase
        self.executor.phase = lambda name: (self.phases.append(name), real_phase(name))[1]

    def place(self, z_drop=0.10):
        return provider.place(self.executor, {"z_drop": z_drop})

    def test_phase_order_and_terminal_phase(self):
        terminal = self.place()
        self.assertEqual(self.phases, ["lowering", "releasing", "retreating"])
        # agent.py only accepts released / retreated / completed as a successful placement
        self.assertEqual(terminal, "retreated")

    def test_torque_is_never_disabled_during_place(self):
        self.place()
        for arm in self.rig.arms:
            self.assertEqual(arm.torque, [], "place must not touch torque; that drops the box")
            self.assertFalse(arm.closed, "place must not close writers; that drops the box")

    def test_box_descends_then_uncradles_then_opens_then_releases(self):
        before = [a.cmd.copy() for a in self.rig.arms]
        self.place(z_drop=0.10)
        for arm, start in zip(self.rig.arms, before):
            down = np.sign(arm.top - arm.bottom)
            # net J0 = descend by z_drop, then retreat back up
            self.assertAlmostEqual(arm.cmd[armctl.J0],
                                   start[armctl.J0] - 0.10 / 0.0465 * down + armctl.RANGE_MARGIN * 0
                                   + provider.RETREAT_TURNS * down, places=5)
            self.assertAlmostEqual(arm.cmd[armctl.ELBOW], arm.elbow_90, places=6)
            self.assertAlmostEqual(arm.cmd[armctl.GRIPPER], arm.grip_open, places=6)

    def test_arms_swing_outward_to_release(self):
        before = [a.cmd[armctl.SWING] for a in self.rig.arms]
        self.place()
        for arm, start in zip(self.rig.arms, before):
            outward = armctl.spread_direction(arm)[armctl.SWING]
            self.assertGreater((arm.cmd[armctl.SWING] - start) * outward, 0,
                               "each arm must swing away from the centreline to free the box")

    def test_release_travel_is_clamped_into_the_calibrated_range(self):
        # A swing range narrower than RELEASE_TRAVEL: the outward swing must stop at the
        # calibrated edge less the margin, not run to the full release travel.
        for arm in self.rig.arms:
            arm.lo[armctl.SWING], arm.hi[armctl.SWING] = -0.10, 0.10
        self.assertGreater(provider.RELEASE_TRAVEL, 0.10 - armctl.RANGE_MARGIN)
        self.place()
        for arm in self.rig.arms:
            self.assertLessEqual(arm.cmd[armctl.SWING], arm.hi[armctl.SWING] - armctl.RANGE_MARGIN + 1e-9)
            self.assertGreaterEqual(arm.cmd[armctl.SWING], arm.lo[armctl.SWING] + armctl.RANGE_MARGIN - 1e-9)

    def test_cancel_mid_place_raises_and_leaves_torque_on(self):
        self.executor.cancel.set()
        with self.assertRaises(armctl.Cancelled):
            self.place()
        for arm in self.rig.arms:
            self.assertEqual(arm.torque, [])
            self.assertFalse(arm.closed)


class LookAroundTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        for patcher in (patch.object(armctl, "time", self.clock),
                        patch.object(provider, "time", self.clock)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.rig = FakeRig()
        self.executor = provider.Executor(self.rig, log=lambda *a: None)
        self.phases = []
        real = self.executor.phase
        self.executor.phase = lambda n: (self.phases.append(n), real(n))[1]

    def test_phase_order(self):
        terminal = provider.look_around(self.executor, {"stations": 4})
        self.assertEqual(self.phases, ["surveying", "settling"])
        self.assertEqual(terminal, "completed")

    def test_rotates_once_per_station_and_stops_between(self):
        provider.look_around(self.executor, {"stations": 4})
        turning = [t for t in self.rig.twists if t != (0.0, 0.0)]
        self.assertEqual(len(turning), 4, "one twist command per station")
        for v, w in turning:
            self.assertEqual(v, 0.0, "a survey turns in place; it must not translate")
            self.assertAlmostEqual(abs(w), provider.SURVEY_YAW_RATE)

    def test_base_is_zeroed_on_the_way_out(self):
        provider.look_around(self.executor, {"stations": 4})
        self.assertEqual(self.rig.twists[-1], (0.0, 0.0))

    def test_cancel_mid_survey_still_zeroes_the_base(self):
        real_dwell = armctl.dwell
        calls = []

        def dwell(secs, cancel=None):
            calls.append(secs)
            if len(calls) == 3:
                cancel.set()
            return real_dwell(secs, cancel)

        with patch.object(armctl, "dwell", dwell):
            with self.assertRaises(armctl.Cancelled):
                provider.look_around(self.executor, {"stations": 4})
        self.assertEqual(self.rig.twists[-1], (0.0, 0.0),
                         "a cancelled survey must not leave the base spinning")

    def test_dwell_outlasts_perception_freshness(self):
        # perception rejects frames older than 0.8 s, so a shorter dwell buys no usable observation
        self.assertGreater(provider.SURVEY_SETTLE_S, 0.8)


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        for patcher in (patch.object(armctl, "time", self.clock),
                        patch.object(provider, "time", self.clock)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.rig = FakeRig()
        self.executor = provider.Executor(self.rig, log=lambda *a: None)

    def test_stop_zeroes_twist_and_keeps_torque(self):
        self.executor.stop()
        self.assertIn((0.0, 0.0), self.rig.twists)
        self.assertTrue(self.executor.cancel.is_set())
        for arm in self.rig.arms:
            self.assertEqual(arm.torque, [], "stop must be load-preserving")

    def test_holding_is_unknown_without_an_evidence_source(self):
        holding = self.executor.holding()
        self.assertEqual(holding.status, "unknown")
        self.assertEqual(holding.source, "no-possession-sensor")

    def test_read_state_reports_motion_and_phase(self):
        self.assertEqual(self.executor.read_state().motion, "stopped")
        self.executor.begin()
        self.executor.phase("lowering")
        state = self.executor.read_state()
        self.assertEqual((state.motion, state.phase, state.ready), ("running", "lowering", False))
        self.executor.end("retreated")
        self.assertEqual(self.executor.read_state().motion, "stopped")

    def test_close_refuses_to_drop_a_load(self):
        with self.assertRaises(RuntimeError):
            armctl.Rig.close(self.rig)


class WrapTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        for patcher in (patch.object(armctl, "time", self.clock),
                        patch.object(provider, "time", self.clock)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.rig = FakeRig()
        self.executor = provider.Executor(self.rig, log=lambda *a: None)

    def test_unresolvable_geometry_reports_no_effects_started(self):
        def resolve(request):
            raise ValueError("stale epoch")
        call = provider._wrap(self.executor, lambda e, a: "retreated", resolve)
        result = call(object())
        self.assertFalse(result.success)
        self.assertEqual(result.effects_started, "no", "a rejected request must not claim effects")
        self.assertEqual(result.phase, "rejected")

    def test_cancelled_routine_is_a_recoverable_blocked_motion(self):
        def routine(executor, args):
            raise armctl.Cancelled()
        call = provider._wrap(self.executor, routine, lambda r: {})
        result = call(object())
        self.assertFalse(result.success)
        # agent.py only allows recovery on blocked_motion for a held box
        self.assertEqual(result.error_code, "blocked_motion")
        self.assertEqual(result.effects_started, "unknown")

    def test_successful_routine_reports_its_terminal_phase(self):
        call = provider._wrap(self.executor, lambda e, a: "retreated", lambda r: {})
        result = call(object())
        self.assertTrue(result.success)
        self.assertEqual((result.phase, result.effects_started), ("retreated", "yes"))


if __name__ == "__main__":
    unittest.main()
