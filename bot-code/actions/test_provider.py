import importlib.util
import io
import math
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
        self.lo[armctl.SWING], self.hi[armctl.SWING] = -0.3, 0.3
        self.lo[armctl.WRIST_YAW], self.hi[armctl.WRIST_YAW] = -0.3, 0.3
        self.top, self.bottom = 0.0, 2.0
        self.elbow_90 = self.sign * 0.25
        self.grip_open, self.grip_closed = self.sign * 0.3, 0.0
        self.cmd = np.zeros(self.dof)
        self.cmd[armctl.ELBOW] = self.elbow_90 + self.sign * 0.03   # cradled, as pickup leaves it
        self.cmd[armctl.J0] = 0.2
        home = np.zeros(self.dof)
        home[armctl.ELBOW] = self.elbow_90
        self.cfg = SimpleNamespace(q2urdf=lambda q: q, ik=SimpleNamespace(fk=self.fk),
                                   wheel_radius=0.0465, home=home)
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
        self.cfg_drive = SimpleNamespace(max_linear_vel=0.3, max_angular_vel=0.9)
        self._r_imu = None
        self.yaw = 0.0          # simulated body yaw, advanced by whatever is commanded

    def measured_yaw(self):
        return self.yaw

    def set_twist(self, v, w):
        self.twists.append((v, w))                 # as requested, in the +CCW convention
        armctl.Rig.set_twist(self, v, w)           # real clamping and hardware inversion
        # the base physically turns opposite the commanded sign, which is the inversion being
        # corrected: applying it twice lands back on the direction the caller asked for
        self.yaw += armctl.YAW_COMMAND_SIGN * float(self._twist[1]) / armctl.MOTION_RATE_HZ


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

    def test_a_survey_covers_a_full_turn(self):
        # the first live run swept 137 deg of an intended 360 on open-loop timing
        start = self.rig.yaw
        provider.look_around(self.executor, {"stations": 4})
        swept = abs(self.rig.yaw - start)
        self.assertAlmostEqual(swept, 2 * math.pi, delta=0.2,
                               msg=f"survey swept {math.degrees(swept):.0f} deg, expected 360")

    def test_a_survey_turns_in_place_and_never_translates(self):
        provider.look_around(self.executor, {"stations": 4})
        for v, _ in self.rig.twists:
            self.assertEqual(v, 0.0, "a survey turns in place; it must not translate")

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


class PickupTests(unittest.TestCase):
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
        self.order = []
        real_ramp = armctl.ramp_joint
        patcher = patch.object(armctl, "ramp_joint", lambda arms, joint, *a, **k: (
            self.order.append(joint), real_ramp(arms, joint, *a, **k))[1])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_phase_order_and_terminal_phase(self):
        terminal = provider.pickup(self.executor, {})
        self.assertEqual(self.phases, ["grasping", "lifting"])
        self.assertEqual(terminal, "lifted")

    def test_spreads_before_it_lowers(self):
        provider.pickup(self.executor, {})
        # the forearms must already straddle the box before the lift descends onto it
        first_swing = self.order.index(armctl.SWING)
        j0_moves = [i for i, j in enumerate(self.order) if j == armctl.J0]
        self.assertLess(first_swing, j0_moves[-1], "spread must precede the descent")

    def test_ends_cradled_closed_and_lifted(self):
        provider.pickup(self.executor, {})
        for arm in self.rig.arms:
            self.assertAlmostEqual(arm.cmd[armctl.J0], arm.top, places=6)
            self.assertAlmostEqual(arm.cmd[armctl.ELBOW], arm.cradle(provider.CRADLE_TILT), places=6)
            self.assertAlmostEqual(arm.cmd[armctl.GRIPPER], arm.grip_closed, places=6)

    def test_torque_is_never_disabled_and_no_writer_is_closed(self):
        provider.pickup(self.executor, {})
        for arm in self.rig.arms:
            self.assertEqual(arm.torque, [])
            self.assertFalse(arm.closed)

    def test_cancel_leaves_the_load_held(self):
        self.executor.cancel.set()
        with self.assertRaises(armctl.Cancelled):
            provider.pickup(self.executor, {})
        for arm in self.rig.arms:
            self.assertEqual(arm.torque, [])

    def test_spread_stays_inside_the_calibrated_range(self):
        provider.pickup(self.executor, {})
        for arm in self.rig.arms:
            self.assertLessEqual(arm.cmd[armctl.SWING], arm.hi[armctl.SWING])
            self.assertGreaterEqual(arm.cmd[armctl.SWING], arm.lo[armctl.SWING])


class DriveTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        for patcher in (patch.object(armctl, "time", self.clock),
                        patch.object(provider, "time", self.clock)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.rig = FakeRig()
        self.executor = provider.Executor(self.rig, log=lambda *a: None)

    def moving_target(self, start=2.0, bearing=0.0):
        """A target in the base frame that responds to the robot's own motion.

        Driving forward closes the range; turning reduces the bearing. Without the second half
        the robot would turn forever and the stuck detector would fire.
        """
        state = {"range": start, "bearing": bearing}
        dt = 1.0 / armctl.MOTION_RATE_HZ

        def target_fn():
            v = float(self.rig._twist[0])
            # _twist carries the hardware inversion; the base physically turns the other way
            w = armctl.YAW_COMMAND_SIGN * float(self.rig._twist[1])
            state["range"] = max(0.0, state["range"] - v * dt)
            state["bearing"] -= w * dt
            return (state["range"] * math.cos(state["bearing"]),
                    state["range"] * math.sin(state["bearing"]), 0.0)
        return target_fn

    def test_turns_before_it_creeps(self):
        seen = []
        real = self.rig.set_twist

        def record(v, w):
            seen.append((v, w))
            real(v, w)
        self.rig.set_twist = record
        armctl.drive_to_standoff(self.rig, self.moving_target(bearing=0.6), 0.45,
                                 log=lambda *a: None)
        turning = [i for i, (v, w) in enumerate(seen) if w != 0.0]
        creeping = [i for i, (v, w) in enumerate(seen) if v != 0.0]
        self.assertTrue(turning and creeping)
        self.assertLess(max(turning), min(creeping), "must finish turning before creeping")
        for v, w in seen:
            self.assertFalse(v != 0.0 and w != 0.0, "turn and creep are separate phases")

    def test_stops_at_the_standoff_not_at_the_target(self):
        final = armctl.drive_to_standoff(self.rig, self.moving_target(), 0.45, log=lambda *a: None)
        self.assertLessEqual(abs(final - 0.45), armctl.RANGE_TOL + 1e-6)

    def test_respects_the_speed_limits(self):
        seen = []
        self.rig.set_twist = lambda v, w: (seen.append((v, w)), armctl.Rig.set_twist(self.rig, v, w))[1]
        armctl.drive_to_standoff(self.rig, self.moving_target(bearing=0.6), 0.45,
                                 log=lambda *a: None, speed=0.05, omega=0.10)
        for v, w in seen:
            self.assertLessEqual(abs(v), 0.05 + 1e-9)
            self.assertLessEqual(abs(w), 0.10 + 1e-9)

    def test_base_is_zeroed_on_every_exit_path(self):
        armctl.drive_to_standoff(self.rig, self.moving_target(), 0.45, log=lambda *a: None)
        self.assertEqual(self.rig.twists[-1], (0.0, 0.0))

    def test_cancel_stops_the_base(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(armctl.Cancelled):
            armctl.drive_to_standoff(self.rig, self.moving_target(), 0.45, cancel=cancel,
                                     log=lambda *a: None)
        self.assertEqual(self.rig.twists[-1], (0.0, 0.0))

    def test_a_target_that_stops_closing_is_stuck_not_an_infinite_grind(self):
        with self.assertRaises(armctl.Stuck):
            armctl.drive_to_standoff(self.rig, lambda: (2.0, 0.0, 0.0), 0.45, log=lambda *a: None)
        self.assertEqual(self.rig.twists[-1], (0.0, 0.0))

    def test_a_target_that_cannot_be_resolved_ends_the_drive_and_stops(self):
        def gone():
            raise RuntimeError("box is not in the current snapshot")
        with self.assertRaises(RuntimeError):
            armctl.drive_to_standoff(self.rig, gone, 0.45, log=lambda *a: None)
        self.assertEqual(self.rig.twists[-1], (0.0, 0.0),
                         "losing the target must stop the base, not continue blind")

    def test_move_to_build_is_slower_than_an_empty_approach(self):
        self.assertLess(provider.CARRY_SPEED, armctl.DRIVE_SPEED)
        self.assertLess(provider.CARRY_OMEGA, armctl.DRIVE_OMEGA)

    def test_move_to_build_does_not_touch_the_arms(self):
        before = [a.cmd.copy() for a in self.rig.arms]
        provider.move_to_build(self.executor, {"target_fn": self.moving_target(), "loaded": True})
        for arm, start in zip(self.rig.arms, before):
            np.testing.assert_allclose(arm.cmd, start)
            self.assertEqual(arm.torque, [], "a carry must not disturb the squeeze holding the box")


class YawTests(unittest.TestCase):
    """The base turns clockwise for a positive command; that inversion is corrected once."""

    def test_a_positive_command_reaches_the_hardware_positive(self):
        rig = FakeRig()
        armctl.Rig.set_twist(rig, 0.0, 0.4)          # ask for +CCW
        # measured against the raw IMU: +twist[1] turns the base CCW, so no correction is applied
        self.assertAlmostEqual(float(rig._twist[1]), 0.4)

    def test_the_measured_sign_matches_the_documented_convention(self):
        # perception's base_yaw uses the opposite convention; it is not what drive.ctrl speaks
        self.assertEqual(armctl.YAW_COMMAND_SIGN, 1.0)

    def test_turn_by_stops_on_measured_angle_not_elapsed_time(self):
        rig = FakeRig()
        turned = armctl.Rig.turn_by(rig, 0.5, 1.0, log=lambda *a: None)
        self.assertAlmostEqual(turned, 0.5, delta=armctl.YAW_TOL + 0.05)
        self.assertAlmostEqual(rig.yaw, 0.5, delta=armctl.YAW_TOL + 0.05)
        self.assertEqual(rig.twists[-1], (0.0, 0.0), "a finished turn leaves the base stopped")

    def test_turn_by_goes_the_way_it_was_asked_to(self):
        rig = FakeRig()
        armctl.Rig.turn_by(rig, -0.4, 1.0, log=lambda *a: None)
        self.assertLess(rig.yaw, 0.0, "a negative target must turn the base negative")

    def test_turn_by_accumulates_across_the_pi_seam(self):
        rig = FakeRig()
        rig.yaw = math.pi - 0.05          # a short turn from here wraps to -pi
        turned = armctl.Rig.turn_by(rig, 0.2, 1.0, log=lambda *a: None)
        self.assertAlmostEqual(turned, 0.2, delta=armctl.YAW_TOL + 0.05)

    def test_no_imu_falls_back_to_open_loop_and_says_so(self):
        rig = FakeRig()
        rig.measured_yaw = lambda: None
        said = []
        self.assertIsNone(armctl.Rig.turn_by(rig, 0.3, 1.0, log=said.append))
        self.assertTrue(any("open-loop" in m for m in said))
        self.assertEqual(rig.twists[-1], (0.0, 0.0))


class OwnershipTests(unittest.TestCase):
    """The rig must refuse to become the motion owner while somebody else already is."""

    def test_no_owners_means_free(self):
        with patch.object(armctl, "topic_owners", lambda *a, **k: {}):
            armctl.Rig.require_free()          # must not raise

    def test_a_live_owner_is_refused_by_name(self):
        owners = {"arm_left.ctrl": [(11770, "python3 pickup.py")]}
        with patch.object(armctl, "topic_owners", lambda *a, **k: owners):
            with self.assertRaises(armctl.HardwareBusy) as caught:
                armctl.Rig.require_free()
        message = str(caught.exception)
        self.assertIn("11770", message)
        self.assertIn("pickup.py", message)
        self.assertIn("arm_left.ctrl", message)
        # the message must steer away from the dangerous fix, not just report the clash
        self.assertIn("Do NOT kill the owner", message)

    def test_the_check_precedes_opening_any_writer(self):
        owners = {"drive.ctrl": [(42, "teleop.py")]}
        opened = []
        with patch.object(armctl, "topic_owners", lambda *a, **k: owners),              patch.object(armctl, "Arm", lambda side: opened.append(side)):
            with self.assertRaises(armctl.HardwareBusy):
                armctl.Rig()
        self.assertEqual(opened, [], "a busy robot must be detected before any writer is claimed")

    def test_topic_owners_is_read_only_and_portable(self):
        # returns {} rather than exploding where /proc does not exist
        self.assertIsInstance(armctl.topic_owners(("drive.ctrl",)), dict)


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
