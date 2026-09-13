import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


spec = importlib.util.spec_from_file_location("pickup_under_test", Path(__file__).with_name("pickup.py"))
pickup = importlib.util.module_from_spec(spec)
bbos = ModuleType("bbos")
for name in ("Reader", "Writer", "Type", "Config"):
    setattr(bbos, name, Mock(side_effect=AssertionError("Hardware access in offline test")))
with patch.dict(sys.modules, {"bbos": bbos}), patch("signal.signal"):
    spec.loader.exec_module(pickup)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        if self.now > 60:
            raise AssertionError("Motion did not finish within virtual time limit")


class FakeArm(pickup.Arm):
    def __init__(self, side):
        self.side = side
        self.sign = 1 if side == "left" else -1
        self.dof = 8
        self.lo = np.full(self.dof, -1.0)
        self.hi = np.full(self.dof, 1.0)
        self.lo[pickup.SWING], self.hi[pickup.SWING] = -0.2, 0.2
        self.top, self.bottom = 0.0, 2.0
        self.elbow_90 = self.sign * 0.25
        self.grip_open, self.grip_closed = self.sign * 0.3, 0.0
        self.cmd = np.zeros(self.dof)
        self.contact_position = 0.05 if side == "left" else -0.02
        home = np.zeros(self.dof)
        home[pickup.ELBOW] = self.elbow_90
        self.cfg = SimpleNamespace(home=home, q2urdf=lambda q: q, ik=SimpleNamespace(fk=self.fk))
        self.commands = []
        self.torque = []
        self.closed = False

    def fk(self, q):
        yaw, pitch = q[pickup.WRIST_YAW], q[pickup.WRIST_PITCH]
        return np.array([0.4, self.sign * 0.5 + q[pickup.SWING] + 0.5 * yaw + 0.1 * pitch,
                         0.2 + 0.25 * yaw + pitch]), None

    def live(self):
        pos = self.cmd.copy()
        if self.contact_position is not None and pos[pickup.J0] > 0.5:
            pos[pickup.SWING] = self.sign * max(self.sign * pos[pickup.SWING],
                                               self.sign * self.contact_position)
        return pos

    def write_cmd(self, pos):
        self.cmd = np.asarray(pos).copy()
        self.commands.append(self.cmd.copy())

    def set_torque(self, on):
        self.torque.append(on)

    def close(self):
        self.closed = True


class PickupTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        for patcher in (patch.object(pickup, "time", self.clock),
                        patch.object(pickup, "_stop", False),
                        patch("sys.stdout", new_callable=io.StringIO)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_main(self, *args, hold_seconds=0.02, settle_effect=None):
        arms, holds, ramps = [], [], []
        hold_args = [] if hold_seconds is None else ["--hold", str(hold_seconds)]

        def make_arm(side):
            arm = FakeArm(side)
            arms.append(arm)
            return arm

        def ramp(selected, joint, targets, speed, **kwargs):
            ramps.append(joint)
            for arm, target in zip(selected, targets):
                pos = arm.cmd.copy()
                pos[joint] = target
                arm.write_cmd(pos)

        def hold(selected, seconds):
            holds.append((seconds, [arm.cmd.copy() for arm in selected]))

        with patch.object(pickup, "Arm", side_effect=make_arm), \
                patch.object(pickup, "ramp_joint", side_effect=ramp), \
                patch.object(pickup, "settle_joint", side_effect=settle_effect), \
                patch.object(pickup, "hold", side_effect=hold), \
                patch.object(sys, "argv", ["pickup.py", *hold_args, *args]):
            pickup.main()
        return arms, holds, ramps

    def test_default_grasps_inward_after_lowering_and_holds_without_lifting(self):
        arms, holds, ramps = self.run_main()
        self.assertEqual(ramps.count(pickup.J0), 2)
        self.assertEqual(holds[-1][0], 0.02)
        for arm, final in zip(arms, holds[-1][1]):
            self.assertAlmostEqual(final[pickup.SWING],
                                   arm.contact_position - arm.sign * pickup.PINCH_SQUEEZE)
            self.assertAlmostEqual(final[pickup.J0], 1.0)
            self.assertAlmostEqual(final[pickup.ELBOW], arm.elbow_90 - arm.sign * 20 / 360)
            spread = arm.sign * 0.18
            low_commands = [pos for pos in arm.commands if pos[pickup.J0] == 1.0]
            self.assertAlmostEqual(low_commands[0][pickup.SWING], spread)
            swings = [arm.sign * pos[pickup.SWING] for pos in low_commands]
            self.assertTrue(all(a >= b - 1e-9 for a, b in zip(swings, swings[1:])))
            self.assertTrue(arm.closed)
            self.assertEqual(arm.torque, [False, True, False])

    def test_elbows_extend_twenty_degrees_before_descent_and_hold_for_grasp(self):
        for args in ((), ("--lower-only",), ("--hook", "0"),
                     ("--arm", "left"), ("--arm", "right")):
            with self.subTest(args=args):
                arms, _, ramps = self.run_main(*args)
                self.assertEqual(ramps.count(pickup.ELBOW), 2)
                for arm in arms:
                    target = arm.elbow_90 - arm.sign * 20 / 360
                    extension = next(i for i, pos in enumerate(arm.commands)
                                     if abs(pos[pickup.ELBOW] - target) < 1e-9)
                    before, after = arm.commands[extension - 1:extension + 1]
                    self.assertAlmostEqual(after[pickup.J0], arm.top)
                    self.assertAlmostEqual(after[pickup.SWING], arm.sign * 0.18)
                    held = [j for j in range(arm.dof) if j != pickup.ELBOW]
                    np.testing.assert_array_equal(after[held], before[held])
                    for pos in arm.commands[extension:]:
                        self.assertAlmostEqual(pos[pickup.ELBOW], target)
                    self.assertAlmostEqual(arm.cmd[pickup.J0], 1.0)

    def test_elbow_extension_is_tunable_and_zero_preserves_home_bend(self):
        for extension in (0.0, 0.025):
            with self.subTest(extension=extension):
                arms, _, ramps = self.run_main("--elbow-extension", str(extension))
                self.assertEqual(ramps.count(pickup.ELBOW), 1 + (extension > 0))
                for arm in arms:
                    self.assertAlmostEqual(arm.cmd[pickup.ELBOW], arm.elbow_90 - arm.sign * extension)

    def test_unreachable_elbow_extension_aborts_before_torque_enable(self):
        initialize = FakeArm.__init__
        created = []

        def limited_elbow(arm, side):
            initialize(arm, side)
            arm.lo[pickup.ELBOW], arm.hi[pickup.ELBOW] = sorted([arm.sign * 0.22, arm.sign * 0.5])
            created.append(arm)

        with patch.object(FakeArm, "__init__", limited_elbow), self.assertRaisesRegex(ValueError, "elbow"):
            self.run_main()
        for arm in created:
            self.assertNotIn(True, arm.torque)
            self.assertEqual(arm.commands, [])
            self.assertTrue(arm.closed)

    def test_elbow_extension_failure_or_stop_prevents_descent_and_pinch(self):
        for stop in (True, False):
            created = []
            initialize = FakeArm.__init__

            def make_arm(arm, side):
                initialize(arm, side)
                created.append(arm)

            def settle(arms, joint, **kwargs):
                if joint == pickup.ELBOW and abs(arms[0].cmd[joint] - arms[0].elbow_90) > 1e-9:
                    self.assertTrue(kwargs.get("require_arrival"))
                    if stop:
                        pickup._sigint()
                    else:
                        raise RuntimeError("Elbow extension stalled")

            with self.subTest(stop=stop), patch.object(FakeArm, "__init__", make_arm):
                pickup._stop = False
                if stop:
                    self.run_main(settle_effect=settle)
                    self.assertTrue(pickup._stop)
                else:
                    with self.assertRaisesRegex(RuntimeError, "extension stalled"):
                        self.run_main(settle_effect=settle)
            for arm in created:
                self.assertTrue(all(pos[pickup.J0] == arm.top for pos in arm.commands))
                self.assertEqual(arm.torque, [False, True, False])
                self.assertTrue(arm.closed)

    def test_extension_cannot_pass_straight_or_calibrated_stop_margin(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                arm = FakeArm(side)
                with self.assertRaisesRegex(ValueError, "straight"):
                    arm.reach_elbow(0.3)
                target = arm.elbow_90 - arm.sign * 20 / 360
                if arm.sign > 0:
                    arm.lo[pickup.ELBOW] = target - pickup.RANGE_MARGIN / 2
                else:
                    arm.hi[pickup.ELBOW] = target + pickup.RANGE_MARGIN / 2
                with self.assertRaisesRegex(ValueError, "range margins"):
                    arm.reach_elbow(20 / 360)

    def test_pickup_cradle_is_relative_to_the_extended_elbow_pose(self):
        for tilt in (0.0, 0.01):
            with self.subTest(tilt=tilt):
                arms, _, ramps = self.run_main("--pickup", "--elbow-extension", "0.025", "--cradle", str(tilt))
                self.assertEqual(ramps.count(pickup.ELBOW), 2 + (tilt > 0))
                for arm in arms:
                    self.assertAlmostEqual(arm.cmd[pickup.ELBOW], arm.elbow_90 + arm.sign * (tilt - 0.025))
                    self.assertAlmostEqual(arm.cmd[pickup.J0], arm.top)

    def test_elbow_extension_interpolates_without_changing_other_joints(self):
        arms = [FakeArm(side) for side in ("left", "right")]
        for arm in arms:
            arm.cmd = arm.initial_pose()
            arm.cmd[pickup.SWING] = arm.sign * 0.18
            arm.cmd[pickup.WRIST_PITCH] = -arm.sign * pickup.HOOK_MAX_TRAVEL
        starts = [arm.cmd.copy() for arm in arms]
        targets = [arm.reach_elbow(pickup.ELBOW_EXTENSION) for arm in arms]
        pickup.ramp_joint(arms, pickup.ELBOW, targets, pickup.ELBOW_EXTENSION_SPEED, ease=pickup.smootherstep)
        self.assertGreaterEqual(self.clock.now, pickup.ELBOW_EXTENSION / pickup.ELBOW_EXTENSION_SPEED)
        for arm, start, target in zip(arms, starts, targets):
            held = [j for j in range(arm.dof) if j != pickup.ELBOW]
            self.assertAlmostEqual(arm.cmd[pickup.ELBOW], target)
            bends = [arm.sign * pos[pickup.ELBOW] for pos in arm.commands]
            self.assertTrue(all(a >= b - 1e-9 for a, b in zip(bends, bends[1:])))
            for pos in arm.commands:
                np.testing.assert_array_equal(pos[held], start[held])
                self.assertGreaterEqual(arm.sign * pos[pickup.ELBOW], arm.sign * target - 1e-9)

    def test_claws_roll_quarter_turn_while_raised_and_hold_through_pickup(self):
        for args in ((), ("--pickup",), ("--lower-only",), ("--hook", "0"),
                     ("--arm", "left"), ("--arm", "right")):
            with self.subTest(args=args):
                arms, _, _ = self.run_main(*args)
                for arm in arms:
                    target = arm.cfg.home[4] + arm.sign * 0.25
                    first_roll = next(i for i, pos in enumerate(arm.commands)
                                      if abs(pos[4] - target) < 1e-9)
                    self.assertAlmostEqual(arm.commands[first_roll][pickup.J0], arm.top)
                    self.assertAlmostEqual(arm.commands[first_roll][pickup.SWING], arm.cfg.home[pickup.SWING])
                    for pos in arm.commands[first_roll:]:
                        self.assertAlmostEqual(pos[4], target)

    def test_rolled_wrists_bend_inward_on_j6_without_moving_j5_or_claws(self):
        arms, _, _ = self.run_main()
        for arm in arms:
            self.assertAlmostEqual(arm.cmd[pickup.WRIST_PITCH], -arm.sign * pickup.HOOK_MAX_TRAVEL)
            self.assertAlmostEqual(arm.cmd[pickup.WRIST_YAW], arm.cfg.home[pickup.WRIST_YAW])
            self.assertAlmostEqual(arm.cmd[pickup.GRIPPER], arm.grip_open)

    def test_roll_outside_calibration_is_rejected_before_enabling_torque(self):
        initialize = FakeArm.__init__
        created = []

        def limited_roll(arm, side):
            initialize(arm, side)
            arm.lo[4], arm.hi[4] = -0.2, 0.2
            created.append(arm)

        with patch.object(FakeArm, "__init__", limited_roll), self.assertRaises(ValueError):
            self.run_main()
        for arm in created:
            self.assertNotIn(True, arm.torque)
            self.assertEqual(arm.commands, [])
            self.assertTrue(arm.closed)

    def test_roll_keeps_margin_from_calibrated_stops(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                arm = FakeArm(side)
                arm.lo[pickup.WRIST_ROLL], arm.hi[pickup.WRIST_ROLL] = -0.26, 0.26
                with self.assertRaisesRegex(ValueError, "wrist roll"):
                    arm.initial_pose()

    def test_rolled_wrist_inward_motion_respects_calibration_and_holds_other_joints(self):
        for side in ("left", "right"):
            for travel in (0.03, pickup.HOOK_MAX_TRAVEL):
                with self.subTest(side=side, travel=travel):
                    arm = FakeArm(side)
                    arm.cmd = arm.initial_pose()
                    arm.cmd[pickup.SWING] = arm.sign * 0.18
                    arm.lo[pickup.WRIST_PITCH], arm.hi[pickup.WRIST_PITCH] = -0.09, 0.09
                    start = arm.cmd.copy()
                    pickup.creep_to_contact([arm], lambda a: pickup.hook_direction(a, joint=pickup.WRIST_PITCH),
                                            pickup.HOOK_SPEED, pickup.HOOK_CONTACT_ERR, pickup.HOOK_SQUEEZE,
                                            max_travel=travel, label="rolled wrist")
                    limit = min(travel, 0.07)
                    self.assertAlmostEqual(arm.cmd[pickup.WRIST_PITCH], -arm.sign * limit)
                    held = [j for j in range(arm.dof) if j != pickup.WRIST_PITCH]
                    for pos in arm.commands:
                        self.assertLessEqual(abs(pos[pickup.WRIST_PITCH]), limit + 1e-9)
                        np.testing.assert_array_equal(pos[held], start[held])

    def test_lower_only_preserves_spread_pose(self):
        arms, _, ramps = self.run_main("--lower-only")
        self.assertEqual(ramps.count(pickup.J0), 2)
        for arm in arms:
            self.assertAlmostEqual(arm.cmd[pickup.SWING], arm.sign * 0.18)

    def test_pickup_keeps_inward_squeeze_through_grip_cradle_and_lift(self):
        arms, _, ramps = self.run_main("--pickup", "--hook", "0")
        self.assertEqual(ramps.count(pickup.J0), 3)
        for arm in arms:
            self.assertAlmostEqual(arm.cmd[pickup.SWING],
                                   arm.contact_position - arm.sign * pickup.PINCH_SQUEEZE)
            self.assertAlmostEqual(arm.cmd[pickup.J0], arm.top)
            self.assertAlmostEqual(arm.cmd[pickup.ELBOW],
                                   arm.elbow_90 + arm.sign * (pickup.CRADLE_TILT - 20 / 360))
            self.assertAlmostEqual(arm.cmd[pickup.GRIPPER], arm.grip_closed)

    def test_single_arm_grasps_inward(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                arms, _, _ = self.run_main("--arm", side)
                self.assertEqual(len(arms), 1)
                arm = arms[0]
                self.assertAlmostEqual(arm.cmd[pickup.SWING],
                                       arm.contact_position - arm.sign * pickup.PINCH_SQUEEZE)

    def test_lower_only_and_pickup_are_mutually_exclusive(self):
        with redirect_stdout(io.StringIO()), patch("sys.stderr", new_callable=io.StringIO), \
                self.assertRaises(SystemExit) as error:
            self.run_main("--lower-only", "--pickup")
        self.assertEqual(error.exception.code, 2)

    def creep(self, arm, max_travel=np.inf):
        arm.cmd[pickup.J0] = 1.5
        arm.cmd[pickup.ELBOW] = arm.elbow_90
        arm.cmd[pickup.SWING] = arm.sign * 0.18
        pickup.creep_to_contact([arm], pickup.pinch_direction, pickup.PINCH_SPEED,
                                pickup.PINCH_CONTACT_ERR, pickup.PINCH_SQUEEZE,
                                max_travel=max_travel, label="pinch")

    def test_contact_squeeze_respects_calibration_and_travel_cap(self):
        for side in ("left", "right"):
            for limit in (np.inf, 0.1):
                with self.subTest(side=side, limit=limit):
                    arm = FakeArm(side)
                    travel = min(0.36, limit)
                    arm.contact_position = arm.sign * (0.18 - travel + 0.018)
                    self.creep(arm, max_travel=limit)
                    for pos in arm.commands:
                        distance = 0.18 - arm.sign * pos[pickup.SWING]
                        self.assertGreaterEqual(distance, -1e-9)
                        self.assertLessEqual(distance, travel + 1e-9)
                    self.assertAlmostEqual(arm.cmd[pickup.SWING], arm.sign * (0.18 - travel))

    def test_no_contact_stops_at_calibrated_limit(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                arm = FakeArm(side)
                arm.contact_position = None
                self.creep(arm)
                self.assertAlmostEqual(arm.cmd[pickup.SWING], -arm.sign * 0.18)
                self.assertAlmostEqual(arm.cmd[pickup.ELBOW], arm.elbow_90)
                self.assertAlmostEqual(arm.cmd[pickup.J0], 1.5)

    def test_wrists_turn_inward_at_top_and_stay_rotated_after_descent(self):
        wrist = [pickup.WRIST_YAW, pickup.WRIST_PITCH]
        for args in ((), ("--pickup",), ("--arm", "left"), ("--arm", "right")):
            with self.subTest(args=args):
                arms, _, _ = self.run_main(*args)
                for arm in arms:
                    changed = [pos for previous, pos in zip(arm.commands, arm.commands[1:])
                               if not np.array_equal(previous[wrist], pos[wrist])]
                    self.assertTrue(changed)
                    for pos in changed:
                        self.assertAlmostEqual(pos[pickup.J0], arm.top)
                        self.assertAlmostEqual(pos[pickup.SWING], arm.sign * 0.18)
                        self.assertAlmostEqual(pos[pickup.ELBOW], arm.elbow_90)
                    first_low = next(i for i, pos in enumerate(arm.commands) if pos[pickup.J0] > 0.5)
                    expected = [0.0, -arm.sign * pickup.HOOK_MAX_TRAVEL]
                    for pos in arm.commands[first_low:]:
                        np.testing.assert_allclose(pos[wrist], expected, atol=1e-9)

    def test_hook_disabled_and_lower_only_skip_inward_wrist_preparation(self):
        for args in (("--hook", "0"), ("--lower-only",)):
            with self.subTest(args=args), patch.object(pickup, "hook_direction") as hook:
                arms, _, _ = self.run_main(*args)
                hook.assert_not_called()
                for arm in arms:
                    self.assertTrue(all(np.all(pos[[pickup.WRIST_YAW, pickup.WRIST_PITCH]] == 0)
                                        for pos in arm.commands))

    def test_defaults_increase_squeeze_and_raise_low_pose_on_mirrored_lifts(self):
        self.assertEqual(pickup.PINCH_SQUEEZE, 0.03)
        self.assertEqual(pickup.J0_BOTTOM_MARGIN, 1.0)
        for sign in (1, -1):
            self.assertAlmostEqual(pickup.j0_low_target(0, sign * 2, pickup.J0_BOTTOM_MARGIN), sign * 1.0)
            self.assertEqual(pickup.j0_low_target(0, sign * 2, 3), 0)

    def test_grasp_tuning_options_control_wrist_squeeze_and_height(self):
        arms, _, _ = self.run_main("--hook", "0.1", "--squeeze", "0.04", "--bottom-margin", "1.25")
        for arm in arms:
            self.assertAlmostEqual(arm.cmd[pickup.WRIST_YAW], 0.0)
            self.assertAlmostEqual(arm.cmd[pickup.WRIST_PITCH], -arm.sign * 0.1)
            self.assertAlmostEqual(arm.cmd[pickup.SWING], arm.contact_position - arm.sign * 0.04)
            self.assertAlmostEqual(arm.cmd[pickup.J0], 0.75)

    def test_invalid_wrist_and_squeeze_values_fail_before_hardware_access(self):
        for option in ("--hook", "--squeeze", "--elbow-extension"):
            for value in ("-0.1", "nan", "inf"):
                with self.subTest(option=option, value=value):
                    with patch.object(pickup, "Arm") as arm, \
                            patch.object(sys, "argv", ["pickup.py", option, value]), \
                            patch("sys.stderr", new_callable=io.StringIO), \
                            self.assertRaises(SystemExit) as error:
                        pickup.main()
                    self.assertEqual(error.exception.code, 2)
                    arm.assert_not_called()

    def test_optional_level_hook_respects_both_joint_ranges_and_preserves_height(self):
        wrist = [pickup.WRIST_YAW, pickup.WRIST_PITCH]
        for side in ("left", "right"):
            with self.subTest(side=side):
                arm = FakeArm(side)
                arm.cmd[pickup.SWING] = arm.sign * 0.18
                arm.cmd[pickup.ELBOW] = arm.elbow_90
                arm.lo[pickup.WRIST_PITCH], arm.hi[pickup.WRIST_PITCH] = -0.04, 0.04
                pickup.creep_to_contact([arm], lambda a: pickup.hook_direction(a, keep_height=True),
                                        pickup.HOOK_SPEED, pickup.HOOK_CONTACT_ERR, pickup.HOOK_SQUEEZE,
                                        max_travel=pickup.HOOK_MAX_TRAVEL, label="hook")
                np.testing.assert_allclose(arm.cmd[wrist], [-arm.sign * 0.08, arm.sign * 0.02], atol=1e-9)
                for pos in arm.commands:
                    self.assertTrue(np.all(pos[wrist] >= arm.lo[wrist] + pickup.RANGE_MARGIN - 1e-9))
                    self.assertTrue(np.all(pos[wrist] <= arm.hi[wrist] - pickup.RANGE_MARGIN + 1e-9))
                    self.assertAlmostEqual(pos[pickup.SWING], arm.sign * 0.18)
                    self.assertAlmostEqual(pos[pickup.J0], arm.top)
                    self.assertAlmostEqual(arm.fk(pos)[0][2], 0.2)

    def test_wrist_contact_is_independent_for_each_hand(self):
        live = FakeArm.live

        def blocked_wrist(arm):
            pos = live(arm)
            limit = 0.05 if arm.side == "left" else 0.1
            travel = min(-arm.sign * pos[pickup.WRIST_PITCH], limit)
            pos[pickup.WRIST_PITCH] = -arm.sign * travel
            return pos

        with patch.object(FakeArm, "live", blocked_wrist):
            arms, _, _ = self.run_main()
        for arm in arms:
            travel = (0.05 if arm.side == "left" else 0.1) + pickup.HOOK_SQUEEZE
            self.assertAlmostEqual(arm.cmd[pickup.WRIST_YAW], 0.0)
            self.assertAlmostEqual(arm.cmd[pickup.WRIST_PITCH], -arm.sign * travel)
            self.assertAlmostEqual(arm.cmd[pickup.J0], 1.0)

    def test_wrist_preparation_preserves_initialized_roll_j5_and_claw_extension(self):
        initialize = FakeArm.__init__

        def rolled_claws(arm, side):
            initialize(arm, side)
            arm.cfg.home[pickup.WRIST_YAW] = arm.sign * 0.07
            arm.cmd[pickup.WRIST_YAW] = -arm.sign * 0.04

        with patch.object(FakeArm, "__init__", rolled_claws):
            arms, _, ramps = self.run_main()
        self.assertEqual(ramps.count(pickup.GRIPPER), 1)
        for arm in arms:
            first_home = next(i for i, pos in enumerate(arm.commands)
                              if np.array_equal(pos, arm.initial_pose()))
            for pos in arm.commands[first_home:]:
                self.assertAlmostEqual(pos[pickup.GRIPPER], arm.grip_open)
                self.assertAlmostEqual(pos[pickup.WRIST_YAW], arm.sign * 0.07)
                self.assertAlmostEqual(pos[pickup.WRIST_ROLL], arm.sign * 0.25)
            self.assertAlmostEqual(arm.cmd[pickup.WRIST_PITCH], -arm.sign * pickup.HOOK_MAX_TRAVEL)

    def test_yaw_rotation_does_not_require_pitch_height_compensation(self):
        for side in ("left", "right"):
            for pitch_height in (0.0, 0.1):
                with self.subTest(side=side, pitch_height=pitch_height):
                    arm = FakeArm(side)
                    arm.cfg.ik.fk = lambda q: (np.array([
                        0.4, arm.sign * 0.5 + q[pickup.WRIST_YAW],
                        0.2 + q[pickup.WRIST_YAW] + pitch_height * q[pickup.WRIST_PITCH]]), None)
                    start = arm.cmd.copy()
                    direction = pickup.hook_direction(arm)
                    expected = np.zeros(arm.dof)
                    expected[pickup.WRIST_YAW] = -arm.sign
                    np.testing.assert_array_equal(direction, expected)
                    np.testing.assert_array_equal(arm.cmd, start)

    def test_yaw_rotation_respects_range_and_travel_without_moving_other_joints(self):
        for side in ("left", "right"):
            for travel in (0.03, pickup.HOOK_MAX_TRAVEL):
                with self.subTest(side=side, travel=travel):
                    arm = FakeArm(side)
                    arm.cmd[pickup.SWING] = arm.sign * 0.18
                    arm.cmd[pickup.ELBOW] = arm.elbow_90
                    arm.cmd[pickup.WRIST_PITCH] = arm.sign * 0.07
                    arm.cmd[pickup.GRIPPER] = arm.grip_open
                    arm.lo[pickup.WRIST_YAW], arm.hi[pickup.WRIST_YAW] = -0.09, 0.09
                    start = arm.cmd.copy()
                    pickup.creep_to_contact([arm], pickup.hook_direction, pickup.HOOK_SPEED,
                                            pickup.HOOK_CONTACT_ERR, pickup.HOOK_SQUEEZE,
                                            max_travel=travel, label="wrist yaw")
                    limit = min(travel, 0.07)
                    self.assertAlmostEqual(arm.cmd[pickup.WRIST_YAW], -arm.sign * limit)
                    held = [j for j in range(arm.dof) if j != pickup.WRIST_YAW]
                    for pos in arm.commands:
                        self.assertLessEqual(abs(pos[pickup.WRIST_YAW]), limit + 1e-9)
                        np.testing.assert_array_equal(pos[held], start[held])

    def test_stop_during_wrist_preparation_prevents_descent(self):
        sleep = self.clock.sleep

        def interrupt_wrist(seconds):
            sleep(seconds)
            if self.clock.now >= 0.02:
                pickup._sigint()

        with patch.object(self.clock, "sleep", side_effect=interrupt_wrist):
            arms, _, ramps = self.run_main()
        self.assertEqual(ramps.count(pickup.J0), 1)
        for arm in arms:
            self.assertAlmostEqual(arm.cmd[pickup.J0], arm.top)
            self.assertAlmostEqual(arm.cmd[pickup.SWING], arm.sign * 0.18)
            self.assertEqual(arm.torque, [False, True, False])
            self.assertTrue(arm.closed)

    def test_default_hold_republishes_rotated_grasp_pose_until_stop(self):
        arms, holds, _ = self.run_main(hold_seconds=None)
        self.assertIsNone(holds[-1][0])
        for arm in arms:
            arm.commands.clear()
            arm.torque.clear()
            arm.set_torque(True)
        sleep = self.clock.sleep
        ticks = 0

        def stop_after_ticks(seconds):
            nonlocal ticks
            sleep(seconds)
            ticks += 1
            if ticks == 5:
                pickup._sigint()

        with patch.object(self.clock, "sleep", side_effect=stop_after_ticks):
            pickup.hold(arms, None)
        for arm, final in zip(arms, holds[-1][1]):
            self.assertEqual(len(arm.commands), 5)
            for pos in arm.commands:
                np.testing.assert_array_equal(pos, final)
            self.assertEqual(arm.torque, [True])

    def test_initial_pose_uses_each_arms_home_without_mutating_it(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                arm = FakeArm(side)
                arm.cfg.home = arm.sign * np.array([0.2, 0.03, 0.02, 0.25, -0.04, 0.02, -0.01, 0.1])
                original = arm.cfg.home.copy()
                arm.cmd[:] = 0.12
                expected = original.copy()
                expected[pickup.J0], expected[pickup.GRIPPER] = arm.top, arm.grip_open
                expected[pickup.WRIST_ROLL] += arm.sign * 0.25
                np.testing.assert_array_equal(arm.initial_pose(), expected)
                np.testing.assert_array_equal(arm.cfg.home, original)

    def test_initializer_reaches_the_same_pose_from_different_starts(self):
        results = []
        for variant in (1, -1):
            arms = [FakeArm(side) for side in ("left", "right")]
            targets = []
            for arm in arms:
                arm.cfg.home[1] = arm.sign * 0.03
                arm.cfg.home[4] = -arm.sign * 0.04
                arm.cfg.home[pickup.WRIST_YAW] = arm.sign * 0.02
                arm.cfg.home[pickup.WRIST_PITCH] = -arm.sign * 0.01
                arm.cmd = variant * arm.sign * np.array([0, -0.1, 0.1, 0.1, 0.15, -0.12, 0.1, 0.05])
                arm.cmd[pickup.J0] = 1.0
                targets.append(arm.initial_pose())
            self.assertTrue(pickup.initialize_pose(arms, targets, pickup.J0_SPEED))
            for arm, target in zip(arms, targets):
                np.testing.assert_allclose(arm.cmd, target, atol=1e-9)
                np.testing.assert_allclose(arm.live(), target, atol=1e-9)
                for previous, pos in zip(arm.commands, arm.commands[1:]):
                    if not np.array_equal(previous[[1, 2, 4, 5, 6]], pos[[1, 2, 4, 5, 6]]):
                        self.assertAlmostEqual(pos[pickup.J0], arm.top)
                        self.assertAlmostEqual(pos[pickup.ELBOW], arm.elbow_90)
            results.append([arm.cmd.copy() for arm in arms])
        np.testing.assert_allclose(results[0], results[1], atol=1e-9)

    def test_main_repeats_the_same_grasp_without_accumulating_joint_offsets(self):
        initialize = FakeArm.__init__
        results = []
        for variant in (1, -1, 0):
            def different_start(arm, side):
                initialize(arm, side)
                arm.cfg.home[[1, 2, 4, 5, 6]] = arm.sign * np.array([0.03, 0.02, -0.04, 0.02, -0.01])
                arm.cmd = (variant * arm.sign * np.array([0, -0.1, 0.1, 0.1, 0.15, -0.12, 0.1, 0.05])
                           if variant else results[0][0 if side == "left" else 1].copy())

            with patch.object(FakeArm, "__init__", different_start):
                arms, _, _ = self.run_main("--spread", "0.12", "--hook", "0.1")
            for arm in arms:
                self.assertTrue(any(np.array_equal(pos, arm.initial_pose()) for pos in arm.commands))
                self.assertAlmostEqual(arm.cmd[pickup.WRIST_PITCH], arm.cfg.home[pickup.WRIST_PITCH] - arm.sign * 0.1)
                self.assertAlmostEqual(arm.cmd[pickup.WRIST_ROLL], arm.cfg.home[pickup.WRIST_ROLL] + arm.sign * 0.25)
                np.testing.assert_array_equal(arm.cmd[[1, 5]], arm.cfg.home[[1, 5]])
            results.append([arm.cmd.copy() for arm in arms])
        for result in results[1:]:
            np.testing.assert_allclose(results[0], result, atol=1e-9)

    def test_invalid_initial_pose_is_rejected_before_enabling_torque(self):
        initialize = FakeArm.__init__
        for invalid in (np.zeros(7), np.array([0, 0, 0, 0.25, np.nan, 0, 0, 0]),
                        np.array([0, 2, 0, 0.25, 0, 0, 0, 0])):
            with self.subTest(home=invalid):
                created = []

                def bad_home(arm, side):
                    initialize(arm, side)
                    arm.cfg.home = invalid.copy()
                    created.append(arm)

                with patch.object(FakeArm, "__init__", bad_home), self.assertRaises(ValueError):
                    self.run_main()
                for arm in created:
                    self.assertNotIn(True, arm.torque)
                    self.assertEqual(arm.commands, [])
                    self.assertTrue(arm.closed)

    def test_initialization_requires_both_arms_to_arrive_not_just_stall(self):
        arms = [FakeArm(side) for side in ("left", "right")]
        for arm in arms:
            arm.cmd[1] = 0.1
        for timeout in (pickup.ARRIVE_TIMEOUT_S, 0.02):
            with self.subTest(timeout=timeout), patch.object(arms[0], "live", return_value=np.zeros(8)):
                with self.assertRaisesRegex(RuntimeError, "J1"):
                    pickup.settle_joint(arms, 1, timeout=timeout, require_arrival=True)
                pickup.settle_joint(arms, 1, timeout=0.02)
        pickup.settle_joint(arms, 1, require_arrival=True)

    def test_initialization_failure_blocks_wrist_preparation_and_grasp(self):
        for blocked_joint in (1, pickup.WRIST_ROLL):
            def fail_home(arms, joint, **kwargs):
                if joint == blocked_joint and kwargs.get("require_arrival"):
                    raise RuntimeError(f"J{joint} did not arrive")

            with self.subTest(joint=blocked_joint), patch.object(pickup, "creep_to_contact") as creep:
                with self.assertRaisesRegex(RuntimeError, f"J{blocked_joint}"):
                    self.run_main(settle_effect=fail_home)
                creep.assert_not_called()

    def test_initialization_rechecks_the_full_pose_before_pickup(self):
        arm = FakeArm("left")
        arm.cfg.home[pickup.WRIST_PITCH] = 0.04
        target = arm.initial_pose()
        live = arm.live

        def drifted_shoulder():
            pos = live()
            if abs(arm.cmd[pickup.WRIST_PITCH] - target[pickup.WRIST_PITCH]) < 1e-9:
                pos[1] += 0.1
            return pos

        with patch.object(arm, "live", side_effect=drifted_shoulder):
            with self.assertRaisesRegex(RuntimeError, "initialization pose"):
                pickup.initialize_pose([arm], [target], pickup.J0_SPEED)

    def test_cancellation_during_command_flush_never_enables_torque(self):
        arms = [FakeArm(side) for side in ("left", "right")]
        with patch.object(pickup, "Arm", side_effect=arms), \
                patch.object(pickup, "hold", side_effect=pickup._sigint), \
                patch.object(sys, "argv", ["pickup.py"]):
            pickup.main()
        for arm in arms:
            self.assertNotIn(True, arm.torque)
            self.assertEqual(len(arm.commands), 1)
            self.assertTrue(arm.closed)

    def test_initialization_cancellation_does_not_run_later_stages(self):
        arm = FakeArm("left")
        arm.cmd[pickup.J0] = 1.0
        start = arm.cmd.copy()
        sleep = self.clock.sleep

        def interrupt_initialization(seconds):
            sleep(seconds)
            if self.clock.now >= 0.02:
                pickup._sigint()

        with patch.object(self.clock, "sleep", side_effect=interrupt_initialization):
            self.assertFalse(pickup.initialize_pose([arm], [arm.initial_pose()], pickup.J0_SPEED))
        held = [j for j in range(arm.dof) if j != pickup.ELBOW]
        for pos in arm.commands:
            np.testing.assert_array_equal(pos[held], start[held])

    def test_stop_prevents_inward_commands(self):
        arm = FakeArm("left")
        with patch.object(pickup, "_stop", True):
            self.creep(arm)
        self.assertEqual(arm.commands, [])


if __name__ == "__main__":
    unittest.main()
