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
        self.cfg = SimpleNamespace(q2urdf=lambda q: q, ik=SimpleNamespace(fk=self.fk))
        self.commands = []
        self.torque = []
        self.closed = False

    def fk(self, q):
        return np.array([0.4, self.sign * 0.5 + q[pickup.SWING], 0.2]), None

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

    def run_main(self, *args):
        arms, holds, ramps = [], [], []

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
                patch.object(pickup, "settle_joint"), \
                patch.object(pickup, "hold", side_effect=hold), \
                patch.object(sys, "argv", ["pickup.py", "--hold", "0.02", *args]):
            pickup.main()
        return arms, holds, ramps

    def test_default_grasps_inward_after_lowering_and_holds_without_lifting(self):
        arms, holds, ramps = self.run_main()
        self.assertEqual(ramps.count(pickup.J0), 2)
        self.assertEqual(holds[-1][0], 0.02)
        for arm, final in zip(arms, holds[-1][1]):
            self.assertAlmostEqual(final[pickup.SWING],
                                   arm.contact_position - arm.sign * pickup.PINCH_SQUEEZE)
            self.assertAlmostEqual(final[pickup.J0], 1.5)
            self.assertAlmostEqual(final[pickup.ELBOW], arm.elbow_90)
            spread = arm.sign * 0.18
            low_commands = [pos for pos in arm.commands if pos[pickup.J0] == 1.5]
            self.assertAlmostEqual(low_commands[0][pickup.SWING], spread)
            swings = [arm.sign * pos[pickup.SWING] for pos in low_commands]
            self.assertTrue(all(a >= b - 1e-9 for a, b in zip(swings, swings[1:])))
            self.assertTrue(arm.closed)
            self.assertEqual(arm.torque, [False, True, False])

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
            self.assertAlmostEqual(arm.cmd[pickup.ELBOW], arm.cradle(pickup.CRADLE_TILT))
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

    def test_stop_prevents_inward_commands(self):
        arm = FakeArm("left")
        with patch.object(pickup, "_stop", True):
            self.creep(arm)
        self.assertEqual(arm.commands, [])


if __name__ == "__main__":
    unittest.main()
