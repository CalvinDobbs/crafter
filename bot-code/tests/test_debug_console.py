"""Offline tests for the debug console.

The console itself ships no simulator — it opens the robot's real providers, and off the robot it
says so. These tests supply MockAgentWorld as the component double, which is exactly what it is
for, and check that the console never opens the motors until it is armed.
"""

import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_types import BuildRequirements
from debug_console import ARGUMENTS, DEFAULT_VOXEL, OPERATIONS, DebugConsole, robot_sources
from mock_agent_world import MockAgentWorld

VOXEL = (0.3, 0.3, 0.3)


def real_time_world(box_count=4, voxel_size=VOXEL, action_duration=.05):
    """MockAgentWorld on the wall clock.

    Its virtual time only advances when the agent's run loop sleeps, and nothing drives that loop
    here — the operator does. A request stamped with the real clock would read as admitted from the
    future and be refused, so the double has to share the clock the console stamps with.
    """
    class RealTimeWorld(MockAgentWorld):
        @property
        def now(self):
            return time.time()+self._skew

        @now.setter
        def now(self, value):
            self._skew = value-time.time()

    world = RealTimeWorld(box_count, voxel_size=voxel_size, action_duration=action_duration,
                          action_timeout=120.0)
    world.now = time.time()
    world.all_visible = True
    return world


class ConsoleTests(unittest.TestCase):
    def console(self, *, kind="test double", box_count=4, action_duration=.05, actions=True,
                observations_error=None, actions_error=None):
        world = real_time_world(box_count, VOXEL, action_duration)
        opened = {"observations": 0, "actions": 0}

        def open_observations():
            opened["observations"] += 1
            if observations_error:
                raise observations_error
            return world.observations, None

        def open_actions(observations):
            opened["actions"] += 1
            if actions_error:
                raise actions_error
            self.assertIs(observations, world.observations)
            return world.actions, None

        console = DebugConsole(open_observations, open_actions if actions else None,
                               kind=kind, voxel_size=VOXEL)
        self.addCleanup(console.close)
        console.world, console.opened = world, opened
        return console

    def armed(self, **kwargs):
        console = self.console(**kwargs)
        console.observe()
        console.arm(True)
        return console

    def settle(self, console, timeout=6.0):
        deadline = time.monotonic()+timeout
        while console.state()["active"] is not None and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertIsNone(console.state()["active"], "action never reached a terminal status")

    def test_every_agent_operation_is_offered_with_its_arguments(self):
        from agent_types import MOTION_OPS, Operation, Step
        self.assertEqual(set(OPERATIONS), set(Operation.__args__))
        self.assertEqual(set(ARGUMENTS), set(OPERATIONS))
        fields = {field for arguments in ARGUMENTS.values() for field in arguments}
        self.assertTrue(fields <= set(Step.__dataclass_fields__))
        self.assertTrue(MOTION_OPS <= set(OPERATIONS))

    def test_the_console_ships_no_simulator(self):
        """A fake reading is indistinguishable from a real one, so none may be available here."""
        source = (Path(__file__).resolve().parents[1] / "debug_console.py").read_text()
        for banned in ("MockAgentWorld", "mock_agent_world", "simulated_factory", "mock=True"):
            with self.subTest(banned=banned):
                self.assertNotIn(banned, source)
        observations, actions = robot_sources(DEFAULT_VOXEL)
        self.assertTrue(callable(observations) and callable(actions))

    def test_nothing_is_opened_before_the_first_call(self):
        console = self.console()
        state = console.state()
        self.assertFalse(state["perception_open"])
        self.assertFalse(state["actions_open"])
        self.assertFalse(state["armed"])
        self.assertEqual(console.opened, {"observations": 0, "actions": 0})
        self.assertIsNone(state["snapshot"])
        self.assertIsNone(state["capabilities"]["observations"])
        json.dumps(state, allow_nan=False)

    def test_inspecting_perception_never_opens_the_motors(self):
        """Readers are unlimited; the writers are single-owner. Looking must not claim them."""
        console = self.console()
        console.observe()
        console.find_sites()
        console.select_site("floor-a")
        console.observe("floor-a")
        console.verify()
        self.assertEqual(console.opened, {"observations": 1, "actions": 0})
        state = console.state()
        self.assertTrue(state["perception_open"])
        self.assertFalse(state["actions_open"])
        self.assertIsNone(state["executor"])
        self.assertIsNone(state["capabilities"]["actions"])
        self.assertTrue(state["snapshot"]["valid"])

    def test_arming_is_what_opens_the_hardware_writers(self):
        console = self.console()
        console.observe()
        self.assertEqual(console.opened["actions"], 0)
        console.arm(True)
        self.assertEqual(console.opened["actions"], 1)
        state = console.state()
        self.assertTrue(state["armed"] and state["actions_open"])
        self.assertTrue(state["capabilities"]["actions"]["operations"])

    def test_disarming_keeps_the_writers_so_a_held_box_is_not_dropped(self):
        console = self.armed()
        console.arm(False)
        state = console.state()
        self.assertFalse(state["armed"])
        self.assertTrue(state["actions_open"], "closing the writers would cut arm torque")
        with self.assertRaisesRegex(ValueError, "arm actions"):
            console.submit("look_around", search="materials")
        console.arm(True)
        self.assertEqual(console.opened["actions"], 1, "re-arming must not re-open the writers")

    def test_observe_records_evidence_and_serializes_the_whole_snapshot(self):
        console = self.console()
        console.observe()
        state = console.state()
        json.dumps(state, allow_nan=False)
        self.assertEqual(len(state["snapshot"]["boxes"]), 4)
        self.assertTrue(state["snapshot"]["images"])
        self.assertNotIn("data_url", state["snapshot"]["images"][0])
        self.assertNotIn("world_model_json", state["snapshot"])
        self.assertEqual(state["log"][0]["title"], "observe()")
        media_type, data = console.image(state["snapshot"]["images"][0]["view"])
        self.assertEqual(media_type, "image/png")
        self.assertTrue(data.startswith(b"\x89PNG"))

    def test_site_selection_follows_the_agents_own_two_calls(self):
        console = self.console()
        console.observe()
        self.assertEqual([site.id for site in console.find_sites()], ["obstructed", "floor-a"])
        self.assertIsNone(console.select_site("obstructed"))
        self.assertIsNotNone(console.select_site("floor-a"))
        snapshot = console.observe()
        self.assertEqual(snapshot.site_id, "floor-a")
        self.assertTrue(snapshot.occupancy_complete)

    def test_actions_carry_the_same_admission_envelope_the_agent_builds(self):
        console = self.armed()
        console.find_sites()
        console.select_site("floor-a")
        console.observe("floor-a")
        console.submit("approach_box", box_id=0, site_id="floor-a", cell=[0, 0, 0])
        self.settle(console)
        request = console.world.requests[-1]
        self.assertEqual(request.job_id, console.job_id)
        self.assertEqual(request.requirements, console.requirements)
        self.assertEqual(request.site.id, "floor-a")
        self.assertEqual(request.box.id, 0)
        request.validate_admission(request.submitted_at)
        self.assertEqual(console.state()["outcome"]["status"], "succeeded")

    def test_a_hand_driven_pick_is_monitored_and_confirms_possession(self):
        console = self.armed()
        console.find_sites()
        console.select_site("floor-a")
        console.observe("floor-a")
        console.submit("approach_box", box_id=0, site_id="floor-a", cell=[0, 0, 0])
        self.settle(console)
        console.submit("pickup", box_id=0)
        self.settle(console)
        state = console.state()
        self.assertEqual(state["snapshot"]["holding"]["status"], "holding")
        self.assertEqual(state["snapshot"]["holding"]["box_id"], 0)
        self.assertTrue(console.world.monitor_calls)
        self.assertTrue(any(entry["kind"] == "monitor" for entry in state["log"]))

    def test_arguments_are_refused_before_anything_is_submitted(self):
        console = self.console()
        with self.assertRaisesRegex(ValueError, "arm actions"):
            console.submit("approach_box", box_id=0)
        console.observe()
        console.arm(True)
        with self.assertRaises(ValueError):
            console.submit("observe")                          # not a motion the provider executes
        with self.assertRaises(ValueError):
            console.submit("approach_box", box_id=99)          # not in the latest observation
        with self.assertRaises(ValueError):
            console.submit("place", box_id=0, site_id="floor-a")   # site was never selected
        self.assertEqual(console.world.requests, [])

    def test_a_motion_needs_an_observation_even_when_armed(self):
        console = self.console()
        console.arm(True)
        with self.assertRaisesRegex(ValueError, "observe first"):
            console.submit("look_around", search="materials")
        self.assertEqual(console.world.requests, [])

    def test_stop_says_so_when_this_panel_owns_no_motors(self):
        console = self.console()
        reply = console.stop()
        self.assertFalse(reply.acknowledged)
        self.assertEqual(console.opened["actions"], 0)
        self.assertIn("owns no motors", console.state()["log"][0]["title"])
        console.observe()
        console.arm(True)
        self.assertTrue(console.stop().acknowledged)

    def test_a_perception_only_console_refuses_every_motion(self):
        console = self.console(actions=False)
        console.observe()
        state = console.state()
        self.assertFalse(state["actions_available"])
        with self.assertRaisesRegex(ValueError, "no action provider"):
            console.arm(True)
        with self.assertRaises(ValueError):
            console.submit("look_around", search="materials")
        self.assertEqual(console.opened["actions"], 0)

    def test_providers_that_cannot_be_opened_are_reported_not_raised_raw(self):
        console = self.console(observations_error=ImportError("No module named 'bbos'"))
        with self.assertRaisesRegex(ValueError, "perception could not be opened"):
            console.observe()
        state = console.state()
        self.assertFalse(state["perception_open"])
        self.assertEqual(state["log"][0]["kind"], "error")
        self.assertIn("bbos", state["error"])

    def test_writers_owned_elsewhere_fail_arming_with_the_reason(self):
        """mc_skills already owns arm_*.ctrl: that must surface, not crash the page."""
        console = self.console(actions_error=RuntimeError("arm_right.ctrl is owned by pid 4242"))
        console.observe()
        with self.assertRaisesRegex(ValueError, "could not open the hardware writers"):
            console.arm(True)
        state = console.state()
        self.assertFalse(state["armed"])
        self.assertFalse(state["actions_open"])
        self.assertIn("4242", state["error"])

    def test_verify_measures_the_target_cells_without_confirming_anything(self):
        console = self.console()
        console.set_requirements(BuildRequirements(((0, 0, 0), (0, 1, 0)), VOXEL, (1, 2, 1)),
                                 "two stacked cells")
        console.observe()
        console.find_sites()
        console.select_site("floor-a")
        cells = console.verify()
        self.assertEqual([item["cell"] for item in cells], [[0, 0, 0], [0, 1, 0]])
        self.assertTrue(all(item["status"] == "empty" for item in cells))
        self.assertIn("NOT accept done", console.state()["log"][0]["data"]["verdict"])

    def test_requirements_cannot_change_under_a_running_action(self):
        console = self.armed(action_duration=2.0)
        console.find_sites()
        console.select_site("floor-a")
        console.observe("floor-a")
        console.submit("approach_box", box_id=0, site_id="floor-a", cell=[0, 0, 0])
        with self.assertRaises(ValueError):
            console.set_requirements(BuildRequirements(((0, 0, 0),), VOXEL, (1, 1, 1)), "other")
        with self.assertRaises(ValueError):
            console.submit("pickup", box_id=0)
        with self.assertRaisesRegex(ValueError, "stop the running action"):
            console.arm(False)
        console.stop()
        self.settle(console)

    def test_the_default_voxel_is_the_real_cardboard_box(self):
        from contracts import BOX_SIZE
        self.assertEqual(DEFAULT_VOXEL, (BOX_SIZE, BOX_SIZE, BOX_SIZE))


if __name__ == "__main__":
    unittest.main()
