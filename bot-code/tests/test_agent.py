import sys
import threading
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import Block, Structure
from agent_types import AgentConfig, Holding, Step
from agent import Agent, BusyError, JobManager, validate_job
from mock_agent_world import MockAgentWorld


def structure():
    return Structure([Block(0, 0, 0), Block(1, 0, 0), Block(0, 1, 0)])


class AgentTests(unittest.TestCase):
    def run_world(self, world=None, config=None, reasoner=None, backend="auto"):
        world = world or MockAgentWorld(3)
        agent = Agent(world, world, config=config, reasoner=reasoner, backend=backend,
                      clock=world.clock, sleep=world.sleep)
        return agent, world, agent.run(structure())

    def test_full_build_surveys_selects_and_carries(self):
        agent, world, result = self.run_world()
        self.assertTrue(result.success, result)
        self.assertEqual(result.placed, 3)
        self.assertIsNone(agent.manager.active)
        ops = [r.step.operation for r in world.requests]
        self.assertIn("look_around", ops)
        self.assertEqual(ops.count("pickup"), 3)
        self.assertEqual(ops.count("move_to_build"), 3)
        self.assertEqual(ops.count("place"), 3)
        self.assertEqual(world.holding.status, "empty")
        self.assertEqual(len(set(world.placed.values())), 3)
        self.assertEqual(world.selected_site, "floor-a")

    def test_structure_validation_and_copy(self):
        source = Structure([Block(-3, 0, 8), Block(-3, 1, 8)])
        job = validate_job(source, "job", AgentConfig())
        source.blocks[0].x = 100
        self.assertEqual(job.requirements.cells, ((0, 0, 0), (0, 1, 0)))
        self.assertEqual(job.original_cells[0], (-3, 0, 8))
        for blocks in ([], [Block(0, 1, 0)], [Block(True, 0, 0)],
                       [Block(0, 0, 0), Block(0, 0, 0)], [Block(0, -1, 0)]):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                validate_job(Structure(blocks), "job", AgentConfig())

    def test_concurrent_admission_rejects_instead_of_queuing(self):
        manager = JobManager()
        barrier = threading.Barrier(3)
        results = []
        def submit(i):
            barrier.wait()
            try:
                manager.submit(structure(), str(i))
                results.append("accepted")
            except BusyError:
                results.append("busy")
        threads = [threading.Thread(target=submit, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        self.assertCountEqual(results, ["accepted", "busy"])

    def test_known_pregrasp_failure_recovers(self):
        world = MockAgentWorld(3, faults={"pickup": ["pregrasp"]})
        _, _, result = self.run_world(world)
        self.assertTrue(result.success, result)
        self.assertGreater(sum(r.step.operation == "pickup" for r in world.requests), 3)

    def test_known_prerelease_failure_recovers_without_new_pick(self):
        world = MockAgentWorld(3, faults={"place": ["prerelease"]})
        _, _, result = self.run_world(world)
        self.assertTrue(result.success, result)
        ops = [r.step.operation for r in world.requests]
        self.assertEqual(ops.count("pickup"), 3)
        self.assertEqual(ops.count("place"), 4)

    def test_unknown_pick_stops_and_remains_busy(self):
        world = MockAgentWorld(3, faults={"pickup": ["unknown_holding"]})
        agent, _, result = self.run_world(world)
        self.assertEqual(result.status, "NEEDS_OPERATOR")
        self.assertNotIn("move_to_build", [r.step.operation for r in world.requests])
        with self.assertRaises(BusyError):
            agent.manager.submit(structure(), "another")

    def test_timeout_after_dispatch_is_not_replayed(self):
        world = MockAgentWorld(3, faults={"pickup": ["submit_timeout"]})
        _, _, result = self.run_world(world)
        self.assertEqual(result.status, "NEEDS_OPERATOR")
        self.assertEqual(sum(r.step.operation == "pickup" for r in world.requests), 1)
        self.assertEqual(len({r.request_id for r in world.requests}), len(world.requests))

    def test_unverified_place_does_not_count(self):
        world = MockAgentWorld(3, faults={"place": ["bad_placement"]})
        _, _, result = self.run_world(world)
        self.assertEqual(result.status, "NEEDS_OPERATOR")
        self.assertEqual(result.placed, 0)

    def test_postrelease_failure_can_be_verified(self):
        world = MockAgentWorld(3, faults={"place": ["postrelease"]})
        _, _, result = self.run_world(world)
        self.assertTrue(result.success, result)
        self.assertEqual(sum(r.step.operation == "place" for r in world.requests), 3)

    def test_no_sites_and_insufficient_materials_are_bounded(self):
        for world in (MockAgentWorld(1), MockAgentWorld(3, no_sites=True)):
            with self.subTest(world=world):
                _, _, result = self.run_world(world, AgentConfig(max_searches=2))
                self.assertFalse(result.success)
                self.assertLessEqual(sum(r.step.operation == "look_around" for r in world.requests), 2)
                self.assertEqual(result.placed, 0)

    def test_invented_model_decision_falls_back_without_dispatch(self):
        class BadModel:
            def decide(self, context, choices):
                return Step("pickup", box_id=999)
        _, world, result = self.run_world(reasoner=BadModel())
        self.assertTrue(result.success, result)
        self.assertNotIn(999, [r.step.box_id for r in world.requests])
        self.assertTrue(any(e["event"] == "model_fallback" for e in result.events))

    def test_strict_llm_failure_is_not_silent_fallback(self):
        class BadModel:
            def decide(self, context, choices):
                raise RuntimeError("provider failure")
        _, world, result = self.run_world(reasoner=BadModel(), backend="llm")
        self.assertFalse(result.success)
        self.assertFalse(world.requests)

    def test_pose_epoch_change_during_reasoning_rejects_old_decision(self):
        world = MockAgentWorld(3)
        class MovingModel:
            changed = False
            def decide(self, context, choices):
                decision = choices[0]
                if decision.operation == "pickup" and not self.changed:
                    self.changed = True
                    world.epoch += 1
                return decision
        _, _, result = self.run_world(world, reasoner=MovingModel())
        self.assertFalse(result.success)
        self.assertNotIn("pickup", [r.step.operation for r in world.requests])

    def test_cancel_acknowledgement_is_not_stopped(self):
        world = MockAgentWorld(3, faults={"pickup": ["running_forever"]})
        world.cancel_stops = False
        agent, _, result = self.run_world(world)
        self.assertEqual(result.status, "NEEDS_OPERATOR")
        self.assertIsNotNone(agent.manager.active)
        self.assertTrue(world.cancellations)

    def test_stale_snapshots_never_dispatch(self):
        world = MockAgentWorld(3)
        world.stale = True
        _, _, result = self.run_world(world)
        self.assertFalse(result.success)
        self.assertFalse(world.requests)

    def test_logging_failure_never_replays_actions(self):
        world = MockAgentWorld(3)
        def broken(event):
            raise RuntimeError("logging failed")
        agent = Agent(world, world, clock=world.clock, sleep=world.sleep, event_sink=broken)
        result = agent.run(structure())
        self.assertTrue(result.success, result)
        self.assertEqual(sum(r.step.operation == "pickup" for r in world.requests), 3)

    def test_cancel_during_model_call_prevents_dispatch(self):
        world = MockAgentWorld(3)
        class CancellingModel:
            def decide(self, context, choices):
                agent.cancel()
                return choices[0]
        agent = Agent(world, world, reasoner=CancellingModel(), clock=world.clock, sleep=world.sleep)
        result = agent.run(structure())
        self.assertEqual(result.status, "CANCELLED")
        self.assertFalse(world.requests)

    def test_possession_after_approach_does_not_authorize_pickup(self):
        class UnexpectedHolding(MockAgentWorld):
            def submit(self, request):
                receipt = super().submit(request)
                if request.step.operation == "approach_box":
                    self.holding = Holding("holding", request.step.box_id, self.now, "mock-gripper")
                    self.outcomes[receipt.action_id] = replace(self.outcomes[receipt.action_id], holding=self.holding)
                return receipt
        _, world, result = self.run_world(UnexpectedHolding(3))
        self.assertEqual(result.status, "NEEDS_OPERATOR")
        self.assertNotIn("pickup", [r.step.operation for r in world.requests])

    def test_failed_approach_revalidates_lost_box(self):
        class LostBox(MockAgentWorld):
            lost = None
            def submit(self, request):
                receipt = super().submit(request)
                if request.step.operation == "approach_box" and self.lost is None:
                    self.lost = request.step.box_id
                return receipt
            def observe(self, site_id=None):
                snapshot = super().observe(site_id)
                return replace(snapshot, boxes=tuple(b for b in snapshot.boxes if b.id != self.lost))
        world = LostBox(4, faults={"approach_box": ["blocked"]})
        _, _, result = self.run_world(world)
        self.assertTrue(result.success, result)
        self.assertEqual(sum(r.step.operation == "approach_box" and r.step.box_id == world.lost
                             for r in world.requests), 1)

    def test_conflicting_possession_retains_job(self):
        class Conflicting(MockAgentWorld):
            def observe(self, site_id=None):
                return replace(super().observe(site_id), holding=Holding("holding", 99, self.now, "camera"))
        agent, world, result = self.run_world(Conflicting(3))
        self.assertEqual(result.status, "NEEDS_OPERATOR")
        self.assertIsNotNone(agent.manager.active)
        self.assertFalse(world.requests)

    def test_cancel_with_verified_stop_releases_job(self):
        world = MockAgentWorld(3, faults={"pickup": ["running_forever"]})
        agent, _, result = self.run_world(world)
        self.assertEqual(result.status, "CANCELLED")
        self.assertIsNone(agent.manager.active)
        self.assertNotIn("move_to_build", [r.step.operation for r in world.requests])

    def test_duplicate_inventory_is_not_counted_twice(self):
        class Duplicates(MockAgentWorld):
            def observe(self, site_id=None):
                snapshot = super().observe(site_id)
                return replace(snapshot, boxes=snapshot.boxes+snapshot.boxes)
        _, world, result = self.run_world(Duplicates(3))
        self.assertFalse(result.success)
        self.assertNotIn("pickup", [r.step.operation for r in world.requests])

    def test_scripted_reasoner_completes_four_box_demo(self):
        class Scripted:
            def decide(self, context, choices):
                return choices[0]
        world = MockAgentWorld(4)
        source = Structure([Block(0, 0, 0), Block(1, 0, 0), Block(0, 0, 1), Block(0, 1, 0)])
        result = Agent(world, world, reasoner=Scripted(), backend="llm",
                       clock=world.clock, sleep=world.sleep).run(source)
        self.assertTrue(result.success, result)
        self.assertEqual(result.placed, 4)

    def test_invalid_site_candidates_are_not_selected(self):
        for changes in ({"epoch": 99}, {"dimensions": (.01, .01, .01)},
                        {"ts": 1.0}, {"floor_valid": False}, {"feasible": False}):
            class InvalidSite(MockAgentWorld):
                def _site(self, site_id, valid=True):
                    return replace(super()._site(site_id, valid), **changes)
            with self.subTest(changes=changes):
                _, world, result = self.run_world(InvalidSite(3))
                self.assertFalse(result.success)
                self.assertNotIn("pickup", [r.step.operation for r in world.requests])

    def test_changed_support_blocks_later_actions(self):
        class FallenStack(MockAgentWorld):
            def observe(self, site_id=None):
                snapshot = super().observe(site_id)
                if len(self.placed) == 1 and self.requests[-1].step.operation == "approach_box":
                    self.placed.clear()
                    return super().observe(site_id)
                return snapshot
        _, world, result = self.run_world(FallenStack(3))
        self.assertFalse(result.success)
        self.assertEqual(sum(r.step.operation == "pickup" for r in world.requests), 1)

    def test_material_types_are_ignored(self):
        source = structure()
        for block in source.blocks:
            block.kind = "imaginary-material"
        world = MockAgentWorld(3)
        result = Agent(world, world, clock=world.clock, sleep=world.sleep).run(source)
        self.assertTrue(result.success, result)

    def test_observe_only_reasoner_hits_budget(self):
        class Observer:
            def decide(self, context, choices):
                return Step("observe")
        _, world, result = self.run_world(reasoner=Observer(), config=AgentConfig(max_no_progress=3))
        self.assertFalse(result.success)
        self.assertLessEqual(result.steps, 4)
        self.assertFalse(world.requests)


if __name__ == "__main__":
    unittest.main()
