# /// script
# dependencies = [
#   "bbos",
#   "numpy",
#   "fastapi",
#   "uvicorn",
#   "opencv-python-headless",
#   "nbtlib",
#   "openai",
# ]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""minecraft — orchestrator. `run minecraft` entry point.

Pipeline: structure -> scan -> plan -> execute -> celebrate.

Prereq: the body server must be up first —
    uv run ~/bbapps/mc_skills/main.py          (owns all hardware writers)

Usage:
    uv run main.py --structure fixtures/structure_house.json
    uv run main.py --grid-ui              # build in browser first, then go
    uv run main.py --mock                 # full dry run, no robot needed
    uv run main.py --planner deterministic|llm|auto
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from contracts import Block, Structure, cell_center, detections_to_dict
from structure_src import load_structure
from skills_client import SkillsClient, MockSkills


def main(argv=None):
    ap = argparse.ArgumentParser(description="Closed-loop floor-box reasoning; mock mode is offline by default.")
    ap.add_argument("--save-api-key", action="store_true",
                    help="save exported OPENAI_API_KEY privately in ~/.config/crafter; never overwrite an existing key")
    ap.add_argument("--ui", action="store_true", help="Minecraft control panel: real LLM decisions, simulated tools")
    ap.add_argument("--ui-host", default="127.0.0.1")
    ap.add_argument("--ui-port", type=int, default=8005)
    ap.add_argument("--receiver-host", default="0.0.0.0")
    ap.add_argument("--receiver-port", type=int, default=5005)
    ap.add_argument("--mode", choices=["agent", "oneshot"], default="agent",
                    help="agent is default; oneshot is the legacy fixed-grid path, not mobile execution")
    ap.add_argument("--provider", help="explicit live adapter factory: module:function returning AgentProviders")
    ap.add_argument("--observe-perception", metavar="URL",
                    help="inspect a live perception server without actions; --planner llm adds one read-only model decision")
    ap.add_argument("--box-size", type=float, help="uniform box edge in meters; required for live agent")
    ap.add_argument("--max-steps", type=int, default=512)
    ap.add_argument("--max-actions", type=int, default=512)
    ap.add_argument("--max-searches", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--max-no-progress", type=int, default=16)
    ap.add_argument("--fresh-seconds", type=float, default=2.0)
    ap.add_argument("--model", default=os.environ.get("CRAFTER_LLM_MODEL", "gpt-4o-mini"))
    ap.add_argument("--base-url", default=os.environ.get("CRAFTER_LLM_BASE_URL"))
    ap.add_argument("--llm-timeout", type=float, default=15.0)
    ap.add_argument("--json-only", action="store_true", help="explicit compatibility mode; local validation remains strict")
    ap.add_argument("--allow-api-with-mock", action="store_true", help="explicitly allow model API calls with mock hardware")
    ap.add_argument("--structure", default=None)
    ap.add_argument("--grid-ui", action="store_true",
                    help="serve the click-grid until /save, then proceed")
    ap.add_argument("--planner", default="auto",
                    choices=["auto", "deterministic", "llm"])
    ap.add_argument("--skills-url", default="http://localhost:8006")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--sweeps", type=int, default=1,
                    help=">1 rotates the base between scans to find more boxes")
    a = ap.parse_args(argv)
    if a.observe_perception:
        if a.mock or a.ui or a.provider or a.grid_ui or a.mode != "agent" or a.save_api_key or a.sweeps != 1:
            ap.error("--observe-perception is read-only and cannot select mock, UI, actions or legacy execution")
        return observe_perception(a)
    if a.save_api_key:
        from agent_backend import save_api_key_from_env
        try:
            path = save_api_key_from_env()
        except (OSError, ValueError) as exc:
            print(f"[key] {exc}", file=sys.stderr)
            return 2
        print(f"[key] Saved to {path} (owner-only permissions). Key value not displayed.")
        if not a.ui:
            return 0
    if a.ui:
        if a.provider or a.grid_ui or a.mode != "agent" or a.planner == "deterministic":
            ap.error("the UI uses real LLM decisions with simulated tools; do not select another executor")
        from panel import serve_panel
        return serve_panel(host=a.ui_host, port=a.ui_port, receiver_host=a.receiver_host,
                           receiver_port=a.receiver_port, model=a.model, base_url=a.base_url,
                           model_timeout=a.llm_timeout, json_only=a.json_only)
    if a.mode == "agent" and a.sweeps != 1:
        ap.error("agent surveys through look_around; --sweeps is a legacy option")

    # --- 1. target structure -------------------------------------------------
    if a.grid_ui:
        import threading
        from structure_src import serve_grid_ui, OUT
        threading.Thread(target=serve_grid_ui, daemon=True).start()
        print(f"[orchestrator] build in the grid UI, hit SAVE; watching {OUT}",
              flush=True)
        while not OUT.exists():
            time.sleep(0.5)
        time.sleep(1.0)   # let the human finish clicking save
        structure = load_structure(str(OUT))
    elif a.mock and a.mode == "agent" and a.structure is None:
        structure = Structure([Block(0, 0, 0), Block(1, 0, 0), Block(0, 0, 1), Block(0, 1, 0)])
    else:
        structure = load_structure(
            a.structure or str(Path(__file__).parent
                               / "fixtures" / "structure_house.json"))
    print(f"[orchestrator] target: {len(structure.blocks)} blocks, "
          f"{len(structure.layers())} layers", flush=True)

    if a.mode == "agent":
        return run_agent(a, structure)
    from perception import scan_all
    from planner import plan_build
    skills = MockSkills() if a.mock else SkillsClient(a.skills_url)

    # --- 2. scan the world ---------------------------------------------------
    print("[orchestrator] scanning for boxes...", flush=True)
    skills.home()
    boxes = scan_all(skills=None if a.mock else skills,
                     mock=a.mock, sweeps=a.sweeps)
    print(f"[orchestrator] saw {len(boxes)} boxes: "
          f"{json.dumps(detections_to_dict(boxes))}", flush=True)
    if len(boxes) < len(structure.blocks):
        print(f"[orchestrator] WARN: need {len(structure.blocks)} boxes, "
              f"found {len(boxes)}", flush=True)

    # --- 3. plan -------------------------------------------------------------
    plan = plan_build(structure, boxes, backend=a.planner)
    print(f"[orchestrator] plan: {len(plan.actions)} actions", flush=True)
    by_id = {d.id: d for d in boxes}

    # --- 4. execute ----------------------------------------------------------
    narr = iter(plan.narration)
    i = 0
    while i < len(plan.actions):
        pick, place = plan.actions[i], plan.actions[i + 1]
        line = next(narr, "")
        if line:
            skills.say(line)
        box = by_id.get(pick.box_id)
        if box is None:
            print(f"[orchestrator] missing box {pick.box_id}, skipping step",
                  flush=True)
            i += 2
            continue
        print(f"[orchestrator] pick box {box.id} at {box.pos}", flush=True)
        skills.pick(box.pos)
        target = cell_center(*place.cell)
        print(f"[orchestrator] place cell {place.cell} -> {target}", flush=True)
        skills.place(target)
        i += 2

    # --- 5. done -------------------------------------------------------------
    skills.celebrate()
    skills.say("build complete. creepers not included.")
    print("[orchestrator] done", flush=True)


def run_agent(args, structure):
    from agent import Agent, CapabilityError, validate_job
    from agent_adapters import AgentProviders, load_providers
    from agent_types import AgentConfig
    providers = None
    try:
        if args.mock and args.provider:
            raise ValueError("--mock cannot load a live --provider")
        if not args.mock and args.box_size is None:
            raise ValueError("live agent requires --box-size and an explicit --provider")
        if args.mock and args.planner == "llm" and not args.allow_api_with_mock:
            raise ValueError("--planner llm with --mock requires --allow-api-with-mock")
        size = args.box_size if args.box_size is not None else .3
        config = AgentConfig(voxel_size=(size, size, size), max_steps=args.max_steps,
                             max_actions=args.max_actions, max_searches=args.max_searches,
                             max_retries=args.max_retries, max_no_progress=args.max_no_progress,
                             fresh_s=args.fresh_seconds)
        validate_job(structure, "preflight", config)
        reasoner = None
        use_model = args.planner != "deterministic" and (not args.mock or args.allow_api_with_mock)
        from agent_backend import load_api_key
        key = load_api_key() if use_model else None
        if use_model and (key or args.base_url or args.planner == "llm"):
            if not key and not args.base_url:
                raise ValueError("LLM requires OPENAI_API_KEY or an explicitly configured compatible endpoint")
            from agent_backend import OpenAIReasoner
            reasoner = OpenAIReasoner(model=args.model, base_url=args.base_url,
                                     api_key=key or "unused", timeout=args.llm_timeout,
                                     json_only=args.json_only)
        kwargs = {}
        if args.mock:
            from mock_agent_world import MockAgentWorld
            world = MockAgentWorld(len(structure.blocks), voxel_size=config.voxel_size)
            providers = AgentProviders(world.actions, world.observations, world.close)
            kwargs = {"clock": world.clock, "sleep": world.sleep}
        else:
            providers = load_providers(args.provider)
        agent = Agent(providers.actions, providers.observations, config=config,
                      reasoner=reasoner, backend=args.planner, **kwargs)
        result = agent.run(structure)
        print(json.dumps(asdict(result), allow_nan=False), flush=True)
        return 0 if result.success else 1
    except (ValueError, CapabilityError, ImportError, AttributeError, TypeError) as exc:
        print(f"[agent] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        if providers is not None:
            providers.close()


def observe_perception(args):
    from agent_adapters import PerceptionObservations
    from agent_types import AgentConfig, Step
    try:
        with PerceptionObservations(args.observe_perception) as provider:
            deadline = time.monotonic()+4.0
            while True:
                try:
                    snapshot = provider.observe()
                    break
                except (TimeoutError, RuntimeError):
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.05)
            capabilities = asdict(provider.capabilities())
        observation = asdict(snapshot)
        observation.pop("world_model_json")
        observation["world_model"] = snapshot.world_model
        context = dict(observation, phase="PERCEPTION", execution_enabled=False,
                       inventory=observation["boxes"], confirmed=[], enabled_operations=["observe", "stop"],
                       perception_capabilities=capabilities)
        if args.structure:
            from agent import validate_job
            size = args.box_size if args.box_size is not None else .3
            job = validate_job(load_structure(args.structure), "perception-only",
                               AgentConfig(voxel_size=(size, size, size)))
            context["goal"] = asdict(job.requirements)
        decision = None
        if args.planner == "llm":
            from agent_backend import OpenAIReasoner, load_api_key
            key = load_api_key()
            if not key and not args.base_url:
                raise ValueError("LLM requires OPENAI_API_KEY or an explicitly configured compatible endpoint")
            reasoner = OpenAIReasoner(model=args.model, base_url=args.base_url, api_key=key or "unused",
                                     timeout=args.llm_timeout, json_only=args.json_only)
            choices = (Step("observe", "Inspect another live scene"), Step("stop", "Actions are not connected"))
            decision = reasoner.decide(context, choices)
        observation["images"] = [{k: v for k, v in asdict(image).items() if k != "data_url"}
                                 for image in snapshot.images]
        print(json.dumps({"execution_enabled": False, "observation": observation,
                          "perception_capabilities": capabilities,
                          "decision": asdict(decision) if decision else None}, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        print(f"[perception] {type(exc).__name__}: inspection failed; check the perception server and model configuration",
              file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
