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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from contracts import load_structure, cell_center, detections_to_dict
from skills_client import SkillsClient, MockSkills
from perception import scan_all
from planner import plan_build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", default=None)
    ap.add_argument("--grid-ui", action="store_true",
                    help="serve the click-grid until /save, then proceed")
    ap.add_argument("--planner", default="auto",
                    choices=["auto", "deterministic", "llm"])
    ap.add_argument("--skills-url", default="http://localhost:8006")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--sweeps", type=int, default=1,
                    help=">1 rotates the base between scans to find more boxes")
    a = ap.parse_args()

    skills = MockSkills() if a.mock else SkillsClient(a.skills_url)

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
    else:
        structure = load_structure(
            a.structure or str(Path(__file__).parent
                               / "fixtures" / "structure_house.json"))
    print(f"[orchestrator] target: {len(structure.blocks)} blocks, "
          f"{len(structure.layers())} layers", flush=True)

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


if __name__ == "__main__":
    main()
