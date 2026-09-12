"""planner — Structure + detected boxes -> ordered Plan.

Two backends behind plan_build():
  - "deterministic": layer-by-layer, nearest-box assignment. Always works,
    no API key needed. This is the guaranteed demo path.
  - "llm": OpenAI chat call that reasons about assignment + writes narration
    lines. Falls back to deterministic on ANY error — never let the demo die
    on a bad API call.

IMPORTANT DESIGN RULE: the LLM chooses *which box goes to which cell*. It never
emits joint angles or positions — those come from cell_center() + IK inside
mc_skills. LLM = planner, not controller.
"""
from __future__ import annotations

import json
import os

import numpy as np

from contracts import (Action, Detection, Plan, Structure, cell_center,
                       plan_from_dict)

PROMPT = """You are the build planner for a robot arm that recreates Minecraft
builds with real cardboard cubes. Given the target structure (minecraft coords:
x,z horizontal, y = layer height) and the boxes the robot can see (id +
position in meters), output the ordered build plan.

Rules:
- Build bottom-up: a block at layer y>0 needs (x, y-1, z) placed earlier.
- Assign each structure block to a physical box id (each box used once).
- Prefer nearer boxes first; if 'color' is set, match it to the block kind.
- Also write one short, punchy pit-crew-style voice line per step
  (<=12 words, e.g. "dirt block, corner piece, going in hot").

Return ONLY json: {"steps": [{"box_id": int, "cell": [x,y,z], "say": str}]}

structure: {structure}
boxes: {boxes}
"""


def plan_deterministic(structure: Structure, boxes: list[Detection]) -> Plan:
    """Bottom-up fill; each cell gets the closest unused box."""
    order = sorted(structure.blocks, key=lambda b: (b.y, b.x, b.z))
    unused = {d.id: d for d in boxes}
    plan = Plan()
    for b in order:
        if not unused:
            break
        target = np.array(cell_center(b.x, b.y, b.z))
        best = min(unused.values(),
                   key=lambda d: float(np.linalg.norm(
                       np.array(d.pos)[:2] - target[:2])))
        del unused[best.id]
        plan.actions.append(Action("pick", box_id=best.id))
        plan.actions.append(Action("place", cell=(b.x, b.y, b.z)))
        plan.narration.append(f"{b.kind} block, layer {b.y}, going in")
    return plan


def plan_llm(structure: Structure, boxes: list[Detection],
             model="gpt-4o-mini") -> Plan:
    from openai import OpenAI
    client = OpenAI()  # OPENAI_API_KEY from env
    msg = PROMPT.format(
        structure=json.dumps([vars(b) for b in structure.blocks]),
        boxes=json.dumps([vars(d) for d in boxes]))
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": msg}],
        temperature=0.3)
    steps = json.loads(resp.choices[0].message.content)["steps"]

    # validate: cells legal, boxes exist, support ordering holds
    box_ids = {d.id for d in boxes}
    cells = {(b.x, b.y, b.z) for b in structure.blocks}
    placed: set[tuple[int, int, int]] = set()
    plan = Plan()
    for s in steps:
        cell = tuple(s["cell"])
        if s["box_id"] not in box_ids or cell not in cells:
            raise ValueError(f"bad step {s}")
        fake = type("B", (), {"x": cell[0], "y": cell[1], "z": cell[2]})
        if not Structure().supported(fake, placed):
            raise ValueError(f"unsupported cell {cell}")
        placed.add(cell)
        plan.actions.append(Action("pick", box_id=s["box_id"]))
        plan.actions.append(Action("place", cell=cell))
        plan.narration.append(s.get("say", ""))
    return plan


def plan_build(structure: Structure, boxes: list[Detection],
               backend="auto") -> Plan:
    if backend == "deterministic":
        return plan_deterministic(structure, boxes)
    try:
        return plan_llm(structure, boxes)
    except Exception as e:
        print(f"[planner] llm failed ({e}); deterministic fallback", flush=True)
        return plan_deterministic(structure, boxes)


if __name__ == "__main__":
    from contracts import load_structure, detections_from_dict
    import sys
    s = load_structure(sys.argv[1] if len(sys.argv) > 1
                       else "fixtures/structure_house.json")
    ds = detections_from_dict(json.load(open("fixtures/world_state.json")))
    print(json.dumps(__import__("contracts").plan_to_dict(
        plan_build(s, ds, backend=os.environ.get("BACKEND", "deterministic"))),
        indent=2))
