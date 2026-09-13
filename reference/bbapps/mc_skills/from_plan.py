"""Turn a decided Plan into phase events, then narrate. Does not choose actions."""
from __future__ import annotations

import json
from pathlib import Path

from narrator import line

FIXTURE = Path(__file__).parent / "fixture_plan.json"

PICK_PHASES = ("pick.approach", "pick.descend", "pick.grasp", "pick.lift")
PLACE_PHASES = ("place.approach", "place.descend", "place.release", "place.retreat")


def events_from_plan(plan: dict) -> list[tuple[str, dict]]:
    """plan: {actions: [{kind, box_id?, cell?}]} — already decided pick/place."""
    actions = plan.get("actions") or []
    n_place = sum(1 for a in actions if a.get("kind") == "place")
    out: list[tuple[str, dict]] = [
        ("scan.start", {}),
        ("home.start", {}),
        ("plan.ready", {"n": n_place}),
    ]
    pending_id = None
    for a in actions:
        kind = a.get("kind")
        if kind == "pick":
            pending_id = a.get("box_id")
            ctx = {"id": pending_id}
            for ev in PICK_PHASES:
                out.append((ev, ctx))
        elif kind == "place":
            cell = a.get("cell") or [0, 0, 0]
            y = cell[1]
            ctx = {"id": pending_id, "y": y, "cell": cell}
            for ev in PLACE_PHASES:
                out.append((ev, ctx))
            pending_id = None
        else:
            out.append(("fail", {"kind": kind}))
    out.append(("build.done", {}))
    return out


def narrate(plan: dict) -> list[tuple[str, dict, str]]:
    rows = []
    for event, ctx in events_from_plan(plan):
        rows.append((event, ctx, line(event, ctx)))
    return rows


def _dump(plan: dict) -> None:
    for event, ctx, text in narrate(plan):
        shown = repr(text) if text else "(skip)"
        extra = ""
        if "id" in ctx and ctx["id"] is not None:
            extra += f" id={ctx['id']}"
        if "y" in ctx:
            extra += f" y={ctx['y']}"
        print(f"{event:16}{extra:16} {shown}")


if __name__ == "__main__":
    plan = json.loads(FIXTURE.read_text())
    _dump(plan)
