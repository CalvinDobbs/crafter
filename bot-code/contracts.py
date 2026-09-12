"""Shared data model for the minecraft-build app. Pure python — NO bbos import,
so every module (and off-robot laptops) can use it.

Coordinate conventions:
- Minecraft coords: (x, z) horizontal plane, y = height/layer (0 = ground).
- Robot base frame: +x forward, +y left, +z up, meters (matches camera.points).
- The build grid is an axis-aligned region on the work surface in front of the
  robot; GRID_ORIGIN is the base-frame position of cell (x=0, z=0) at layer 0
  (i.e. the CENTER of where a box bottom sits — so add half box height for the
  box center when placing).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


# ---- tunables (edit once at the table) -------------------------------------

BOX_SIZE = 0.05          # m, cardboard cube edge length
CELL = BOX_SIZE          # grid pitch = one box
# base-frame position of the center of cell (0,0) at layer 0, at table height:
GRID_ORIGIN = [0.40, 0.0, 0.05]   # x fwd, y left, z up — TUNE ON ROBOT
COL_DIR = [0.0, 1.0, 0.0]  # minecraft +x runs along base +y (left)
ROW_DIR = [1.0, 0.0, 0.0]  # minecraft +z runs along base +x (away)


def cell_center(x: int, y_layer: int, z: int) -> list[float]:
    """Minecraft cell -> base-frame position of the box CENTER."""
    return [
        GRID_ORIGIN[0] + COL_DIR[0] * x * CELL + ROW_DIR[0] * z * CELL,
        GRID_ORIGIN[1] + COL_DIR[1] * x * CELL + ROW_DIR[1] * z * CELL,
        GRID_ORIGIN[2] + y_layer * CELL + BOX_SIZE / 2.0,
    ]


# ---- types ------------------------------------------------------------------

@dataclass
class Block:
    x: int
    y: int            # layer, 0 = on the table
    z: int
    kind: str = "cube"     # minecraft block name or "cube"


@dataclass
class Structure:
    """The target: what the player built in-game."""
    blocks: list[Block] = field(default_factory=list)

    def layers(self) -> list[list[Block]]:
        out: dict[int, list[Block]] = {}
        for b in self.blocks:
            out.setdefault(b.y, []).append(b)
        return [out[k] for k in sorted(out)]

    def supported(self, b: Block, placed: set[tuple[int, int, int]]) -> bool:
        return b.y == 0 or (b.x, b.y - 1, b.z) in placed


@dataclass
class Detection:
    """A physical box found by perception."""
    id: int
    pos: list[float]          # base-frame xyz of box center
    color: str | None = None  # optional tag/color label
    size: float = BOX_SIZE


@dataclass
class Action:
    """One step the executor can run. kind: 'pick' | 'place'."""
    kind: str
    box_id: int | None = None          # pick: which Detection
    cell: tuple[int, int, int] | None = None  # place: minecraft (x,y,z)


@dataclass
class Plan:
    actions: list[Action] = field(default_factory=list)
    narration: list[str] = field(default_factory=list)  # per-action voice lines


# ---- json io ----------------------------------------------------------------

def _block(d) -> Block: return Block(**d)
def _det(d) -> Detection: return Detection(**d)


def structure_from_dict(d: dict) -> Structure:
    return Structure(blocks=[_block(b) for b in d["blocks"]])


def structure_to_dict(s: Structure) -> dict:
    return {"blocks": [asdict(b) for b in s.blocks]}


def load_structure(path: str) -> Structure:
    with open(path) as f:
        return structure_from_dict(json.load(f))


def save_structure(s: Structure, path: str) -> None:
    with open(path, "w") as f:
        json.dump(structure_to_dict(s), f, indent=2)


def detections_from_dict(d: dict) -> list[Detection]:
    return [_det(x) for x in d["detections"]]


def detections_to_dict(ds: list[Detection]) -> dict:
    return {"detections": [asdict(x) for x in ds]}


def plan_from_dict(d: dict) -> Plan:
    acts = [Action(kind=a["kind"], box_id=a.get("box_id"),
                   cell=tuple(a["cell"]) if a.get("cell") else None)
            for a in d["actions"]]
    return Plan(actions=acts, narration=d.get("narration", []))


def plan_to_dict(p: Plan) -> dict:
    return {"actions": [asdict(a) for a in p.actions], "narration": p.narration}
