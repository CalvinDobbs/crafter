"""Speak-order for each mapped command. Used by MOCK tests and by patched mc_skills."""
from __future__ import annotations

from hook import narrate


def pick(_pos=None, ctx=None) -> list[str]:
    # Body.pick: gripper(True) silent, then:
    seq = ["pick.approach", "pick.descend", "pick.grasp", "pick.lift"]
    for e in seq:
        narrate(e, ctx)
    return seq


def place(_pos=None, ctx=None) -> list[str]:
    seq = ["place.approach", "place.descend", "place.release", "place.retreat"]
    for e in seq:
        narrate(e, ctx)
    return seq


def home() -> None:
    narrate("home.start")


def park() -> None:
    return None


def rotate(_rad=None) -> None:
    narrate("scan.start")


def celebrate() -> None:
    narrate("build.done")


def gripper(open_: bool) -> None:
    narrate("place.release" if open_ else "pick.grasp")


def fail() -> None:
    narrate("fail")
