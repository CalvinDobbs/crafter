"""Neutral event → spoken line. Personality packs come later."""
from __future__ import annotations

MAX_WORDS = 8

# Empty string = skip (do not speak).
NEUTRAL: dict[str, str] = {
    "scan.start": "Looking for boxes.",
    "plan.ready": "{n} blocks to place.",
    "home.start": "Waking the arms up.",
    "pick.approach": "Going to box {id}.",
    "pick.grasp": "Picking it up.",
    "pick.lift": "Got it.",
    "place.approach": "Placing on layer {y}.",
    "place.release": "Setting it down.",
    "place.retreat": "",
    "build.done": "Build complete.",
    "fail": "That miss, trying the next one.",
}

EVENTS = (
    "scan.start",
    "plan.ready",
    "home.start",
    "pick.approach",
    "pick.grasp",
    "pick.lift",
    "place.approach",
    "place.release",
    "place.retreat",
    "build.done",
    "fail",
)

_DEMO_CTX = {
    "plan.ready": {"n": 4},
    "pick.approach": {"id": 3},
    "place.approach": {"y": 1},
}


class _Fmt(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def line(event: str, ctx: dict | None = None) -> str:
    """Return a <=8-word neutral line, or '' to skip."""
    tmpl = NEUTRAL.get(event)
    if tmpl is None:
        return ""
    if not tmpl:
        return ""
    text = tmpl.format_map(_Fmt(ctx or {})).strip()
    if len(text.split()) > MAX_WORDS:
        text = " ".join(text.split()[:MAX_WORDS])
    return text


def _dump() -> None:
    for event in EVENTS:
        text = line(event, _DEMO_CTX.get(event))
        words = len(text.split()) if text else 0
        shown = repr(text) if text else "(skip)"
        print(f"{event:16} {words}w  {shown}")


if __name__ == "__main__":
    _dump()
