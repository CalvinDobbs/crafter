"""Event → spoken line. Packs swap personality; flavor.prewrite can override.

Names from bracketbot-184 (bbos daemons / mc_skills), not nicknames:
  base.mode 0 = MODE_BALANCE (2-wheel self-balancing drive)
  J0 / lj0,rj0 = vertical stage (slide up/down; j0_homing)
  J1 = frontward/back (clearance); IK bends the 7 arm joints
  J7 / gripper = claws (index 7; right_left_gripper / left_left_gripper)
  pick: gripper open, goto APPROACH_H, descend, gripper close, lift
  place: goto above, descend, gripper open, retreat
  home: staged_home_arms (torque enable + J0 homing)
"""
from __future__ import annotations

MAX_WORDS = 8

# Empty string = skip (do not speak). Default / neutral pack templates.
NEUTRAL: dict[str, str] = {
    "scan.start": "Head camera scanning for boxes.",
    "plan.ready": "{n} blocks to place.",
    "home.start": "Staged home, J0 homing.",
    "pick.approach": "J0 slide toward box {id}.",
    "pick.descend": "Descending to the box.",
    "pick.grasp": "Closing the J7 gripper.",
    "pick.lift": "J0 vertical stage lifting.",
    "place.approach": "Arm bending to layer {y}.",
    "place.descend": "Descending to the cell.",
    "place.release": "Opening the J7 gripper.",
    "place.retreat": "Retreating to approach height.",
    "build.done": "Done. Still in balance mode.",
    "fail": "Missed that. Gripper opening.",
}

EVENTS = (
    "scan.start",
    "plan.ready",
    "home.start",
    "pick.approach",
    "pick.descend",
    "pick.grasp",
    "pick.lift",
    "place.approach",
    "place.descend",
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
    """Return a short spoken line, or '' to skip.

    Order: ahead-of-time script override → active pack template → NEUTRAL.
    """
    max_words = MAX_WORDS
    try:
        from packs import get_pack
        max_words = getattr(get_pack(), "max_words", MAX_WORDS)
    except Exception:
        pass

    try:
        from flavor import lookup
        override = lookup(event, ctx)
        if override:
            text = override.strip()
            if len(text.split()) > max_words:
                text = " ".join(text.split()[:max_words])
            return text
    except Exception:
        pass

    tmpl = None
    try:
        from packs import get_pack
        pack = get_pack()
        tmpl = pack.templates.get(event)
    except Exception:
        tmpl = None

    if tmpl is None:
        tmpl = NEUTRAL.get(event)
    if tmpl is None or not tmpl:
        return ""
    text = tmpl.format_map(_Fmt(ctx or {})).strip()
    if len(text.split()) > max_words:
        text = " ".join(text.split()[:max_words])
    return text


def _dump() -> None:
    for event in EVENTS:
        text = line(event, _DEMO_CTX.get(event))
        words = len(text.split()) if text else 0
        shown = repr(text) if text else "(skip)"
        print(f"{event:16} {words}w  {shown}")


if __name__ == "__main__":
    _dump()
