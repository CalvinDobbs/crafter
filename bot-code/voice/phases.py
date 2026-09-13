"""Mock pick/place phases: say() at each real mc_skills step, then sleep (motion)."""
from __future__ import annotations

import time

from speech_queue import SpeechQueue

# (event, mock motion seconds) — matches Body.pick / Body.place order
PICK_STEPS = (
    ("pick.approach", 0.12),  # goto above + descend
    ("pick.grasp", 0.12),     # gripper(False) close J7
    ("pick.lift", 0.12),      # goto above
)
PLACE_STEPS = (
    ("place.approach", 0.12),
    ("place.release", 0.12),  # gripper(True) open J7
    ("place.retreat", 0.12),
)


def run_pick(q: SpeechQueue, box_id: int) -> list[str]:
    ctx = {"id": box_id}
    order = []
    for event, dur in PICK_STEPS:
        q.say(event, ctx)
        order.append(event)
        time.sleep(dur)
    return order


def run_place(q: SpeechQueue, box_id: int, y: int) -> list[str]:
    ctx = {"id": box_id, "y": y}
    order = []
    for event, dur in PLACE_STEPS:
        q.say(event, ctx)
        order.append(event)
        time.sleep(dur)
    return order


def _demo() -> None:
    q = SpeechQueue(play_s=0.08)
    cmd = run_pick(q, 3)
    time.sleep(0.3)
    print("commands", cmd)
    print("started ", q.started)
    i_app = next(i for i, t in enumerate(q.started) if "box 3" in t)
    i_grip = next(i for i, t in enumerate(q.started) if "Closing the J7" in t)
    order_ok = cmd[:2] == ["pick.approach", "pick.grasp"]
    speak_ok = i_app < i_grip
    print("PASS" if order_ok and speak_ok else "FAIL")
    q.close()
    raise SystemExit(0 if order_ok and speak_ok else 1)


if __name__ == "__main__":
    _demo()
