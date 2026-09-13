"""Back-compat: python3 queue.py still runs the 5.8 immediate-return check."""
from __future__ import annotations

import time

from speech_queue import DO_NOT_HOLD_MOTION_LOCK, SpeechQueue


def _demo() -> None:
    q = SpeechQueue(play_s=0.3)
    t0 = time.perf_counter()
    d1 = q.say("pick.approach", {"id": 3})
    time.sleep(0.4)
    d2 = q.say("pick.grasp", {"id": 3})
    elapsed = time.perf_counter() - t0
    time.sleep(0.5)
    print(f"say1={d1:.4f}s say2={d2:.4f}s fake_pick={elapsed:.3f}s")
    print("started", q.started)
    print("DO_NOT_HOLD_MOTION_LOCK", DO_NOT_HOLD_MOTION_LOCK)
    ok = d1 < 0.05 and d2 < 0.05 and elapsed < 0.5
    print("PASS" if ok else "FAIL")
    q.close()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    _demo()
