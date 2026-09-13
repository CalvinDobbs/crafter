"""8.5: MOCK pick sequence (no FastAPI, no robot)."""
from __future__ import annotations

import hook
import moves


def main() -> None:
    hook.log.clear()
    seq = moves.pick([0.4, 0.0, 0.05], {"id": 3})
    print("commands", seq)
    print("log      ", hook.log)
    ok = seq == ["pick.approach", "pick.descend", "pick.grasp", "pick.lift"]
    ok = ok and hook.log == seq
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
