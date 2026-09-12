# actions — bimanual pickup with web control

Single file: `pickup.py`. Nothing moves until you click a button.

```bash
# stop any other arm owner first (mc_skills, quest_teleop, mimic) — one writer per topic
uv run ~/crafter/bot-code/actions/pickup.py            # open http://<robot-ip>:8009
uv run ~/crafter/bot-code/actions/pickup.py --mock     # UI only, no hardware
```

## Phases (joint space, both arms together)

1. **Down** — J0 vertical stage descends (`down_frac`, 1.0 = grippers at floor level)
2. **Out** — J2 swings both arms sideways away from the body centre (`out_deg`)
3. **Pinch** — J2 swings back inward past neutral (`pinch_deg`) and grippers close
4. **Raise** — J0 back to the top (shoulder level) while still pinching

"Run pickup" executes 1→4 with a pause between phases; each phase also has its
own button so you can step through manually. The final pose is held until
**Release** (torque off, arms limp) or the process exits.

## UI

- **Home arms** — staged torque enable + home (required before any phase)
- **Stop** — cuts the current ramp short; arms hold where they are
- **Arms neutral / Grip open / Grip close** — single-joint helpers for tuning
- **Behaviour panel** — every parameter above plus J0 speed, swing time and
  inter-phase pause; **Apply** takes effect on the next phase you trigger

## Tune on the robot

- If **2 Out** moves the arms toward each other, set *Out direction* to -1.
- If the grippers open when they should close, swap `grip_open`/`grip_closed`
  in `Arm.__init__` (they come from `ranges.calibration.json`).
- Raise `down_frac` toward 1.0 only if the grippers stop above the box.
