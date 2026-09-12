# Voice Plan

Live voice checklist (this file only). Do not push. Team overview stays in `PLAN.md`.

## Goal
Test: given a **decided action/phase**, speak a short **neutral** line while that action runs; `/say` never blocks arms.
Final: same pipeline; personality + LLM **wording** after 6.5 (e.g. startup “it’s boxing time!”).
Success: 5.15 approved (test voice); 7.9 approved (final).

## Context
Architecture: `PLAN.md`. Voice root: `bot-code/voice/`.
**Split:** planner/`mc_skills` **choose the action** (pick box N, place cell, grasp…). Voice **only narrates** that choice. 5.1 is a template renderer — not wired to the robot.
Pipeline: `Plan`/`Action` → phase event `{event, ctx}` → `line()` (now) → LLM wording (7.7) → TTS.
Events: `scan.start` `plan.ready` `home.start` `pick.approach|grasp|lift` `place.approach|release` `build.done` `fail`.
`ctx` carries facts from the plan (`id`, `y`, `n`), not invented by voice.

## Limits
- **Voice folder only:** all voice code/docs/research in `bot-code/voice/`. Do not edit files outside it. No hooks into `main.py` / `mc_skills` until an explicit approved integration task.
- Voice does **not** decide motion. No LLM choosing pick/place/joints.
- **No robot copy** until the user says to deploy.
- Gate: Test + Approval on each ID before the next. Record Approval blocked until user sign-off.
- No personality/API/LLM **speech** before 6.5. Neutral templates until then.
- No second `speaker.audio` Writer. `/say` must not take `_busy`.
- Keep changes small, localized, and reversible.

## Current State
- Current: 5.3
- Blocked: 5.3 — wait for user; do not rsync/copy to robot

## 5) Neutral voice (testing)
- [x] 5.1 `bot-code/voice/narrator.py`: `line(event, ctx) -> str`; `neutral` only; `__main__` dumps events
- [x] 5.2 Test: `python3 bot-code/voice/narrator.py`
- [ ] 5.3 Approval: 5.2
- [ ] 5.4 From a mock `Plan` (pick/place + box id/cell), emit phase events in order, then `line()` — speech follows planned actions, still in `bot-code/voice/`
- [ ] 5.5 Test: fixture plan → event sequence matches pick then place; lines use that `id`/`y`
- [ ] 5.6 Approval: 5.5
- [ ] 5.7 Speech queue in `bot-code/voice/`: enqueue, immediate return; document `_busy` rule for later `/say`
- [ ] 5.8 Test: enqueue during a fake long pick; call returns at once
- [ ] 5.9 Approval: 5.8
- [ ] 5.10 Drive 5.4 events off mock pick/place **phases** (approach before grasp); 1+1 queue; preempt
- [ ] 5.11 Test: approach logged before grasp for the same box id
- [ ] 5.12 Approval: 5.11
- [ ] 5.13 Local TTS module in `bot-code/voice/`; cache; fail-open; PCM framed for `speaker.audio`
- [ ] 5.14 Test: play/save wav locally; TTS down still returns
- [ ] 5.15 Approval: 5.14

## 6) Research — no product code
- [ ] 6.1 Online voice APIs → `bot-code/voice/voice_research.md`
- [ ] 6.2 Personality-in-voice (IDs, SSML, catchphrase)
- [ ] 6.3 LLM **wording** given `{event, ctx}` (not action selection); timing, prompt, timeout, fallback
- [ ] 6.4 Test: comparison table + recommended default/fallback
- [ ] 6.5 Approval: 6.4 — required before 7.x

## 7) Personality voice
- [ ] 7.1 `VoicePack` in `bot-code/voice/`; startup “it’s boxing time!”
- [ ] 7.2 Test: pack switch; `neutral` remains
- [ ] 7.3 Approval: 7.2
- [ ] 7.4 Approved TTS API behind voice module; local fail-open
- [ ] 7.5 Test: one live say; disconnect still returns
- [ ] 7.6 Approval: 7.5
- [ ] 7.7 Pre-plan LLM lines **from already-decided events**; never block caller; templates if LLM fails
- [ ] 7.8 Test: LLM down → templates; event `ctx` still matches the mock plan
- [ ] 7.9 Approval: 7.8

## Handoff
- Decisions: Action identity from planner/phases, not from voice/LLM. 5.1 = renderer only. Tie-in starts at 5.4 (mock Plan → events). LLM wording = 6.3/7.7 after 6.5. Folder-only until approved integration.
- Verified: laptop dump only. Robot `narrator.py` removed after unsolicited copy.
- Open: 5.3. Do not copy to the robot until asked.
- Next: 5.3 — user approval. Do not start 5.4; do not rsync.
