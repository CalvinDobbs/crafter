# Reference material

Vendored read-only snapshots. Nothing here is built, imported, or run by this project.

## `bbapps/`

Copied from `bracketbot@100.66.148.86:~/bbapps` on 2026-09-12 — the BracketBot stock app
collection, kept here as the reference for how to talk to the robot.

Start with [bbapps/AGENTS.md](bbapps/AGENTS.md): the `bbos` `Reader`/`Writer` IPC model, the full
topic catalog (`drive.ctrl`, `arm_left.ctrl`, `camera.*`, `slam.pose`, `speaker.audio`, ...), and
the PEP 723 header every bot app needs. [bbapps/examples/](bbapps/examples/) has one short script
per subsystem; `quest_teleop/scripts/homing.py` and `inference/bracketbot_adapter.py` cover arm
homing and joint normalization.

Excluded from the copy: generated TTS `.wav` caches (~23 MB under `mc_skills/wavs`,
`play_sound/wavs`, `mc_skills/.tts_cache`) and `__pycache__`. Re-sync with:

```bash
ssh bracketbot@100.66.148.86 'cd ~/bbapps && tar -cz --exclude=wavs --exclude=__pycache__ --exclude=.tts_cache --exclude="*.pyc" .' | tar -xz -C reference/bbapps
```
