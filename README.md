# Crafter

**You build it in Minecraft. Crafter builds it in real life.**

*A hackathon project — built in under 24 hours.*

Crafter is a robot that turns virtual block designs into physical structures. Design a build
in Minecraft, click a button, and a BracketBot finds real cardboard boxes, works out which one
belongs where, and stacks them until the pile matches your blueprint.

You design the result. The robot handles everything between the blueprint and the floor.

## How it works

```text
Minecraft mod  ──▶  Blueprint  ──▶  Reasoning agent  ──▶  Perception  ──▶  Arms + base
 scan a build       JSON over        picks the next       finds boxes      pick and place
 with one click     TCP:5005         block and cell       in 3D
```

1. **Scan.** A Fabric mod adds a Structure Scanner item. Right-click it and the mod encodes
   every block above the world origin as a machine-readable blueprint.
2. **See.** The robot surveys the room with its stereo camera, turning colour and depth into
   box detections in its own coordinate frame.
3. **Reason.** An LLM decides which physical box goes in which grid cell, building bottom-up so
   every block is supported.
4. **Build.** A tool-calling loop drives real pick-and-place actions, one block at a time, and
   feeds the result back so the plan can adapt when a box slips.

## Repo

| Path | What's in it |
| --- | --- |
| [`minecraft-mod/`](minecraft-mod/) | Fabric mod for Minecraft 1.21.11 — the Structure Scanner |
| [`bot-code/`](bot-code/) | Robot app: reasoning agent, perception, planner, actions, voice, control panel |
| [`WIRE_FORMAT.md`](WIRE_FORMAT.md) | The blueprint format on the wire |

## Quick start

### Build the mod

```bash
cd minecraft-mod
./gradlew build
```

The jar lands in `minecraft-mod/build/libs/`. Drop it in `.minecraft/mods` next to Fabric API
(Fabric Loader 0.19.5, Fabric API 0.141.6+1.21.11, Java 21).

To run a dev client aimed at a local receiver instead of the robot:

```bash
POSSTREAM_HOST=127.0.0.1 ./gradlew runClient
```

### Set up the world

Crafter expects a superflat world whose surface sits at y=0, so the floor never ends up in a scan:

```
65*minecraft:gray_concrete;minecraft:plains
```

Then, in game:

```
/gamemode creative
/gamerule doMobSpawning false
/give @s posstream:structure_scanner
```

### Run the panel

The panel receives designs on TCP 5005, previews them, and starts builds. **It runs on the robot,
not on your laptop** — the cameras, the arms and the saved API key all live there.

```bash
# on the bot
uv run --offline /home/bracketbot/crafter/bot-code/main.py --ui

# on your PC
ssh -N -L 127.0.0.1:8005:127.0.0.1:8005 bracketbot@100.66.148.86
```

Open <http://127.0.0.1:8005>, review the preview, then press Start. Nothing builds on its own.

## Status

Perception, reasoning and the build loop run end to end against an offline simulator, so a build
needs no model, no API key and no robot. Physical motion is documented and partly implemented,
but no live action provider is wired up yet — a hardware build fails preflight by design.

## What's next

- More than cardboard: a wider range of block types, colours and textures
- Closed-loop visual feedback so Crafter spots a fallen block and fixes it mid-build
- Larger structures, built alongside you while you're still designing in-game

## More docs

[`bot-code/README.md`](bot-code/README.md) is the robot architecture and runbook.
[`AGENTS.md`](AGENTS.md) has the operational rules, API-key handling and verified commands.
[`PLAN.md`](PLAN.md) is the goals-and-sequencing plan.
