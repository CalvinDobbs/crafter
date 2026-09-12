# [crafter]

Build a structure in Minecraft, send it to a robot.

The overall project plan (goals, sequencing, gaps, voice) is **[PLAN.md](PLAN.md)**. This README is how to build and run the Minecraft mod.

## Components

### `minecraft-mod/`

A Fabric mod for Minecraft 1.21.11. Adds a **Structure Scanner** item: right-click it and the mod
scans the area above the world origin, then sends every block it finds to the bot as a single JSON
payload over TCP.

Built against Fabric Loader 0.19.5, Fabric API 0.141.6+1.21.11, Java 21. The Gradle project and
mod id are still `posstream`, left over from an earlier iteration that streamed player position
instead of structures.

## The bot side

Python for the robot lives in [`bot-code/`](bot-code/) (planner, perception, orchestrator). Hardware
motion is a separate process, `mc_skills` on the BracketBot (see PLAN.md). The TCP receiver that
should listen on tcp/5005 is **not written yet**; until it is, ingest a `.json` / `.nbt` / grid-UI
file. Anything that consumes a live scan must follow [WIRE_FORMAT.md](WIRE_FORMAT.md).

## Building the mod

```bash
cd minecraft-mod
./gradlew build
```

The jar lands in `minecraft-mod/build/libs/`. Drop it in `.minecraft/mods` alongside Fabric API.

To run a dev client with the mod loaded, optionally aimed at a local receiver instead of the bot:

```bash
cd minecraft-mod
POSSTREAM_HOST=127.0.0.1 ./gradlew runClient
```

`POSSTREAM_HOST` defaults to the bot's address when unset.

## The world

The mod expects a superflat world whose surface sits at y=0, so the floor is never included in a
scan. Create one with this preset:

```
65*minecraft:gray_concrete;minecraft:plains
```

Then, in game:

```
/gamemode creative
/gamerule doMobSpawning false
/give @s posstream:structure_scanner
```
