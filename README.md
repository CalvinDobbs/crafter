# crafter

Build a structure in Minecraft, send it to a robot.

## Components

### `minecraft-mod/`

A Fabric mod for Minecraft 1.21.11. Adds a **Structure Scanner** item: right-click it and the mod
scans the area above the world origin, then sends every block it finds to the bot as a single JSON
payload over TCP.

Built against Fabric Loader 0.19.5, Fabric API 0.141.6+1.21.11, Java 21. The Gradle project and
mod id are still `posstream`, left over from an earlier iteration that streamed player position
instead of structures.

## The bot side

Not in this repo yet. The receiver runs on the BracketBot (Ubuntu, aarch64), listens on tcp/5005,
and is written against [WIRE_FORMAT.md](WIRE_FORMAT.md) — the contract covering TCP framing, the
JSON schema, coordinate conventions, and the edge cases a receiver has to handle. Read that before
writing anything that consumes a structure.

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
