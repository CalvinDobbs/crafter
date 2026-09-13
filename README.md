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

Python for the robot lives in [`bot-code/`](bot-code/) (planner, perception, orchestrator).
The `main.py --ui` panel receives Minecraft designs on TCP 5005 and previews them before Start.
Builds use real LLM calls with simulated tools; this mode does not execute hardware action scripts.
Hardware motion integration is separate (see PLAN.md). Live scans follow [WIRE_FORMAT.md](WIRE_FORMAT.md).

## Model key and developer access

### Store the key on the bot

The shared panel runs as `bracketbot` on `100.66.148.86`. Its API key belongs on that server,
not in Git, GitHub variables, chat, or a developer's checkout. Developers using the remote panel
do not need to retrieve the key or set `OPENAI_API_KEY` on their own PCs.

Connect with your approved SSH identity; [AGENTS.md](AGENTS.md) includes the configured Windows-PC
command. Do not share SSH private keys. Run the following in **your own interactive Bash terminal
on the bot**, as `bracketbot`. The prompt hides the key, and the value is not part of shell history:

```bash
set +x
read -r -s -p "OpenAI API key (hidden): " OPENAI_API_KEY
printf '\n'
export OPENAI_API_KEY
python -B /home/bracketbot/crafter/bot-code/main.py --save-api-key
unset OPENAI_API_KEY
```

If the key is already exported in that bot terminal, run just the `--save-api-key` command.
Saving creates `/home/bracketbot/.config/crafter/openai_api_key` with mode `600`, and a newly
created `crafter` directory has mode `700`. It refuses to overwrite an existing file. Coordinate
any key rotation with the server owner rather than deleting or replacing a working credential.

Panel startup uses `OPENAI_API_KEY` when set, otherwise it loads the saved file. The loader
rejects symlinks, non-regular files, incorrect ownership, and group/world-accessible files.
The saved key works across new SSH sessions. A panel that was already running must be restarted
by its owner to load a newly saved key; saving does not reconfigure that process automatically.
Never print the key, copy it into logs, or commit a key-containing file.

### Use the bot-hosted panel from a developer PC

On the bot, first check whether the panel or another receiver is already using its ports:

```bash
ss -ltnp '( sport = :8005 or sport = :5005 )'
```

Use an existing panel if it is already running; do not stop another team's service. If the ports
are free, start the panel on the bot and leave this terminal running:

```bash
uv run --offline /home/bracketbot/crafter/bot-code/main.py --ui
```

In a separate terminal **on your PC**, forward the panel using your approved SSH identity
(add the same `-i`/identity options you use to connect):

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8005:127.0.0.1:8005 bracketbot@100.66.148.86
```

Open **http://127.0.0.1:8005** locally and keep the tunnel running. The key and model requests
stay on the bot. Minecraft sends scans directly to the bot on TCP 5005 by default; the tunnel
above forwards only the browser panel. Review the preview before clicking Start: model calls
are real and billable, but physical tool execution is simulated. No build starts automatically.

GitHub Actions secrets are for workflow injection, not a downloadable developer key store:
`gh secret list` and the API expose metadata, not secret values. For model code running natively
on a developer PC instead of on the bot, obtain a separately authorized local-development key
from the project owner; do not use a workflow to print or export the server key.

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
