# Crafter UI and Minecraft receiver handoff

## Current state

- Branch: `main`
- Latest UI commit: `d217bfb` (`Unify the panel around a light red-accented theme`)
- Live panel: `http://127.0.0.1:8005/` through the PC-to-bot SSH forward
- Minecraft receiver: bot TCP `5005`, listening on `0.0.0.0`
- Bot panel process serves UI on bot loopback TCP `8005`
- The panel was verified live after the latest deployment; `/`, `/panel.css`, and `/panel.js` matched commit `d217bfb`.
- UI mode uses real model decisions and simulated tools. It does not execute robot hardware actions.

The bot checkout had concurrent, unrelated work at the last check:

- Modified: `bot-code/perception.py`
- Untracked: `bot-code/summaryofperception.md`

Do not overwrite, clean, stage, or discard those files. They were preserved during every bot update.

## What was diagnosed

Minecraft blueprint delivery was working, but the browser was pointed at the wrong panel:

- The Minecraft mod sends to `100.66.148.86:5005`.
- The main panel uses UI `8005` and receiver `5005`.
- An isolated preview used UI `8015` and receiver `5015`.
- The temporary browser URL `http://127.0.0.1:51282/` forwarded the isolated preview, so it never displayed scans sent to `5005`.

The correct panel URL is `http://127.0.0.1:8005/`. End-to-end verification sent a fragmented, EOF-framed five-block payload from the PC to bot TCP `5005`; it appeared as a valid Minecraft blueprint without starting a build.

## UI work completed

### Viewport layout

Commit: `e4aa925`

- The panel occupies the browser viewport without page-level scrolling.
- Long content scrolls inside bounded regions such as the activity feed.
- Preview canvases and primary controls remain visible in short, narrow, and mobile-sized windows.
- Build, failure, configuration, completion, notice, and unsupported-design states are covered by browser layout checks.

### Simplified design screen

Commit: `1b19856`

- Removed the **Latest design** card.
- Removed the **The demo setup** card.
- Expanded the schematic preview to the full available width.
- Kept Start, Clear blueprint, model-key configuration, receiver status, warnings, and the example control.

### Frontend organization and logo

Commit: `603ecf9`

Frontend files moved from loose files in `bot-code/` to:

```text
bot-code/web/
  panel.html
  panel.css
  panel.js
  assets/
    Crafter-transparent.svg
```

`bot-code/panel.py` remains the Python panel/receiver module and serves those assets. The top-left inline icon/text was replaced by `Crafter-transparent.svg`, served only through `/assets/Crafter-transparent.svg`.

### Light theme

Commit: `d217bfb`

- Converted all screens to a light, warm-neutral palette.
- Primary accent is muted brick red: `#b84946`.
- Updated buttons, badges, warnings, forms, activity cards, progress UI, completion UI, and native control color scheme.
- Updated the canvas renderer in `bot-code/web/panel.js` so preview backgrounds, grid lines, ghost blocks, and selection outlines also fit the light theme.
- Added minimum contrast checks for body text and white text on the primary accent.

## Important files

- `bot-code/panel.py` — FastAPI panel, API routes, and EOF-framed TCP structure receiver
- `bot-code/web/panel.html` — panel markup
- `bot-code/web/panel.css` — responsive light theme and viewport layout
- `bot-code/web/panel.js` — polling, rendering, interactions, and canvas previews
- `bot-code/web/assets/Crafter-transparent.svg` — header logo
- `bot-code/tests/test_panel.py` — receiver, session, HTTP, asset, theme, and browser-layout checks
- `minecraft-mod/src/main/java/com/calvin/posstream/StructureScannerItem.java` — scanner and TCP sender
- `WIRE_FORMAT.md` — Minecraft-to-bot framing and payload contract
- `AGENTS.md` — bot access, safety, deployment, and verification rules

## Verification

### Portable panel tests

From `bot-code/tests`:

```bash
PYTHONUTF8=1 uv run --offline --no-project python -B -m unittest test_panel -v
```

This runs stdlib-compatible tests and skips optional web/JavaScript checks if their dependencies are unavailable.

### Full bot-side panel tests

Use the bot's installed dependency cache:

```bash
/home/bracketbot/.local/bin/uv run --offline --no-project \
  --with fastapi --with uvicorn --with httpx --with openai --with quickjs==1.19.4 \
  python -B -m unittest discover -s /home/bracketbot/crafter/bot-code/tests \
  -p test_panel.py -v
```

Latest result: 29 tests passed; only the optional Chromium layout test was skipped on the bot.

### Chromium viewport/theme test

The test requires Node 22+ and an isolated Chromium/Edge CDP endpoint. Never use a personal signed-in browser profile.

```bash
cd bot-code/tests
PYTHONUTF8=1 CRAFTER_LAYOUT_CDP=http://127.0.0.1:<port> \
  uv run --offline --no-project python -B -m unittest \
  test_panel.PanelAssetsTests.test_screens_fit_the_browser_viewport -v
```

Latest result: all 80 viewport/screen combinations passed. The check covers page overflow, clipped controls, canvas visibility, loaded logo, light native controls, light canvas/input backgrounds, full-width design preview, and bounded activity feed.

## Bot deployment workflow

Use SSH from the Windows PC as documented in `AGENTS.md`. Before updating:

1. Inspect `/home/bracketbot/crafter` status.
2. Inspect listeners on `8005`, `5005`, and `8006`.
3. Preserve all unrelated modified/untracked files.
4. Confirm that incoming commits do not modify an already-modified bot file before fast-forwarding.
5. Run panel tests on the bot.
6. Verify `/api/state` reports `receiver.listening: true` and receiver port `5005`.

Static HTML/CSS/JS changes are read from disk per request and normally do not require restarting the panel. Changes to `panel.py` or asset paths require restarting only the idle panel after explicit approval. Never restart or interfere with the hardware or perception services.

## Operational notes

- The bot/Tailscale SSH connection was intermittently slow and occasionally timed out. Retry with a longer `ConnectTimeout` rather than changing SSH configuration.
- Browser preview URLs can hide the backend port. Always inspect `/api/state`; for default Minecraft scans, the receiver must report port `5005`.
- Minecraft uses UTF-8 JSON framed by TCP EOF/half-close. The receiver must read until EOF, not assume one `recv()` call is a complete message.
- A successful Minecraft chat message only confirms that bytes were flushed. Receiver rejection details appear in panel state as a notice.
- Do not run real hardware or paid model calls for routine verification.
- `contracts.py` and `WIRE_FORMAT.md` remain frozen.
