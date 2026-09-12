# Structure Wire Format

Everything a receiver needs to accept a build from the Structure Scanner item. One right-click in
Minecraft opens one TCP connection, writes one JSON object, and hangs up. Your job is to accept,
read to EOF, and parse.

| | |
|---|---|
| Transport | TCP, IPv4 |
| Listen port | 5005 |
| Direction | Game → bot only |
| Encoding | UTF-8 JSON |
| Framing | EOF (half-close) |

## The exchange

The game is always the client; it connects outbound each time. There is no handshake, no header,
and no keep-alive — the connection exists for exactly one structure.

1. **Connect** — the game dials your listener with a **1500 ms** connect timeout. Nothing is sent
   if the connect fails.
2. **Write** — the complete JSON object, UTF-8 encoded. No length prefix, no newline, no trailing
   byte of any kind.
3. **Half-close** — the game calls `shutdownOutput()`. This is your end-of-message signal: your
   read returns EOF.
4. **Close** — the socket closes immediately. The game never reads from it, so anything you write
   back is discarded.

> **The one thing to get right.** Because the message is delimited by EOF and not by a length, a
> single `recv()` is not a message. Read until the stream ends, then parse. A 400-byte structure
> will usually arrive in one packet and lull you into thinking otherwise; a 40 KB one will not.

## A real payload

Verbatim capture: four oak planks placed on the grey concrete at spawn, sent and received.
Whitespace added here for reading — the wire form has none.

```json
{
  "origin":  [0, 1, -1],
  "size":    [1, 2, 3],
  "count":   4,
  "palette": ["minecraft:oak_planks"],
  "blocks":  [[0,0,0,0], [0,0,1,0], [0,1,1,0], [0,0,2,0]]
}
```

That describes a three-block strip running along Z with a fourth block stacked on its middle — an
upside-down T, one block thick.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `origin` | `[int, int, int]` | Absolute Minecraft world coordinates of the bounding box's minimum corner, as `[x, y, z]`. Add this to any block's coordinates to get world space. |
| `size` | `[int, int, int]` | Bounding box extent in blocks, `[width x, height y, depth z]`. Always ≥ 1 on every axis. This is the box, not the block count — the build inside it is usually sparse. |
| `count` | `int` | Number of entries in `blocks`. Redundant by design — use it to sanity-check a truncated read. |
| `palette` | `string[]` | Distinct namespaced block ids, e.g. `minecraft:oak_planks`. Ordered by first appearance during the scan, so index 0 is whatever was found first, not anything semantic. |
| `blocks` | `[int,int,int,int][]` | `[x, y, z, paletteIndex]` per block. Coordinates are **relative to `origin`** and always non-negative. Every index is a valid position in `palette`. |

## Coordinates

Minecraft axes, unchanged: **+X is east, +Z is south, +Y is up**. All values are whole blocks. Y is
the vertical axis — if your robot's frame calls that Z, you are the one who has to swap it.

Block coordinates are relative so a build is translation-independent; `origin` is carried along for
when you do need to know where in the world it sat. The conversion, using the capture above:

```
relative  + origin [0, 1, -1] →  world
[0, 0, 0]                     →  ( 0, 1, -1)
[0, 0, 1]                     →  ( 0, 1,  0)
[0, 1, 1]                     →  ( 0, 2,  0)
[0, 0, 2]                     →  ( 0, 1,  1)
```

### Ordering you can rely on

The scan walks X outermost, then Z, then Y innermost, so `blocks` arrives sorted ascending by x,
then z, then y. Columns therefore arrive bottom-to-top and contiguously, which is convenient if you
are stacking. Treat this as a property of the current scanner, not a promise — sort it yourself if
your algorithm depends on it.

## What the scanner captures

The item scans a fixed box centred on the world origin and keeps every non-air block: X and Z from
−64 to +64, Y from 1 to 64. That is 1,065,024 positions tested per right-click.

Y starts at 1 because the superflat preset puts the grey concrete surface at exactly y=0, so the
floor is never in the payload. Build above y=0 and near the origin.

### Deliberately absent

- **Block states.** Only the block id survives — stair direction, slab half, door hinge,
  waterlogging and rotation are all dropped. A staircase arrives as a set of
  `minecraft:oak_stairs` positions with no facing.
- **The floor**, and anything at or below y=0.
- **Entities** — mobs, item frames, armour stands, paintings. Blocks only.
- **Anything outside the box**, including a build that straddles the ±64 boundary, which is
  silently clipped.
- **Block entity contents.** A chest is a `minecraft:chest` position; its items are not sent.

## Edge cases worth handling

- **An empty scan sends nothing at all.** If no blocks are found, the game reports that in chat and
  never opens a connection. You will not receive an empty `blocks` array, so don't code for one as
  your "nothing there" signal.
- **Unloaded chunks read as air.** The scan reads the client's view of the world. Blocks in chunks
  outside render distance come back empty and simply won't appear — stand near the build.
- **Accept promptly.** The 1500 ms connect timeout runs on Minecraft's main thread. A listener with
  a full backlog queue doesn't just fail the send, it stutters the game for up to a second and a
  half.
- **Sends can overlap.** Nothing rate-limits the right-click. Hold it down and you'll get several
  connections in flight; handle each independently or serialise them yourself.
- **Success in chat is not success on your end.** `Sent 4 blocks (118 bytes)` means the bytes were
  flushed to the socket, nothing more. If your parser throws, the game will never know — log on
  your side.
- **Expect unknown block ids.** Any block placeable in creative can appear. Decide up front whether
  an unrecognised id is an error or just an occupied voxel.

## Minimal receiver

The shape to copy, in any language: accept, read to EOF, parse, handle. This is the whole contract.

```python
# read-to-EOF is the part that matters; everything else is yours
import json, socket

# backlog matters: the game only waits 1500 ms for the connect
srv = socket.create_server(("0.0.0.0", 5005), backlog=8)

while True:
    conn, addr = srv.accept()
    with conn:
        chunks = []
        while True:
            part = conn.recv(65536)
            if not part:        # EOF — the game half-closed, message complete
                break
            chunks.append(part)

    try:
        s = json.loads(b"".join(chunks))
    except ValueError:
        continue             # never trust the wire

    ox, oy, oz = s["origin"]
    for x, y, z, i in s["blocks"]:
        block = s["palette"][i]
        # world position, if you want it: (x+ox, y+oy, z+oz)
        handle(block, x, y, z)
```

## Changing the endpoint

The game side reads `POSSTREAM_HOST` at startup and falls back to `100.66.148.86`, so you can aim a
session at a laptop for testing without rebuilding:

```bash
POSSTREAM_HOST=127.0.0.1 ./gradlew runClient
```

The port is the `PORT` constant in `StructureScannerItem.java` and is not configurable at runtime.
On startup the mod logs its target, which is the quickest way to confirm where a build is going:

```
[Render thread/INFO] (posstream) Structure target: 127.0.0.1:5005
```

---

Describes the Structure Scanner item in `posstream` 1.0.0, built against Minecraft 1.21.11 with
Fabric Loader 0.19.5. The example payload and the startup log line are real captures from that
build.
