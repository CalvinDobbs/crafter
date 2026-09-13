"""Ahead-of-time personality lines for a decided plan.

Assignment (which box → which cell) stays in the planner.
This module only writes spoken text, before any motion starts.
OpenAI optional; templates always work offline.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

from narrator import EVENTS, line
from packs import VoicePack, get_pack

# event|id=…|y=… → spoken line (filled before motion)
_overrides: dict[str, str] = {}


def script_key(event: str, ctx: dict | None = None) -> str:
    ctx = ctx or {}
    parts = [event]
    if "id" in ctx and ctx["id"] is not None:
        parts.append(f"id={ctx['id']}")
    if "y" in ctx and ctx["y"] is not None:
        parts.append(f"y={ctx['y']}")
    if "n" in ctx and ctx["n"] is not None:
        parts.append(f"n={ctx['n']}")
    return "|".join(parts)


def clear_script() -> None:
    _overrides.clear()


def load_script(mapping: dict[str, str]) -> None:
    clear_script()
    _overrides.update({k: v for k, v in mapping.items() if v})


def lookup(event: str, ctx: dict | None = None) -> str | None:
    return _overrides.get(script_key(event, ctx))


def template_lines(
    events: list[tuple[str, dict]],
    pack: VoicePack | None = None,
) -> dict[str, str]:
    """Fill every event with pack/neutral templates (no network)."""
    if pack is not None:
        from packs import set_pack
        set_pack(pack.id)
    out: dict[str, str] = {}
    for event, ctx in events:
        text = line(event, ctx)
        if text:
            out[script_key(event, ctx)] = text
    return out


def _openai_rewrite(
    events: list[tuple[str, dict]],
    pack: VoicePack,
    *,
    model: str,
    timeout: float,
    client_factory: Callable | None = None,
) -> dict[str, str]:
    """One chat call → rewritten lines. Raises on any failure."""
    base = template_lines(events, pack)
    items = []
    for event, ctx in events:
        key = script_key(event, ctx)
        if key not in base:
            continue
        items.append({"key": key, "event": event, "ctx": ctx, "fallback": base[key]})
    if not items:
        return base

    prompt = {
        "style": pack.style,
        "rules": [
            "Rewrite each fallback into one spoken line in the given style.",
            "Keep the same facts (box id, layer y, count n) when present.",
            "At most 8 words per line.",
            "No emoji. No quotes in the strings.",
            "Return ONLY json: {\"lines\": [{\"key\": str, \"text\": str}, ...]}",
            "Include every key exactly once.",
        ],
        "items": items,
    }
    if client_factory is not None:
        client = client_factory()
    else:
        from openai import OpenAI
        client = OpenAI(timeout=timeout)

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You write short robot voice lines. JSON only.",
            },
            {"role": "user", "content": json.dumps(prompt)},
        ],
        temperature=0.6,
    )
    elapsed = time.perf_counter() - t0
    if elapsed > timeout:
        raise TimeoutError(f"openai flavor took {elapsed:.1f}s > {timeout}")

    raw = resp.choices[0].message.content or ""
    data = json.loads(raw)
    lines = data.get("lines") or data.get("rewrites") or []
    out = dict(base)
    for row in lines:
        key = row.get("key")
        text = (row.get("text") or "").strip()
        if key in out and text:
            words = text.split()
            if len(words) > 8:
                text = " ".join(words[:8])
            out[key] = text
    return out


def _resolve_openai_key() -> str | None:
    """Env first, then shared bot file ~/.config/crafter/openai_api_key."""
    value = os.environ.get("OPENAI_API_KEY", "").strip()
    if value:
        return value
    path = Path.home() / ".config" / "crafter" / "openai_api_key"
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text or None
    except OSError:
        return None


def prewrite(
    events: list[tuple[str, dict]],
    pack: str | VoicePack | None = None,
    *,
    flavor: str = "template",
    model: str | None = None,
    timeout: float = 8.0,
    client_factory: Callable | None = None,
) -> dict[str, str]:
    """Build a full script before motion. Never raises; fail-open to templates.

    flavor:
      - template: pack templates only
      - openai: one OpenAI rewrite pass (needs OPENAI_API_KEY or saved bot key); else templates
    """
    vp = pack if isinstance(pack, VoicePack) else get_pack(pack)
    from packs import set_pack
    set_pack(vp.id)

    script = template_lines(events, vp)
    if flavor != "openai":
        load_script(script)
        return script

    if client_factory is None:
        key = _resolve_openai_key()
        if not key:
            print("[voice] OPENAI_API_KEY missing; using templates", flush=True)
            load_script(script)
            return script
        os.environ["OPENAI_API_KEY"] = key

    try:
        script = _openai_rewrite(
            events,
            vp,
            model=model or os.environ.get("VOICE_MODEL", "gpt-4o-mini"),
            timeout=timeout,
            client_factory=client_factory,
        )
        print(f"[voice] openai flavored {len(script)} lines ({vp.id})", flush=True)
    except Exception as e:
        print(f"[voice] openai flavor failed ({e}); templates", flush=True)
        script = template_lines(events, vp)

    load_script(script)
    return script


def pair_lines_for_plan(
    plan_actions: list[dict],
    kinds_by_cell: dict[tuple[int, int, int], str] | None = None,
    pack: VoicePack | None = None,
) -> list[str]:
    """One orchestrator-style line per pick/place pair (legacy Plan.narration)."""
    vp = pack or get_pack()
    kinds_by_cell = kinds_by_cell or {}
    out: list[str] = []
    pending_id = None
    for a in plan_actions:
        kind = a.get("kind") if isinstance(a, dict) else a.kind
        if kind == "pick":
            pending_id = a.get("box_id") if isinstance(a, dict) else a.box_id
        elif kind == "place":
            cell = a.get("cell") if isinstance(a, dict) else a.cell
            cell_t = tuple(cell) if cell else (0, 0, 0)
            y = cell_t[1]
            block_kind = kinds_by_cell.get(cell_t, "block")
            text = vp.pair.format(kind=block_kind, y=y, id=pending_id)
            words = text.split()
            if len(words) > 8:
                text = " ".join(words[:8])
            out.append(text)
            pending_id = None
    if vp.startup:
        # startup is separate; caller may prepend via celebrate/home
        pass
    return out


if __name__ == "__main__":
    from from_plan import events_from_plan
    import json
    from pathlib import Path

    plan = json.loads((Path(__file__).parent / "fixture_plan.json").read_text())
    events = events_from_plan(plan)
    for flavor in ("template",):
        s = prewrite(events, pack="boxing", flavor=flavor)
        print(f"=== {flavor} boxing ({len(s)} lines) ===")
        for k, v in s.items():
            print(f"  {k:28} {v!r}")
