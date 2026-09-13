"""Personality packs: templates + style for LLM flavor. TTS voice id comes later."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VoicePack:
    id: str
    """Short style note fed to the LLM when rewriting lines."""
    style: str
    """Said once at build start (empty = skip)."""
    startup: str = ""
    """Event → template. Missing keys fall back to narrator.NEUTRAL."""
    templates: dict[str, str] = field(default_factory=dict)
    """One line per pick/place pair when not using phase-level speech."""
    pair: str = "{kind} block, layer {y}, going in"


NEUTRAL = VoicePack(
    id="neutral",
    style="Plain, factual, calm robot telemetry. No jokes.",
    startup="",
    templates={},  # use narrator.NEUTRAL defaults
    pair="{kind} block, layer {y}.",
)

BOXING = VoicePack(
    id="boxing",
    style=(
        "Energetic pit-crew hype for a cardboard-box stacking robot. "
        "Short punches, sports-announce energy, never mean. "
        "May reference boxing / boxes wordplay lightly."
    ),
    startup="It's boxing time!",
    templates={
        "scan.start": "Scouting the ring for boxes.",
        "plan.ready": "{n} rounds on the card.",
        "home.start": "Gloves on. Arms waking up.",
        "pick.approach": "Closing in on box {id}.",
        "pick.descend": "Dropping in for the grab.",
        "pick.grasp": "Clamped. Box secured.",
        "pick.lift": "Up and clear.",
        "place.approach": "Heading to layer {y}.",
        "place.descend": "Lining up the drop.",
        "place.release": "And it's planted.",
        "place.retreat": "Backing out clean.",
        "build.done": "Card done. Still balanced.",
        "fail": "Slip! Resetting the grip.",
    },
    pair="{kind} on layer {y} — going in hot.",
)

PACKS: dict[str, VoicePack] = {
    NEUTRAL.id: NEUTRAL,
    BOXING.id: BOXING,
}

_active: str = NEUTRAL.id


def get_pack(name: str | None = None) -> VoicePack:
    key = (name or _active).strip().lower() or NEUTRAL.id
    if key not in PACKS:
        raise KeyError(f"unknown voice pack {key!r}; have {sorted(PACKS)}")
    return PACKS[key]


def set_pack(name: str) -> VoicePack:
    global _active
    pack = get_pack(name)
    _active = pack.id
    return pack


def list_packs() -> list[str]:
    return sorted(PACKS)
