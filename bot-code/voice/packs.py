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


# Telemetry fallback — hardware-true, no personality.
NEUTRAL = VoicePack(
    id="neutral",
    style="Plain, factual, calm robot telemetry. No jokes.",
    startup="",
    templates={},  # use narrator.NEUTRAL defaults
    pair="{kind} block, layer {y}.",
)

# Demo personality (locked 2026-09-12):
# calm engineer + light puns + first person + audience-friendly (not J0/J7).
# Startup catchphrase only: "It's boxing time!"
BOXING = VoicePack(
    id="boxing",
    style=(
        "Calm first-person engineer narrating a cardboard-box build. "
        "Audience-friendly words (claw, lift, layer) — never joint names. "
        "Light wordplay only; stay short and steady. No hype yelling."
    ),
    startup="It's boxing time!",
    templates={
        "scan.start": "I'm scanning for boxes.",
        "plan.ready": "{n} boxes on the list.",
        "home.start": "Waking my arms up.",
        "pick.approach": "Heading for box {id}.",
        "pick.descend": "Dropping in carefully.",
        "pick.grasp": "Got it — claws closed.",
        "pick.lift": "Lifting clear.",
        "place.approach": "Taking this to layer {y}.",
        "place.descend": "Lining up the drop.",
        "place.release": "And that's planted.",
        "place.retreat": "Backing off.",
        "build.done": "Build's done. Still steady.",
        "fail": "Missed that — opening claws.",
    },
    pair="{kind} for layer {y} — my turn.",
)

PACKS: dict[str, VoicePack] = {
    NEUTRAL.id: NEUTRAL,
    BOXING.id: BOXING,
}

_active: str = BOXING.id


def get_pack(name: str | None = None) -> VoicePack:
    key = (name or _active).strip().lower() or BOXING.id
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
