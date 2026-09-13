"""Personality packs: OpenAI style prompt + bland fail-open templates."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VoicePack:
    id: str
    """Full brief for the OpenAI voice-line writer."""
    style: str
    """Fixed startup if OpenAI is off; empty = skip or use LLM `startup` key."""
    startup: str = ""
    """Fail-open only when OpenAI is unavailable. Keep bland."""
    templates: dict[str, str] = field(default_factory=dict)
    """Fail-open pair line for orchestrator Plan.narration."""
    pair: str = "{kind} block, layer {y}."
    """Soft cap for LLM lines (templates still clipped by narrator.MAX_WORDS)."""
    max_words: int = 8


NEUTRAL = VoicePack(
    id="neutral",
    style="Plain, factual robot telemetry. No jokes. No character voice.",
    startup="",
    templates={},
    pair="{kind} block, layer {y}.",
    max_words=8,
)

# Demo personality: Trump-like announcer. Specific lines come from OpenAI.
# Templates below are bland fail-open only — not the spoken product.
TRUMP = VoicePack(
    id="trump",
    style=(
        "You are the spoken voice of a cardboard-box stacking robot demo. "
        "Speak in a playful parody of Donald Trump's public speaking style: "
        "confident, hyperbolic, first person, short punchy clauses, "
        "occasional superlatives (tremendous, beautiful, the best), "
        "and folksy asides — but keep it family-friendly and non-political. "
        "This is about stacking boxes on a table, NOT elections or real people. "
        "Never insult anyone. Never mention parties, elections, or opponents. "
        "Channel the cadence of lines like: \"We're gonna build a big beautiful wall.\" "
        "Audience-friendly words only (box, claw, layer, wall of boxes) — "
        "never hardware jargon (no J0, J7, IK, gripper index). "
        "Vary rhythm across the session: not every line is a punchline. "
        "Use exclamation marks sparingly (at most a couple in the whole build). "
        "Most lines end with a period. Do not repeat the same catchphrase or "
        "the same adjective stack line after line. "
        "Every line must still match the motion event and keep any given facts "
        "(box id, layer y, count n)."
    ),
    startup="",  # OpenAI writes startup into script key "startup"
    templates={},  # fall back to narrator.NEUTRAL if API down
    pair="{kind} on layer {y}.",
    max_words=12,
)

# Back-compat alias for env VOICE_PACK=boxing
BOXING = TRUMP

PACKS: dict[str, VoicePack] = {
    NEUTRAL.id: NEUTRAL,
    TRUMP.id: TRUMP,
    "boxing": TRUMP,
}

_active: str = TRUMP.id


def get_pack(name: str | None = None) -> VoicePack:
    key = (name or _active).strip().lower() or TRUMP.id
    if key not in PACKS:
        raise KeyError(f"unknown voice pack {key!r}; have {sorted(set(PACKS))}")
    return PACKS[key]


def set_pack(name: str) -> VoicePack:
    global _active
    pack = get_pack(name)
    _active = pack.id
    return pack


def list_packs() -> list[str]:
    return sorted({p.id for p in PACKS.values()})
