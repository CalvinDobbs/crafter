"""Call this at each mapped mc_skills step. Never blocks the caller."""
from __future__ import annotations

from speech_queue import DO_NOT_HOLD_MOTION_LOCK, SpeechQueue

assert DO_NOT_HOLD_MOTION_LOCK

_q = SpeechQueue(play_s=0.05)
log: list[str] = []


def set_audio_sink(sink) -> None:
    _q.audio_sink = sink


def narrate(event: str, ctx: dict | None = None) -> None:
    log.append(event)
    _q.say(event, ctx)
    print(f"[narrate] {event} {ctx or {}}", flush=True)


def say_text(text: str) -> None:
    _q.say_text(text)
    print(f"[say_text] {text}", flush=True)
