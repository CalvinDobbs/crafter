"""TTS → int16 16 kHz PCM, 1600-sample / 100 ms frames (speaker.audio).

Backends (first hit wins, fail-open to silence):
  1. Cached wav in wavs/ or .tts_cache/
  2. OpenAI audio.speech (robot / when key present)
  3. macOS `say` + afconvert (laptop)

There is no authorized “Trump voice” clone here — use OPENAI_TTS_VOICE
(default onyx: deep announcer) as a legal stand-in for the trump persona pack.
"""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

RATE = 16000
FRAME = 1600  # 100 ms
OPENAI_TTS_RATE = 24000  # OpenAI pcm output
CACHE = Path(__file__).parent / ".tts_cache"
WAVS = Path(__file__).parent / "wavs"

_SAY = shutil.which("say")
_AFCONVERT = shutil.which("afconvert")


def _voice_tag() -> str:
    # Include TTS backend voice in cache key so Mac vs OpenAI don't collide.
    oai = os.environ.get("OPENAI_TTS_VOICE", "onyx").strip() or "onyx"
    mac = os.environ.get("SAY_VOICE", "").strip()
    return f"oai:{oai}|mac:{mac}"


def _cache_path(text: str, voice: str | None = None) -> Path:
    voice = voice if voice is not None else _voice_tag()
    h = hashlib.sha1(f"{voice}\0{text}".encode()).hexdigest()[:16]
    wav_cand = WAVS / f"{h}.wav"
    if wav_cand.exists():
        return wav_cand
    return CACHE / f"{h}.wav"


def _resolve_openai_key() -> str | None:
    value = os.environ.get("OPENAI_API_KEY", "").strip()
    if value:
        return value
    path = Path.home() / ".config" / "crafter" / "openai_api_key"
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _downsample_24k_to_16k(pcm24: bytes) -> bytes:
    import numpy as np
    x = np.frombuffer(pcm24, dtype=np.int16).astype(np.float32)
    if x.size == 0:
        return b""
    n = max(1, int(round(x.size * RATE / OPENAI_TTS_RATE)))
    idx = np.linspace(0, x.size - 1, n)
    y = np.interp(idx, np.arange(x.size), x)
    return np.clip(y, -32768, 32767).astype(np.int16).tobytes()


def _write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


def _synthesize_openai(text: str, wav_path: Path) -> None:
    key = _resolve_openai_key()
    if not key:
        raise RuntimeError("no OpenAI API key")
    os.environ.setdefault("OPENAI_API_KEY", key)
    from openai import OpenAI
    voice = os.environ.get("OPENAI_TTS_VOICE", "onyx").strip() or "onyx"
    model = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts").strip() or "gpt-4o-mini-tts"
    client = OpenAI(timeout=30)
    # pcm = raw 24 kHz mono LEI16
    try:
        resp = client.audio.speech.create(
            model=model, voice=voice, input=text, response_format="pcm")
    except Exception:
        # older models / accounts: tts-1 + wav
        resp = client.audio.speech.create(
            model=os.environ.get("OPENAI_TTS_MODEL_FALLBACK", "tts-1"),
            voice=voice, input=text, response_format="wav")
        data = resp.content
        # may already be wav — try read; else treat as pcm24
        try:
            with wave.open(io.BytesIO(data), "rb") as w:
                raw = w.readframes(w.getnframes())
                if w.getnchannels() == 2:
                    raw = b"".join(raw[i:i+2] for i in range(0, len(raw), 4))
                rate = w.getframerate()
            if rate == RATE:
                pcm = raw
            elif rate == OPENAI_TTS_RATE:
                pcm = _downsample_24k_to_16k(raw)
            else:
                raise RuntimeError(f"unsupported wav rate {rate}")
            _write_wav(wav_path, pcm)
            # also mirror into wavs/ for robot reuse
            _write_wav(WAVS / wav_path.name, pcm)
            return
        except Exception:
            pcm = _downsample_24k_to_16k(data)
            _write_wav(wav_path, pcm)
            _write_wav(WAVS / wav_path.name, pcm)
            return

    pcm = _downsample_24k_to_16k(resp.content)
    _write_wav(wav_path, pcm)
    _write_wav(WAVS / wav_path.name, pcm)


def _synthesize_mac(text: str, wav_path: Path) -> None:
    if not _SAY or not _AFCONVERT:
        raise FileNotFoundError("need /usr/bin/say and afconvert")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    voice = os.environ.get("SAY_VOICE", "").strip()
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
        aiff = tmp.name
    cmd = ["say", "-o", aiff]
    if voice:
        cmd.extend(["-v", voice])
    cmd.append(text)
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=15)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", aiff, str(wav_path)],
            check=True, capture_output=True, timeout=15)
    finally:
        try:
            os.unlink(aiff)
        except OSError:
            pass


def _synthesize(text: str, dest: Path) -> None:
    # Prefer OpenAI on robot / when key exists; Mac say otherwise.
    errs = []
    if _resolve_openai_key():
        try:
            _synthesize_openai(text, dest)
            return
        except Exception as e:
            errs.append(f"openai:{e}")
    if _SAY and _AFCONVERT:
        try:
            _synthesize_mac(text, dest)
            return
        except Exception as e:
            errs.append(f"mac:{e}")
    raise RuntimeError("no TTS backend worked: " + "; ".join(errs) if errs else "none")


def pcm16(text: str) -> bytes:
    """UTF-8 line → little-endian int16 mono 16 kHz, or b'' on failure."""
    if not text or os.environ.get("TTS_FAIL") == "1":
        return b""
    dest = _cache_path(text)
    try:
        if not dest.exists():
            _synthesize(text, dest)
        with wave.open(str(dest), "rb") as w:
            if w.getsampwidth() != 2:
                return b""
            raw = w.readframes(w.getnframes())
            if w.getnchannels() == 2:
                raw = b"".join(raw[i:i+2] for i in range(0, len(raw), 4))
            if w.getframerate() != RATE:
                return b""
            return raw
    except Exception as e:
        if os.environ.get("TTS_DEBUG"):
            print("tts fail", type(e), e)
        return b""


def frames(pcm: bytes) -> list[bytes]:
    """Split into 1600-sample (3200 byte) chunks; pad last."""
    step = FRAME * 2
    out = []
    i = 0
    while i < len(pcm):
        chunk = pcm[i:i + step]
        if len(chunk) < step:
            chunk = chunk + b"\x00" * (step - len(chunk))
        out.append(chunk)
        i += step
    return out


def speak_line(text: str) -> list[bytes]:
    """TTS or empty list. Never raises."""
    try:
        return frames(pcm16(text))
    except Exception:
        return []


def _wav_wrap(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def _write_preview(text: str) -> Path | None:
    out = Path(__file__).parent / "tts_demo.wav"
    pcm = pcm16(text)
    if not pcm:
        print("TTS produced no pcm")
        return None
    out.write_bytes(_wav_wrap(pcm))
    n = len(frames(pcm))
    print(f"text={text!r}")
    print(f"wrote {out}  {n} frames ({n * 0.1:.1f}s)")
    return out


def prebuild() -> None:
    """Pre-generate pack template wavs so robot does not need local synthesis."""
    from narrator import EVENTS, line
    from packs import PACKS, set_pack

    WAVS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    texts: set[str] = set()
    for pack_id, pack in PACKS.items():
        set_pack(pack_id)
        if pack.startup:
            texts.add(pack.startup)
        for ev in EVENTS:
            texts.add(line(ev))
        for n in range(1, 9):
            texts.add(line("plan.ready", {"n": n}))
        for box_id in range(10):
            texts.add(line("pick.approach", {"id": box_id}))
        for layer in range(5):
            texts.add(line("place.approach", {"y": layer}))
            texts.add(pack.pair.format(kind="block", y=layer, id=0))
            texts.add(pack.pair.format(kind="dirt", y=layer, id=1))

    for t in sorted(x for x in texts if x):
        dest = _cache_path(t)
        if not dest.exists():
            print(f"[prebuild] synthesizing: {t!r}")
            try:
                _synthesize(t, dest)
            except Exception as e:
                print(f"[prebuild] skip {t!r}: {e}")
    print(f"[prebuild] done. {len(list(WAVS.glob('*.wav')))} wavs in {WAVS}")


def _demo_check() -> None:
    from narrator import line
    text = line("pick.grasp", {"id": 3})
    path = _write_preview(text)
    os.environ["TTS_FAIL"] = "1"
    empty = pcm16(text)
    os.environ.pop("TTS_FAIL", None)
    ok = path is not None and path.stat().st_size > 44 and empty == b""
    print(f"fail-open empty={empty == b''}")
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    import argparse
    from narrator import EVENTS, line

    ap = argparse.ArgumentParser(description="Preview one narration line (overwrites tts_demo.wav).")
    ap.add_argument("event", nargs="?", help="e.g. pick.lift  (omit with --check/--list)")
    ap.add_argument("--id", type=int, default=3)
    ap.add_argument("--y", type=int, default=1)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--list", action="store_true", help="print all event lines, no TTS")
    ap.add_argument("--play", action="store_true", help="afplay tts_demo.wav after synth")
    ap.add_argument("--check", action="store_true", help="5.14 regression (PASS/FAIL)")
    ap.add_argument("--prebuild", action="store_true", help="pre-synthesize template lines into wavs/")
    a = ap.parse_args()

    if a.prebuild:
        prebuild()
        raise SystemExit(0)
    if a.check:
        _demo_check()
    if a.list or not a.event:
        ctx = {"id": a.id, "y": a.y, "n": a.n}
        for ev in EVENTS:
            print(f"{ev:16} {line(ev, ctx)!r}")
        if not a.event:
            print("\nPreview one:  python3 tts.py pick.lift")
            print("Play it:      python3 tts.py pick.lift --play")
            raise SystemExit(0)
    ctx = {"id": a.id, "y": a.y, "n": a.n}
    text = line(a.event, ctx)
    if not text:
        print(f"{a.event}: skip (empty line)")
        raise SystemExit(0)
    path = _write_preview(text)
    if path is None:
        raise SystemExit(1)
    if a.play:
        play = shutil.which("afplay")
        if play:
            subprocess.run([play, str(path)], check=False)
