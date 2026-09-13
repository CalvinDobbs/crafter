#!/usr/bin/env python3
"""Generate Trump-style lines + onyx wavs on the laptop, push to the robot, play.

Why: Jetson can reach OpenAI TTS but not chat.completions. So we write the
script here, synthesize audio here, scp wavs (+ script JSON) to the bot, and
only use the bot as a speaker via POST /say.

Usage (from repo root or bot-code/voice):
  python3 push_voice.py                  # fixture plan → generate → scp → /say demo
  python3 push_voice.py --no-play        # generate + scp only
  python3 push_voice.py --plan path.json

Needs OPENAI_API_KEY in the environment, or pulls the shared bot key over SSH
into this process only (never printed).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import flavor
import packs
import tts
from from_plan import events_from_plan

DEFAULT_HOST = os.environ.get("CRAFTER_BOT", "bracketbot@100.66.148.86")
REMOTE_DIR = "/home/bracketbot/bbapps/mc_skills"
SAY_URL_DEFAULT = "http://100.66.148.86:8006/say"


def _ssh(host: str, remote_cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, remote_cmd],
        capture_output=True, text=True)


def ensure_api_key(host: str) -> None:
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return
    # Pull shared bot key into this process only — do not print it.
    r = _ssh(host, "cat ~/.config/crafter/openai_api_key")
    if r.returncode != 0 or not r.stdout.strip():
        raise SystemExit(
            "No OPENAI_API_KEY locally and could not read "
            "~/.config/crafter/openai_api_key on the bot.")
    os.environ["OPENAI_API_KEY"] = r.stdout.strip()
    print("[push_voice] loaded API key from bot keyfile (not displayed)")


def generate(plan: dict, pack: str = "trump") -> dict[str, str]:
    os.environ.setdefault("OPENAI_TTS_VOICE", "onyx")
    packs.set_pack(pack)
    events = events_from_plan(plan)
    print(f"[push_voice] OpenAI chat flavor ({pack}) for {len(events)} events…")
    script = flavor.prewrite(events, pack=pack, flavor="openai", timeout=180)
    if "startup" not in script:
        # Persona reference line if model omitted startup.
        script["startup"] = "We're gonna build a big beautiful wall."
        flavor.load_script(script)
    return script


def synth_wavs(script: dict[str, str]) -> list[Path]:
    os.environ.setdefault("OPENAI_TTS_VOICE", "onyx")
    paths = []
    texts = sorted({t for t in script.values() if t})
    print(f"[push_voice] synthesizing {len(texts)} onyx wavs…")
    for text in texts:
        dest = tts._cache_path(text)
        # Force into wavs/ for robot deploy.
        wav_dest = tts.WAVS / dest.name
        if wav_dest.exists() and wav_dest.stat().st_size > 44:
            print(f"  cached {wav_dest.name}  {text!r}")
            paths.append(wav_dest)
            continue
        pcm = tts.pcm16(text)
        if not pcm:
            print(f"  FAIL synth {text!r}")
            continue
        # pcm16 may have written cache; copy into wavs/
        src = tts._cache_path(text)
        if src.exists():
            wav_dest.write_bytes(src.read_bytes())
        else:
            tts._write_wav(wav_dest, pcm)
        print(f"  wrote {wav_dest.name}  {text!r}")
        paths.append(wav_dest)
    return paths


def push(host: str, script: dict[str, str], wavs: list[Path], say_url: str) -> None:
    script_path = HERE / "voice_script.json"
    script_path.write_text(json.dumps(script, indent=2) + "\n")
    print(f"[push_voice] scp → {host}:{REMOTE_DIR}")
    _ssh(host, f"mkdir -p {REMOTE_DIR}/wavs")
    subprocess.run(
        ["scp", "-o", "BatchMode=yes", str(script_path), f"{host}:{REMOTE_DIR}/voice_script.json"],
        check=True)
    if wavs:
        cmd = ["scp", "-o", "BatchMode=yes"] + [str(p) for p in wavs] + [f"{host}:{REMOTE_DIR}/wavs/"]
        subprocess.run(cmd, check=True)
    # Hot-load into running mc_skills if /voice_script exists.
    try:
        body = json.dumps({"lines": script}).encode()
        base = say_url.rsplit("/", 1)[0]
        req = urllib.request.Request(
            f"{base}/voice_script", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"[push_voice] /voice_script {resp.read().decode()}")
    except Exception as e:
        print(f"[push_voice] /voice_script skip ({e}); restart mc_skills to load file")
    print(f"[push_voice] pushed script ({len(script)} lines) + {len(wavs)} wavs")


def say(url: str, text: str) -> None:
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        print(f"  /say {resp.status} {text!r}")


def play_demo(url: str, script: dict[str, str]) -> None:
    order = ["startup"]
    # Prefer a few highlight lines if present.
    for key in script:
        if key.startswith("pick.grasp") or key.startswith("place.release") or key == "build.done":
            order.append(key)
    seen = set()
    print("[push_voice] playing on robot speaker…")
    for key in order:
        text = script.get(key)
        if not text or text in seen:
            continue
        seen.add(text)
        try:
            say(url, text)
        except Exception as e:
            print(f"  /say failed: {e}")
        time.sleep(2.2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--plan", type=Path, default=HERE / "fixture_plan.json")
    ap.add_argument("--pack", default="trump")
    ap.add_argument("--say-url", default=SAY_URL_DEFAULT)
    ap.add_argument("--no-play", action="store_true")
    ap.add_argument("--no-push", action="store_true", help="generate locally only")
    a = ap.parse_args()

    ensure_api_key(a.host)
    plan = json.loads(a.plan.read_text())
    script = generate(plan, pack=a.pack)
    print("[push_voice] script preview:")
    for k, v in list(script.items())[:8]:
        print(f"  {k}: {v}")
    wavs = synth_wavs(script)
    if not a.no_push:
        push(a.host, script, wavs, a.say_url)
    if not a.no_play and not a.no_push:
        play_demo(a.say_url, script)
    print("[push_voice] done")


if __name__ == "__main__":
    main()
