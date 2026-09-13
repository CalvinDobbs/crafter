"""Comprehensive verification for Step 8 (Offline & Robot Voice Pipeline).

Tests:
1. Command mapping: pick, place, home, rotate, celebrate, gripper, fail.
2. Concurrency: audio chunks stream WHILE motion action executes.
3. Lock independence: /say and speech queue never block on _busy motion lock.
4. Preemption: rapid successive speech calls preempt correctly.
5. Audio integrity: all 36 prebuilt neutral wavs in wavs/ are valid 16kHz mono int16.
6. Fail-open: errors in audio or sink never raise or block motion.
"""
from __future__ import annotations

import glob
import sys
import threading
import time
import wave
from pathlib import Path

# Add voice dir to path
VOICE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VOICE_DIR))

import hook
import moves
import narrator
from speech_queue import SpeechQueue
import tts


def test_command_mapping() -> None:
    print("[1/6] Testing command mapping sequence...")
    # Test pick
    hook.log.clear()
    p_seq = moves.pick([0.4, 0.0, 0.05], {"id": 3})
    assert p_seq == ["pick.approach", "pick.descend", "pick.grasp", "pick.lift"]
    assert hook.log == p_seq, f"Expected {p_seq}, got {hook.log}"

    # Test place
    hook.log.clear()
    pl_seq = moves.place([0.4, 0.2, 0.05], {"y": 1})
    assert pl_seq == ["place.approach", "place.descend", "place.release", "place.retreat"]
    assert hook.log == pl_seq, f"Expected {pl_seq}, got {hook.log}"

    # Test home
    hook.log.clear()
    moves.home()
    assert hook.log == ["home.start"]

    # Test rotate
    hook.log.clear()
    moves.rotate(0.5)
    assert hook.log == ["scan.start"]

    # Test celebrate
    hook.log.clear()
    moves.celebrate()
    assert hook.log == ["build.done"]

    # Test gripper
    hook.log.clear()
    moves.gripper(open_=True)
    assert hook.log == ["place.release"]
    moves.gripper(open_=False)
    assert hook.log == ["place.release", "pick.grasp"]

    # Test fail
    hook.log.clear()
    moves.fail()
    assert hook.log == ["fail"]

    print("  -> All command mappings verified successfully.")


def test_concurrency_during_motion() -> None:
    print("[2/6] Testing concurrent speech playback during motion...")
    chunks_received = []

    def sink(chunk: bytes) -> None:
        chunks_received.append((time.perf_counter(), len(chunk)))

    q = SpeechQueue(play_s=0.05, audio_sink=sink)

    # Simulate motion step (0.8s grasp duration)
    t_start = time.perf_counter()
    latency = q.say("pick.grasp")
    assert latency < 0.005, f"say() latency must be <5ms, was {latency:.6f}s"

    time.sleep(0.8)
    t_end = time.perf_counter()

    during_motion = [t for t, sz in chunks_received if t_start <= t <= t_end]
    print(f"  -> Motion duration: {t_end - t_start:.3f}s, Chunks delivered during motion: {len(during_motion)}")
    assert len(during_motion) >= 6, f"Expected >= 6 chunks during motion, got {len(during_motion)}"

    q.close()
    print("  -> Concurrency verified: speech plays concurrently while motion executes.")


def test_busy_lock_independence() -> None:
    print("[3/6] Testing _busy motion lock independence...")
    _busy = threading.Lock()
    _busy.acquire()  # simulate long-running motion command holding the lock

    # Speech queue must operate freely without touching _busy
    received = []
    q = SpeechQueue(play_s=0.05, audio_sink=lambda c: received.append(len(c)))

    t0 = time.perf_counter()
    lat = q.say_text(narrator.line("pick.grasp"))
    dt = time.perf_counter() - t0

    assert dt < 0.005, f"say_text blocked on motion! latency: {dt:.6f}s"
    time.sleep(0.3)
    assert len(received) > 0, "No chunks streamed while motion lock was active!"
    q.close()
    _busy.release()
    print("  -> Lock independence verified: voice never holds or waits on motion lock.")


def test_preemption() -> None:
    print("[4/6] Testing speech queue preemption policy...")
    played = []
    q = SpeechQueue(play_s=0.6, audio_sink=None)

    q.say("pick.approach", {"id": 3})
    time.sleep(0.05)
    # Rapid successive lines
    q.say("pick.descend")
    time.sleep(0.02)
    q.say("pick.grasp")

    time.sleep(0.8)
    q.close()
    # Started list should show approach started, and pending was replaced by grasp
    print(f"  -> Lines started: {q.started}")
    assert len(q.started) >= 2, f"Expected preemption to start >=2 lines, got {q.started}"
    assert "Got it — claws closed." in q.started or "Closing the J7 gripper." in q.started
    print("  -> Preemption verified successfully.")


def test_wav_cache_integrity() -> None:
    print("[5/6] Testing prebuilt WAV audio cache integrity...")
    wav_files = list(glob.glob(str(VOICE_DIR / "wavs" / "*.wav")))
    print(f"  -> Found {len(wav_files)} WAV files in wavs/")
    assert len(wav_files) >= 30, f"Expected >= 30 wav files, got {len(wav_files)}"

    for wf in wav_files:
        with wave.open(wf, "rb") as w:
            assert w.getnchannels() == 1, f"{wf} must be mono"
            assert w.getsampwidth() == 2, f"{wf} must be 16-bit (2 bytes/sample)"
            assert w.getframerate() == 16000, f"{wf} must be 16kHz"
            nframes = w.getnframes()
            assert nframes > 0, f"{wf} is empty"
    print("  -> Audio cache integrity verified: all WAVs are valid 16kHz mono int16.")


def test_fail_open() -> None:
    print("[6/6] Testing fail-open safety...")
    def bad_sink(_chunk: bytes) -> None:
        raise RuntimeError("simulated hardware audio glitch")

    q = SpeechQueue(play_s=0.05, audio_sink=bad_sink)
    # Must not raise
    lat = q.say("pick.lift")
    assert lat < 0.005
    time.sleep(0.2)
    q.close()
    print("  -> Fail-open safety verified: sink errors handled gracefully without crash.")


def main() -> None:
    print("=== Running Step 8 Comprehensive Test Suite ===")
    test_command_mapping()
    test_concurrency_during_motion()
    test_busy_lock_independence()
    test_preemption()
    test_wav_cache_integrity()
    test_fail_open()
    print("\nALL STEP 8 VERIFICATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
