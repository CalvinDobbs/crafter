"""Non-blocking speech queue.

Later `/say` in mc_skills must NOT take `_busy` (that lock serializes pick/place).
Policy: one line playing, at most one pending. A new say() replaces pending.
If the current line has more than CUT_S left, it is cut so the pending can start.
"""
from __future__ import annotations

import threading
import time

from narrator import line

DO_NOT_HOLD_MOTION_LOCK = True
CUT_S = 0.4


class SpeechQueue:
    def __init__(self, play_s: float = 0.25, audio_sink=None):
        self.play_s = play_s
        self.audio_sink = audio_sink
        self._pending: str | None = None
        self._cv = threading.Condition()
        self._cut = threading.Event()
        self._playing = False
        self._play_t0 = 0.0
        self.started: list[str] = []
        self._stop = False
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def say(self, event: str, ctx: dict | None = None) -> float:
        text = line(event, ctx)
        return self.say_text(text)

    def say_text(self, text: str) -> float:
        t0 = time.perf_counter()
        if text:
            with self._cv:
                if self._playing and (self.play_s - (time.perf_counter() - self._play_t0)) > CUT_S:
                    self._cut.set()
                self._pending = text
                self._cv.notify()
        return time.perf_counter() - t0

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._cut.set()
        self._worker.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop and self._pending is None:
                    return
                text = self._pending
                self._pending = None
                self._playing = True
                self._play_t0 = time.perf_counter()
                self._cut.clear()
            self.started.append(text)
            self._play(text)
            with self._cv:
                self._playing = False

    def _play(self, text: str) -> None:
        if self.audio_sink is not None:
            try:
                import tts
                chunk_list = tts.speak_line(text)
                if chunk_list:
                    for chunk in chunk_list:
                        if self._cut.is_set() or self._stop:
                            return
                        t_start = time.perf_counter()
                        try:
                            self.audio_sink(chunk)
                        except Exception as e:
                            print(f"[speech_queue] audio_sink error: {e}", flush=True)
                            break
                        elapsed = time.perf_counter() - t_start
                        time.sleep(max(0.0, 0.098 - elapsed))
                    return
            except Exception as e:
                print(f"[speech_queue] play error: {e}", flush=True)

        end = time.perf_counter() + self.play_s
        while time.perf_counter() < end:
            if self._cut.is_set() or self._stop:
                return
            time.sleep(0.02)
