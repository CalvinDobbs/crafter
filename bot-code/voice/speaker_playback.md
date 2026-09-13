# Playing sound through the BracketBot speakers

Source-checked on `bracketbot-184` (`100.66.148.86`) on 2026-09-12. This was a read-only review: no playback or TTS requests were triggered, and no hardware services were started or restarted. Actual audibility still needs an operator-approved sound check.

Commands below run **on the bot**, unless stated otherwise. From Windows, connect first using your approved SSH identity; see [the project access notes](../../AGENTS.md). Running a desktop audio player on the PC will not play through the robot.

## 1. The actual playback API

Publish PCM samples through BBOS, not directly to an arbitrary desktop audio device:

```text
WAV / generated PCM / TTS
  -> Writer("speaker.audio", Type("speaker_audio"))
  -> buffer["audio"]
  -> speaker daemon: gain + optional compressor + resampling
  -> pacat / selected PulseAudio server
  -> head-board USB speaker
```

The inspected `speaker/constants.py` defines:

| Property | Value |
|---|---|
| Topic | `speaker.audio` |
| Type | `Type("speaker_audio")` |
| Payload field | `audio` |
| Sample format | Signed 16-bit PCM, `numpy.int16` |
| App-side sample rate | 16,000 Hz |
| Channels | 1, mono |
| Samples per publication | 1,600 |
| Array shape | `(1600, 1)`, not `(1600,)` |
| Frame duration | 100 ms; approximately 10 frames/second |
| PCM bytes per frame | 3,200, excluding BBOS metadata |

Read `Config("speaker").sample_rate`, `.channels`, and `.chunk_size` in code rather than assuming these values will never change. BBOS supplies the timestamp; applications fill the `audio` field.

The hardware-facing stream is **48 kHz**, but the **speaker daemon performs that conversion**. Do not send 48 kHz samples into a topic configured for 16 kHz. WAV/MP3 headers are not PCM samples and must not be copied into `audio`.

## 2. Choose the owner before playing anything

Use **one `Writer` instance per topic**, shared by callers inside its owning process. BBOS rejects a competing live publisher with `RuntimeError` identifying the owner PID.

- For integrated robot speech, reuse the existing audio owner and its queue. Do not create a writer in the reasoning layer or UI for each utterance.
- A standalone sound app is appropriate only after the operator has arranged exclusive ownership of `speaker.audio`.
- Keep a writer alive across a stream or a sequence of utterances. The WAV sample waits **0.5 seconds after opening it** to allow daemon discovery; this is a startup warm-up, not confirmation of audible output.
- Do not start or restart `mc_skills` merely to test sound. It also claims arm/control, torque, drive, and LED resources; its shutdown handler can park homed arms. Coordinate any handoff with the action owner, especially while the robot is loaded.

In a configured bot shell, `list --plain` shows topic ownership and rates. Inspect the `speaker.audio` writer rather than assuming that a listening HTTP port proves audio is connected. Do not use bare `stop` or `restart` to resolve a speaker conflict: those commands affect other robot services and shared-memory topics.

## 3. Use the existing body server for text

A body server was listening on port 8006 during this review. The inspected deployed implementation has a non-blocking `POST /say` route, not merely a print-only stub:

```text
POST /say {text}
  -> hook.say_text
  -> SpeechQueue
  -> tts.speak_line: PCM frames
  -> registered audio_sink
  -> existing speaker.audio writer
```

A read-only health check is:

```bash
curl --fail --silent --show-error --max-time 5 http://127.0.0.1:8006/health
```

**The following request is a real playback request and may incur TTS API charges.** Run it only when the operator wants a sound check:

```bash
curl --fail --show-error --max-time 5 \
  -H 'Content-Type: application/json' \
  -d '{"text":"Speaker check."}' \
  http://127.0.0.1:8006/say
```

Important distinctions:

- `/say` enqueues text without acquiring the motion `_busy` lock. Do not make narration wait for a pick/place operation to finish.
- `{"ok": true}` means the request was accepted, **not** that TTS succeeded, frames reached the daemon, or sound was heard.
- `Body.__init__` catches speaker-writer initialization errors. The HTTP server can remain available without a working audio sink. `/health` does not expose a speaker-ready flag.
- `MOCK=1` skips hardware initialization. The main Crafter panel on 8005 also remains a simulated-tool UI; this guide does not wire physical audio into it.
- The inspected body API accepts text at `/say`; it has no general-purpose WAV-upload or `/play_wav` route.

## 4. Run an existing WAV sample

**Only with exclusive speaker ownership arranged.** Prerecorded WAV playback does not require a model API key. The app's dependencies must already be available for `--offline` execution.

```bash
ls /home/bracketbot/bbapps/play_sound/wavs
/home/bracketbot/.local/bin/uv run --offline /home/bracketbot/bbapps/play_sound/main.py happy_birthday
```

Pass the **filename stem**, without `.wav`. The sample constructs `play_sound/wavs/<stem>.wav`; it does not accept an arbitrary file path in that argument.

The bundled `happy_birthday.wav` header was checked: mono, 16 kHz, 16-bit uncompressed PCM. The script defaults to `greeting` when no argument is supplied, but **`greeting.wav` was not present** in the inspected directory. Supply an existing stem explicitly.

What `play_sound/main.py` does:

1. Waits for `mic.audio` to become ready. If it hangs at “Waiting for mic daemon”, that is an explicit startup loop in this sample.
2. Opens the WAV using `soundfile`.
3. Opens one speaker writer and waits 0.5 seconds for discovery.
4. Reads `CFG.chunk_size` samples as `int16`, zero-pads the final short chunk, reshapes to the channel count, and assigns `b["audio"]` inside `with writer.buf()`.
5. Uses the writer's default BBOS pacing; it does not sleep another 100 ms after every frame.

The sample prints the file's sample rate and channels, but **does not resample or validate them against the speaker configuration**. Convert mismatched files before using it. It catches errors and prints them, so process exit alone is not a sufficient playback-success check.

Other samples:

- `bbapps/examples/view_speaker.py` **plays synthesized chords**. Despite its `view_` name, it is a writer, not a passive diagnostic.
- `bbapps/greeter/main.py` demonstrates a persistent speaker queue, silence on underrun, a four-chunk startup buffer, and conversion from model-generated 24 kHz audio. It is a larger live-model app, not a minimal sound test. Do not copy its private `_update()` polling as a public API pattern.
- `bbapps/play_sound/web_ui.py` is a sample soundboard on 8015. It starts a player subprocess for each selection and kills its previous player; it does not arbitrate ownership with `mc_skills`. It is not a shared robot audio service.

## 5. Minimal standalone WAV publisher

This reference adapts the WAV sample with explicit format checks, quiet gain, and silent bookends. It is for a **speaker-only process with exclusive ownership**, not an additional process beside an existing audio owner. Save it as a Python script on the bot if you need a custom player; this guide does not install or launch it.

```python
import sys
import time
import wave

import numpy as np
from bbos import Config, Type, Writer

cfg = Config("speaker")
gain = 0.25

with wave.open(sys.argv[1], "rb") as wav:
    if (wav.getcomptype() != "NONE" or wav.getsampwidth() != 2
            or wav.getframerate() != cfg.sample_rate
            or wav.getnchannels() != cfg.channels):
        raise ValueError("Use an uncompressed PCM16 WAV matching the speaker rate and channels")

    silence = np.zeros((cfg.chunk_size, cfg.channels), dtype=np.int16)
    with Writer("speaker.audio", Type("speaker_audio")) as speaker:
        time.sleep(0.5)
        with speaker.buf() as buffer:
            buffer["audio"] = silence

        while True:
            raw = wav.readframes(cfg.chunk_size)
            if not raw:
                break
            pcm = np.frombuffer(raw, dtype="<i2").reshape(-1, cfg.channels)
            frame = silence.copy()
            frame[:len(pcm)] = np.clip(
                pcm.astype(np.float32) * gain, -32768, 32767
            ).astype(np.int16)
            with speaker.buf() as buffer:
                buffer["audio"] = frame

        with speaker.buf() as buffer:
            buffer["audio"] = silence
```

Use the existing BBOS app environment, or an isolated `uv` environment with the local BBOS package and NumPy. Replace the script and WAV paths below:

```bash
/home/bracketbot/.local/bin/uv run --offline --no-project \
  --with-editable /home/bracketbot/bbos --with numpy \
  python /path/to/play_wav_once.py /path/to/clip_16k.wav
```

Do not install packages globally or substitute an unrelated package for the robot's local `bbos` source. If the offline cache is incomplete, arrange the app environment with the platform maintainer.

For existing owners, retain their writer and adapt only the frame preparation/assignment. Do not copy the writer-creation block into every callback.

### WAV preparation without playback

This checks a file header using only Python's standard library:

```bash
python -B -c 'import sys, wave; w = wave.open(sys.argv[1], "rb"); print(w.getparams()); w.close()' /path/to/clip.wav
```

On a machine with FFmpeg installed, conversion to a **new** file can be done with:

```bash
ffmpeg -n -i input.wav -ac 1 -ar 16000 -c:a pcm_s16le clip_16k.wav
```

`-n` refuses to overwrite an existing output. FFmpeg was not on the bot's non-interactive SSH `PATH` during review; do not assume it is available there. Merely reshaping audio or casting it to `int16` does not resample it. Normalized floating-point audio also needs scaling and clipping before conversion to signed PCM16.

## 6. Timing and voice integration cautions

- Default `Writer(..., keeptime=True)` uses the type's 100 ms period. In a standalone publisher, let it pace the stream rather than adding a second unconditional frame-duration sleep.
- BBOS `Loop` state is **process-global, not thread-local**. A background speech worker inside a process that also owns motion writers needs a deliberate timing policy. Independent scheduling can use `keeptime=False`, but then the audio owner must supply the full frame cadence itself. Do not disable the global timing loop to fix audio.
- The current [SpeechQueue](speech_queue.py) subtracts sink-call time from an approximately 98 ms wait. The inspected [body implementation](mc_skills_main.py) creates its audio writer with default timing. Review that combination before assuming it is reliable during concurrent motion; this review did not test physical audio/motion concurrency.
- Keep TTS/network work off the motion thread. Keep the speech queue bounded; the existing queue retains one active line and at most one pending line. Cancellation cannot retract audio already delivered into downstream playback buffers.
- [tts.py](tts.py) produces raw PCM bytes; `frames()` pads them to 3,200-byte chunks. The sink converts a chunk using `np.frombuffer(chunk, dtype=np.int16).reshape(-1, 1)` and fills the existing writer's `audio` field.

### Deployed code is not necessarily the local copy

Voice updates arrived during this review. The current repository version and the inspected `/home/bracketbot/bbapps/mc_skills/tts.py` both support cached WAVs, OpenAI TTS with 24 kHz-to-16 kHz conversion, and a macOS `say`/`afconvert` fallback. Older local revisions supported only macOS synthesis. The newer cache tag includes OpenAI and macOS voice settings, so older cache files or different voice settings do not guarantee a cache hit. A cache miss on the robot can cause a billable TTS request.

Compare the deployed and local implementations before changing the voice pipeline. [deploy_to_robot.sh](deploy_to_robot.sh) overwrites files in `mc_skills`; do not run it blindly and replace bot-only fixes. Keep credentials out of the guide, source files, and diagnostic output; use the existing [bot key setup](../../README.md#model-key-and-developer-access).

## 7. Troubleshooting by layer

| Symptom | Check |
|---|---|
| `Writer for speaker.audio already exists (pid=...)` | Another publisher owns the topic. Use that owner's queue or coordinate a handoff; do not kill robot-control processes to free audio. |
| HTTP `/say` returns OK, but silence | Confirm non-mock operation, a connected audio sink, successful TTS/cache lookup, and actual frame delivery. HTTP acknowledgement is not playback verification. |
| First word is missing | Check writer discovery/warm-up and whether the writer is repeatedly recreated. Keep it persistent. |
| Audio is slow, pitched incorrectly, or garbled | Check 16 kHz, mono, PCM16, the exact frame shape, and absence of WAV/MP3 headers in the payload. The daemon handles the later conversion to 48 kHz. |
| Short final chunk fails assignment | Zero-pad to `(CFG.chunk_size, CFG.channels)` before publishing. |
| Choppy audio or accumulating delay | Check approximately 10 publications/second, queue growth, process-global timing interactions, and duplicate pacing. Do not blast an entire clip into the shared-memory slot. |
| Samples arrive but no device output | Inspect the speaker daemon and its selected PulseAudio server/sink. The daemon selects the head-board sink and runs `pacat`; an unrelated default desktop audio device is not the same path. |
| Unexpected loudness or silence despite valid samples | The daemon applies gain, optional compression, and `speaker.volume` overrides (`gain`/`mute`). Its configured gain default was 0.6; a live override can differ. |

Use `list --plain` and `logs speaker` in the configured bot shell for diagnosis. Inspect logs privately and do not paste credentials or full provider-error dumps into chat. Do not run the speaker daemon directly as a test: its startup/recovery code can change PulseAudio modules, volume, and playback processes. Leave daemon recovery and loaded-robot service changes to the platform/action owner.

## Sources reviewed

Paths below are on the bot, relative to `/home/bracketbot/`:

| Source | Relevant behavior |
|---|---|
| `bbapps/AGENTS.md` | IPC contract, timing, single-writer rules, and diagnostic CLI |
| `bbapps/play_sound/main.py` | WAV loading, microphone startup wait, writer discovery delay, padding, and publication |
| `bbapps/play_sound/web_ui.py` | Soundboard subprocess behavior |
| `bbapps/examples/view_speaker.py` | Synthesized PCM16 chord playback |
| `bbapps/greeter/main.py` | `audio_io_loop`, persistent queues, silence, and model-audio conversion |
| `bbos/bbos/daemons/speaker/constants.py` | Speaker configuration, `speaker_audio`, and `speaker_volume` schemas |
| `bbos/bbos/daemons/speaker/daemon.py` | Gain, compressor, 16-to-48 kHz resampling, and head-board output selection |
| `bbos/bbos/ipc.py`, `bbos/bbos/time.py` | Writer ownership, buffer publication, lifecycle, and global timing |
| `bbapps/mc_skills/main.py`, `bbapps/mc_skills/tts.py` | Deployed `/say` route, audio sink, and TTS behavior |

Local integration references: [mc_skills_main.py](mc_skills_main.py), [hook.py](hook.py), [speech_queue.py](speech_queue.py), and [tts.py](tts.py).
