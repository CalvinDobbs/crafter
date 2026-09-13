# BracketBot (bbcore) — app development notes

The git repo `Bracket-Bot-Inc/bbcore` is cloned directly into `$HOME` (git dir at `~/.git`, branch `develop`). Two halves:

- `~/bbos/` — "BracketBot OS": IPC middleware + per-subsystem daemons (`bbos/bbos/daemons/<name>/`)
- `~/bbapps/` — user apps (`~/bbapps/foo.py` or `~/bbapps/foo/main.py`)

Robot: mobile manipulator, Jetson Orin Nano (JetPack 6, CUDA 12.6), 2-wheel self-balancing drive base (STM32 baseboard + ODrive), two 8-DoF arms (`arm_left`, `arm_right`; index 7 = gripper), 3 cameras (head stereo 2560x960, 2 wrist 640x480), mic/speaker, LED, Meta Quest teleop.

## bbos IPC (the API surface for apps)

```python
from bbos import Reader, Writer, Type, Config
```

- Topics = POSIX shm segments at `/dev/shm/<topic.name>`; lock-free ring buffers, single writer, many readers.
- `Writer(name, Type("<type>"), keeptime=True, buf_ms=0)`: `with w.buf() as b: b["field"]=...` or `w["field"]=v`. `w.ready()` → is it this writer's turn.
- `Reader(name, keeptime=True, sync=False, aligned_to=None, decimate=1)`: `r.ready()` → fresh data (handles writer death/recreate); `r.data["field"]`, `r.data["timestamp"]`.
- `sync=True` = consume every message in order (needed for streams like `mic.audio`, `usb.tree`). `aligned_to=<Reader>` = sample closest to another reader's timestamp (cross-topic sync). `decimate=N` = every Nth with sync.
- `keeptime=True` (default) = global `Loop` paces reads/writes to the type's `@realtime(ms)` period. `keeptime=False` + `time.sleep` to self-pace.
- Types declared in `daemons/*/constants.py`: `@realtime(ms)` = periodic, `@state` = unpaced. `timestamp` (datetime64[ns]) field auto-appended. `@register class` = config. `Config("<daemon>")`/`Type("<type>")` resolve lazily — all constants.py are AST-scanned, topo-sorted by Config deps, and exec'd on first use.
- ONE writer per topic — a second `Writer` raises RuntimeError(pid). Daemons react to writer presence: `led` reverts to idle ~3s after last `led.ctrl` write; arm daemon discovers `.ctrl`/`.torque` writers; speaker daemon must discover a new `speaker.audio` writer (sleep ~0.5s after opening).

## Topic catalog (verified live on this robot)

Drive/base (daemon `base`):
- `drive.ctrl` (app→robot, 10ms): `twist` [v m/s, w rad/s] f32, `twist_torque` f32[2]
- `drive.state` (10ms): pos/vel/torque/ff/ctrl/iq/gains f32[2], pos_estimate, yaw_error
- `drive.status` (10s): `voltage`, `errors`[2], `loop_hz`
- `base.mode` (50ms): `mode` (0=balance,1=lean,2=twist), `lean_angle_deg`; `base.health` (1s)
- `imu.orientation` (10ms): `rpy` rad; `imu.raw` (10ms): accel, gyro; `imu.diagnostics`
- `Config("drive")`: robot_width=0.3275, wheel_diam=0.165, max_linear_vel=0.3, max_angular_vel=0.9

Arms (`arm_left`/`arm_right`, dof=8, units = motor turns; index 7 = gripper):
- `<arm>.state` (15ms): pos/vel/torque/temp/current f32[8]
- `<arm>.ctrl` (15ms): pos/vel/tau f32[8], `alpha` f32 (per-cmd smoothing override)
- `<arm>.torque` (@state): `enable` bool[8], `tau_mode` bool[8], `compliance_mode`, `axis_aligned`, `force_only`, `j0_homing`, `calibrating`
- `<arm>.target` (15ms): pre-IK cartesian setpoint xyz+quat+grip+tracking (dataset only)
- IK: `cfg = Config("arm_left")`; `cfg.ik.solve(pos[3], quat_xyzw[4]) -> 7 joints`, `cfg.ik.fk(q)`, `cfg.ik.reset(q)`, `cfg.ik.set_nominal(q)`. Also `cfg.home`, `cfg.startup_waypoints`, `cfg.q2urdf`/`urdf2q`, `cfg.joint_names`, `cfg.gripper_sign` (left=-1)
- Per-robot calibration: `~/bbos/bbos/daemons/arm_<side>/ranges.calibration.json` (`cal_min`/`cal_max` turns); normalize to [-100,100] joints / [0,100] gripper — see `bbapps/inference/bracketbot_adapter.py`
- Homing/parking helpers: `bbapps/quest_teleop/scripts/homing.py` (`staged_home_arms`, `park_arms`); inlined copy in `bracketbot_adapter.py`

Cameras (`camera` daemon): `camera.head.rgb` (stereo u8 960x2560x3; `Config("cam_head").split(img)` → (left,right)), `camera.head.jpeg`, `camera.left.jpeg`, `camera.right.jpeg` (`jpeg_len`+`jpeg`), `camera.<name>.status` (streaming,fps)

Depth (`depth` daemon): `camera.depth` u16 mm, `camera.rect`, `camera.points` (num_points, base-frame xyz, mask). `Config("depth")` → `camera_cal()`, `T_base_cam`.

SLAM/mapping: `slam.pose` (pos+quat map frame, vo_* odom, pgo_count), `slam.health` (@state: degraded/stalled/vo_lost/localized/relocalized), `slam.history_generation`, `slam.trace`; `mapping.grid2d` (grid,origin,robot_pos,robot_heading; 200ms), `mapping.voxels` (500ms)

Audio: `mic.audio` int16 (1600,1) @16kHz, 100ms chunks — open `sync=True`; `mic.ref_level` dbfs; `speaker.audio` write int16 chunks (1600,1), paced 100ms; `speaker.volume` (@state: gain,mute); `wakeword.state` (active bool, 1Hz)

LED: `led.ctrl` (rgb u8[3], brightness i16 -1=default, period_ms u16), `led.state`

Quest: `quest.controllers` (20ms: T_head 4x4, left/right_pose[7] pos+quat, trigger/squeeze/thumbstick[2]/buttons), `quest.joystick`, `quest.link` (@state: connected), `quest.haptic` (@state write: hand,frequency,amplitude,duration)

Recording/cloud: `dataset.flag` (@state write: prefix S200, name S200, text S500, toggle_episode, drop_episode) → dataset daemon records `Config("dataset").sensor_channels` + camera JPEGs → npz/mkv → api.bracketbot.com. `telemetry.flag` similar. `usb.tree` (sync=True stream).

## Daemons

Each `daemons/<name>/`: `constants.py`, `daemon.py` (entry; argv[1]=name), `devenv.nix`+`pyproject.toml`+`uv.lock`, optional `driver.py`/`calibrate.py`/`tests/`. `manager.service` (systemd) runs each: `uv run --frozen --no-sync python daemon.py <name>`, env cached in `.devenv/bbos-env.json` (auto-refresh on devenv/pyproject changes). Markers: `.stopped`, `.disabled`. Auto-restarts on crash. Logs → `/dev/shm/<name>.log`.

## Apps

- `app.py` or `<dir>/main.py`. PEP 723 header required:
  ```python
  # /// script
  # dependencies = ["bbos", "numpy"]
  # [tool.uv.sources]
  # bbos = { path = "/home/bracketbot/bbos", editable = true }
  # ///
  ```
- Run: `run <app>`, `uv run ~/bbapps/<app>.py`, or toggle via app_manager (`app_manager.service` watches `/dev/shm/app-<name>_lock`; file contents = args). Logs → `/dev/shm/app-<name>.log`.
- `~/bbapps/.autostart` — one app per line (+args) starts on boot.
- An app-local `.venv/` in the app dir is used instead of `uv run` if present.
- Reference apps: `teleop.py` (WASD web teleop, FastAPI), `examples/view_*.py` (minimal readers per topic), `mimic/main.py` (arm record/playback), `nav/main.py` (nav stack), `inference/` (`live_inference.py` + `bracketbot_adapter.py` = canonical arm/camera Robot adapter for remote policy servers over gRPC).

## CLI (built by daemons/manager; `rebuild` after editing it)

`help`, `list` (TUI of topics/readers/rates; `list --plain`), `types [name]`, `constants [name]`, `logs [daemon]`, `restart [daemon...]` / `stop [daemon...]`, `activate`/`deactivate <daemon>`, `calibrate <daemon>`, `run <app>`, `clip <note>`, `login` (bb-cloud → /etc/BB_API_KEY), `rebuild`.

## Safety conventions

- Leave arms limp on exit: `finally:` write `enable=zeros(dof)` to `<arm>.torque`.
- Before enabling torque, disable first and flush `ctrl` to live `state.pos` (daemon reseeds its cmd filter only on OFF→ON transitions).
- `Reader.ready()` is False while the writer is down — gate actions on it.
- `BBOS_VERBOSE=1` → loop-lag + dropped-frame logs.
- Don't open a `Writer` for a control topic unless your app intends to own it (collides with running teleop apps; a live `.ctrl` writer signals intent to the daemon).

## Debug

- `logs <name>` or `/dev/shm/<daemon>.log`, `/dev/shm/app-<app>.log`
- `list` — topics, writers, readers, real rates
- `stop` (no args) kills manager+app_manager and wipes /dev/shm topics; `restart` brings services back
