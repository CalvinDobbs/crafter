# Perception handoff: GPU sensing for the agent and preset actions

This is the handoff for the working perception implementation in `perception.py`, checked against the repository on 2026-09-12. It describes existing outputs, their limitations, and the thin integration work an actions/reasoning agent may need. **It is not permission to rewrite the perception stack or change motion routines.**

Read this together with `agent_types.py`, `agent_adapters.py`, the **Agent component interface v2** section of `README.md`, and `../AGENTS.md`. Those define the current controller/provider contract. This document adds perception-specific semantics and the requested conservative box-selection policy.

**Working-tree handoff:** `perception.py` already had local uncommitted changes when this documentation task began. They were preserved unchanged. This describes the bot's working implementation, which may be newer than a fresh checkout of `origin/main`. Coordinate saving/transferring that exact implementation with its owner; do not reset, checkout over, or replace the working GPU stack to make it match an older remote version.

## 1. Start here: preserve these invariants

1. **Keep inference local on the Jetson GPU: TensorRT, FP16.** The laptop is a viewer/development client, not the detector host. Pass the GPU flags explicitly; some constructor/CLI defaults are still CPU or marker-only.
2. **Reuse one long-lived perception owner.** Prefer consuming the existing service. Do not create a new model/session per MCP call, per action, or per camera frame; do not run a second GPU pipeline just to give another agent access.
3. **Use structured observations, not overlay text, to choose targets.** An object has a detector score, timestamp, identity status, and localization quality. A colored rectangle is not movement authorization.
4. **Apply higher confidence thresholds in an adapter/selection policy.** Recommended initial policy: at least **0.25 for target selection**, preferably **0.30 before pickup**, with the other checks below. Do not silently fall below **0.20** to make a demo proceed.
5. **A high score does not prove a single physical box.** A group of nearby objects can form a cuboid-looking silhouette and trigger detection. This is an observed failure mode, not a hypothetical one.
6. **Surface coordinates are not grasp poses or complete box centers.** `grasp_pose` is not supplied. Automatic objects deliberately have `pick_candidate=false`.
7. **Only the designated action owner controls the robot.** Perception reads sensors. It does not drive, home, grasp, release, park, or provide a physical emergency-stop implementation.
8. **Do not turn stale or missing evidence into a fresh success.** Preserve capture times, frame/epoch, identity uncertainty, unknown occupancy and unknown possession.
9. **Do not modify frozen/shared files to bypass integration checks.** Leave `contracts.py`, `WIRE_FORMAT.md`, the working detector, calibration, GPU configuration and teammates' action code alone unless the responsible owner explicitly agrees.

## 2. What actually runs

```text
Jetson sensor topics
  camera.rect.left + camera.points/idx_2d + wheel/IMU state
        |
  Reader-owning capture thread, paced large reads, latest packet only
        |
  TensorRT FP16 worker: full frame + near-field tile
  CPU depth association on the SAME captured frame
        |
  session-local identities, pose-aware world positions, quality metadata
        |
        +--> Scan / JSON observations --> adapter --> agent's legal choices
        |
        +--> bounded WebSocket image + metadata --> debug browser

Agent-selected semantic intent --> validated ActionRequest --> preset action owner
```

The model is **YOLOv8s-Worldv2**, an open-vocabulary small YOLOv8-family detector. The current vocabulary contains `cardboard box` and distractor classes (`person`, `chair`, `backpack`, `laptop`, `table`). Only cardboard-box proposals are returned as automatic objects. This is **not a complete obstacle/person detector**.

CLIP text embeddings are cached; a large language model does not run per video frame. Model assets are pinned/checksummed. The TensorRT engine uses fixed shapes and is keyed by model, vocabulary/embedding hash, precision, input shape, device and runtime.

The current tested deployment uses:

- Jetson Orin Nano Super, CUDA 12.6, TensorRT 10.3 and the compatible Python 3.10 bindings.
- A nominal **512 x 384, 10 Hz** rectified/depth source.
- Detector size **512** and two image passes; the close-up pass helps small boxes.
- One in-flight detector packet with matching RGB, sparse depth indices, timestamp and pose.
- Separate acquisition and processing; visualization-map rebuilding at **2 Hz**.
- Display-only optical-flow propagation for at most **350 ms**. It does not refresh measured object evidence.

### Measured reference, not a universal guarantee

On the implementation test scene, two-pass CPU detection had about **1033 ms median** latency versus **35.5 ms median** for TensorRT FP16. A live sample with stereo depth running delivered approximately **8.9 distinct camera frames/s**, **7.9 detections/s**, and **327 ms p95 capture-to-object age**. Numerical CPU/GPU parity was checked on one scene and image variants; that is not a complete box-recognition accuracy evaluation.

These are useful regression references. Scene complexity, other GPU users, memory pressure, network and browser rendering can change them. Do not equate display FPS with detector FPS or a p95 statistic with a guaranteed safety response interval.

## 3. Operate the existing GPU service; do not launch duplicates

All `/home/bracketbot/...` commands below run **on the bot**, not on a developer laptop. `localhost:8007` on a laptop means that laptop unless an SSH tunnel/preview forwards it.

First inspect the service with bounded read-only requests:

```bash
curl --fail --max-time 2 http://127.0.0.1:8007/scan
curl --fail --max-time 2 http://127.0.0.1:8007/objects
```

If it is not running, coordinate with its owner before starting one copy:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
/home/bracketbot/.local/bin/uv run --offline \
/home/bracketbot/crafter/bot-code/perception.py \
  --viz --detector boxes --detector-backend tensorrt --precision fp16 \
  --detector-size 512 --port 8007
```

`uv` may not be on PATH in non-interactive SSH sessions. Use the absolute executable above. Use the existing compatible environment; do not upgrade Python, replace JetPack libraries or install a generic GPU wheel to work around an environment error. `--offline` assumes the dependencies are already cached.

Confirm **actual runtime**, not merely requested flags:

- `mock == false` and `stale == false` in `/scan`.
- `detector_status.state == "ready"`.
- `detector_status.runtime.backend == "tensorrt"`.
- `detector_status.runtime.precision == "fp16"` and the reported device is the Orin.
- Fresh pose/depth, plausible `detector_fps`, result-age metrics and capture statistics.

The human-readable `detector` string can describe the **requested** backend. It is not proof that the engine loaded. CPU is an explicit reference option, never an acceptable silent fallback for this deployment.

The service was not reachable on port 8007 during this document's final source audit; no restart was performed for this documentation task. Always inspect the real state rather than assuming a prior preview is still running.

### Cache and resource rules

Assets, cached embeddings and engine/manifest files live under `~/.cache/crafter-perception/yolo-world-v2/`. Do not copy an arbitrary engine from another machine or rename one to fit a cache miss. Shape, vocabulary, precision or runtime changes can invalidate the engine cache.

`--prepare-detector` downloads assets; `--prepare-gpu-detector` compiles an engine. These are setup operations, **not per-tool/per-action operations**. A fresh engine build previously required about 1.6 GiB of temporary host memory; the finished engine was about 28 MiB. CPU/GPU share this board's RAM. Coordinate build windows and leave the memory guard intact. Do not kill other SSH/IDE sessions, stop robot daemons, add swap, change clocks or rebuild engines automatically to recover a tool call.

## 4. Current interfaces

Perception exposes **HTTP, WebSocket and Python APIs**, not an MCP server implementation. The teammate's agent can wrap these reads in its existing MCP/tool layer. Keep that adapter separate from model execution and motion.

| Interface | Existing behavior | Integration guidance |
|---|---|---|
| `GET /scan` | Scene JSON, HTTP schema version **2** | Best single request when the consumer needs objects, pose, build frame and runtime/freshness together. |
| `GET /objects` | Smaller object envelope, HTTP schema version **1** | Contains `snapshot_ts`, `pose`, `objects`, `detector_status`, `limitations`. It does not include every `/scan` health/build field. |
| `GET /frame?view=...` | JPEG for `live`, `rect`, `raw`, `range`, or `boxes` | Header `X-Frame-Timestamp` is the image acquisition time. First request to an unsubscribed view may return 503 while the next image is requested. |
| `WS /live?view=...` | Pushes selected image and scene metadata | Views: `camera`, `live`, `rect`, `raw`, `range`, `boxes`. `camera` is an auto selector, not a valid HTTP `/frame` view. |
| `PerceptionSession(...).poll()` | Latest persistent `Scan`, or `None` when there is no new result | Create/poll/close through one owner. `poll()` is not a request to drive or rescan physically. |
| `PerceptionSession.close()` | Closes its own capture/inference resources | Does not stop an action or make a held load safe. |
| `POST /mock/{action}` | Simulation controls only | Live mode rejects these with HTTP 403. They are not physical robot skills. |

HTTP schema versions are **not** `agent_types.INTERFACE_VERSION`. The current action/observation provider contract is **v2** independently of these JSON envelopes.

### `/scan` top-level fields worth preserving

- `schema_version`, `mock`, `ts`, `published_at`, `stale`.
- `frame="base_at_capture"`, `pose` including `x`, `y`, `yaw`, `ts`, `valid`, `epoch`, source/warning.
- `objects`: rich automatic-detector objects, including 2D-only or remembered entries.
- `tracks`: normalized map tracks, including localized automatic objects and marker observations.
- `build`, `anchor_seen`, `surface_cells`, `settings`, `warnings`.
- `detector_status`, `diagnostics`, `telemetry`, `streams`, `live_overlay`.
- Legacy `boxes`, `protected`, `unknown` lists also remain. **Do not assume `boxes` is the complete automatic-detector inventory.** Use `objects` or `tracks`.

`published_at`/HTTP receipt time is not the measurement time. `/objects` does not have top-level `mock` or `stale` flags; an adapter consuming only it must validate pose/timestamps and establish that it is using the real service. Prefer one atomic `/scan` response for live-action admission, including `mock=false` and actual GPU runtime checks, rather than caching a separate health check across service restarts.

Useful diagnostics include `point_count`, `rect_shape`, `pose_camera_skew_s`, `depth_yaw_deg`, capture/replaced-frame statistics, processing timings and detector result ages. `live_overlay` (also `image.overlay` for a live stream image) exposes `source` (`none`, `detected`, `visual_tracking`), `detected_at`, `display_frame_at` and `count`. A display overlay count is not an inventory or placement count.

### Automatic object fields

| Field | Meaning |
|---|---|
| `id` | Integer identity used for controller integration; automatic allocation starts at 1000. |
| `track_id` | Human-readable name such as `box-001`; not a substitute for the integer ID. |
| `tracker_session`, `pose_epoch` | Identity/coordinate scope. Do not carry identities across a restarted service or changed epoch without reconciliation. |
| `label` | Currently `cardboard_box`; a model proposal, not a verified physical fact. |
| `bbox` | `[x1, y1, x2, y2]` in the **original rectified image's pixels**, after undoing detector resizing/crop offsets. |
| `score` | Latest raw detector score, not an averaged historical maximum or calibrated probability. |
| `last_seen` | Acquisition timestamp of the neural observation, in Unix seconds. |
| `confirmations` | Cumulative matches during the track's life. **Not** a guarantee of consecutive high-score frames. |
| `identity_status` | `new`, `tracked`, `ambiguous`, or `pose_epoch_changed`. `tracked` is useful evidence but not perfect re-identification. |
| `observed_position_base_m` | Surface estimate in the base frame at the original observation. Never use this as a fallback motion target when current geometry is invalid. |
| `world_position_m` | Surface estimate in this session's world frame; may be null. |
| `position_base_m` | World estimate re-expressed in the base frame at `pose_timestamp`; may be null. It is not automatically the pose at the instant an action starts. |
| `position_kind` | Currently `visible_surface_centroid`, **not** a box center, full 6-DoF pose or end-effector target. |
| `position_frame`, `pose_timestamp`, `snapshot_ts` | Explicit reference frame and timing of the transformed position/snapshot. |
| `depth_status`, `depth_points`, `support_clearance_m` | Localization quality and local support-plane information; see below. |
| `partial_view` | Detection touches the full image boundary. Treat as insufficient for automatic pickup until re-observed clearly. False is not proof of an unoccluded, complete single box. |
| `age_s`, `current`, `visibility` | Freshness/memory annotations. Validate independently; neither a recent image nor repeated polling refreshes `last_seen`. |
| `zone` | `unassigned`, `protected`, or `outside_build`. Outside does not mean untouched, reachable or safe. |
| `grasp_pose` | Null: no grasp pose is supplied by this backend. |
| `pick_candidate` | Always false for automatic proposals. This deliberate flag prevents raw detections from becoming implicit motion authority. |

Null positions, non-finite values, invalid pose and epoch mismatch must remain unknown/unusable for motion. Do not fill them with zero, a fixture coordinate, a prior pose from another epoch, or the center of the image.

### Depth and identity semantics

- `surface_supported`: depth samples above a fitted nearby support plane gave a surface estimate. This is the strongest currently exposed localization result, **not proof of a single box or a safe grasp**.
- `weak_no_support_plane`: an approximate surface estimate may exist, but the supporting plane was not validated. Not an automatic pickup candidate.
- `missing`: insufficient usable depth; keep the image detection as 2D-only.
- `background_or_flat_surface`: depth did not distinguish an object above the surrounding plane.
- `inconsistent_depth`: insufficient coherent depth after filtering.

The localizer samples an inset rectangle, fits nearby support where possible, and rejects inconsistent ranges. It is **not instance segmentation**. Multiple nearby objects can still be included in a rectangular region and can share a support plane.

The current settings mark evidence current within approximately **0.8 s**, retain object memory up to **30 s**, and expire a remembered anchor after **15 s**. Association also has shorter age gates. HTTP `visibility="recent_detection"` uses a three-second display-oriented window; **that does not override the stricter `current`/freshness requirements**.

Even `current=true` must be combined with valid pose, non-null geometry and matching epochs. For an action decision, also require the object to have been seen in the latest completed detector frame from the same response, rather than accepting an older high-score track that the latest inference missed.

## 5. Coordinate and image discipline

- Base coordinates: **+X forward, +Y left, +Z up, meters**. Yaw is radians.
- World coordinates: the session-local odometry frame, initially aligned to the base. Wheel/IMU dead reckoning drifts; this is not globally localized SLAM.
- Each live capture session gets a unique JS-safe integer pose epoch. Resets/gaps change the epoch. All site, target and action geometry must use compatible `(frame_id, epoch)` values.
- Depth image and sparse pointcloud correspond through `idx_2d[:num_points]` and the shared capture timestamp. Never reshape the sparse cloud into an image or sample depth from a different capture just because it is the latest one.
- Perception already normalizes the depth heading and converts the IMU publisher's degrees into internal radians. **Do not repeat those transforms/conversions in the adapter.**
- `normalize_scan` exposes automatic world coordinates as `frame_id="session-world"`, with base position/yaw. Preserve them rather than substituting base-relative coordinates into the world field.
- Minecraft cells are `(x, layer, z)`. Physical XYZ and schematic layer coordinates are different. `BuildSite.cell_center(cell, requirements.voxel_size)` is the agent interface's site conversion, not a reason to rewrite the frozen global grid constants.
- Automatic track `size` is currently `0.0`, normalized to **unknown (`None`)** by the agent adapter. A configured mock/marker `BOX_SIZE` is not an automatic measurement of real cardboard boxes. Actual dimensions and action tolerances must come from operator measurements or a validated provider.

### WebSocket and imagery

Each `/live` binary message is:

```text
4-byte big-endian unsigned JSON byte length
UTF-8 JSON metadata of that length
JPEG bytes; metadata.image.bytes gives their length (possibly zero)
```

`image` includes `view`, `timestamp`, `revision`, `bytes`, and overlay information for the live view. Map updates may omit `surface_cells`; retain their world coordinates only within the same epoch and reproject using the new pose. Keep one latest packet/decode slot, not an ever-growing queue. The server limits concurrent stream clients and rejects unsupported inbound messages; changing views uses a new subscription URL.

- `live`: current rectified frames. Green detections are measurement evidence; amber `tracked` overlays are short-lived optical-flow predictions.
- `boxes`: original annotated neural-detection snapshot, potentially delayed.
- `rect`: rectified image with marker outlines.
- `raw`: wide left head image with different lens/resolution geometry. Do not naively scale rectified bounding boxes onto it.
- `range`: distance from camera, not height above floor. Black means missing/filtered depth, not empty space.

The pushed image and publication metadata share an envelope, but a remembered object's `bbox` still belongs to its own `last_seen` detector capture. **Do not assume every object in a snapshot was detected in the pushed image.** Check the object's timestamp against the image/detector timestamp before making a crop. Separate HTTP image/JSON requests may observe different revisions.

For a v2 `SceneImage`, use actual fresh PNG/JPEG bytes as an inline base64 data URL with the true capture time, view, frame/epoch and simulation flag. The interface allows **at most three images, 512 KiB each**. A local path, ordinary HTTP URL, generated grid picture or old screenshot is not a live `SceneImage`. The generic `normalize_scan`/`ObservationWorker` does not populate images, but the newly added **`PerceptionObservations` HTTP adapter does**: it brackets a rectified JPEG request with `/scan` reads and validates capture time/epoch. Reuse it rather than reimplementing image delivery.

## 6. Requested higher-confidence selection policy

**This section is a recommended adapter/action-admission policy. It is not implemented by changing `perception.py` in this handoff.**

The detector currently admits proposals at **0.10** for recall/debugging. Keep those observations available for diagnostics. Apply stricter filtering before the controller offers a box as a legal material target:

| Use | Recommended starting minimum |
|---|---:|
| Autonomous target selection / approach candidate | **0.25** |
| Revalidation immediately before pickup | **0.30**, plus all physical preconditions |
| 0.20 <= score < 0.25 | Re-observe or require explicit operator confirmation; not the normal autonomous path |
| Below 0.20 | Do not automatically go after the object |

These values reflect the requested 0.20–0.30 range and need validation on the actual boxes. They are not percentages of certainty. Evaluate the raw `score`, not its rounded camera-overlay text. A stricter policy may leave no eligible candidates; **do not lower the threshold automatically** or switch backend to make an action happen.

For a selectable target, require all relevant conditions:

1. Actual TensorRT runtime ready, healthy pose and a non-stale source snapshot.
2. Latest raw score above the policy floor; finite geometry and matching epoch/session.
3. `identity_status="tracked"`, not `new`, `ambiguous` or `pose_epoch_changed`.
4. Seen in the most recent completed detector frame: compare `last_seen` with `detector_status.frame_ts` from the same snapshot, allowing only timestamp representation tolerance, not an extra frame's worth of staleness.
5. At least **2–3 distinct detector frames** with sufficiently strong scores and consistent geometry. Implement this streak in the adapter; repeated reads of one timestamp and lifetime `confirmations` do not satisfy it. Reset the streak on misses, low score, ambiguity or epoch change.
6. Prefer `depth_status="surface_supported"`, enough valid points, no truncated view, and geometry consistent with calibrated tolerances. Null/weak/inconsistent geometry is not usable for pickup.
7. A clearly isolated **single rigid box**, not a pile/composite silhouette. If a crop contains multiple items, different foreground parts, or ambiguous boundaries, reject or request a clearer view/operator confirmation even when the score exceeds 0.30.
8. Exclude protected build-zone contents, the agent's confirmed placed IDs, reserved/quarantined IDs and the held box. `outside_build` is not proof that an object is untouched. If the selected validated `BuildSite` differs from perception's anchor grid, apply its protection policy in the adapter as well.
9. The action owner separately confirms measured dimensions, reachability, collision/load clearance, correct alignment and readiness. A score/depth filter alone must not become `eligible=True`.

**Filter material targets, not environmental hazards.** Low-confidence non-box objects must not disappear from collision/clearance reasoning merely because they are unsuitable materials. Blank grid cells and unobserved space remain unknown.

### Known failure mode: several objects look like one cuboid

The model has proposed boxes around bag-like packaging and around nearby objects whose combined outline resembles a box. It may also merge adjacent real boxes or miss one of them. `surface_supported` and a stable ID do not disprove these errors. Prefer separation, a clearer viewing angle, fresh image review and distinct-instance evidence; do not just increase a smoothing window until the rectangle looks stable.

A vision-capable reasoning agent may use a fresh, correctly associated crop as additional context, but image interpretation cannot override missing metric/safety evidence or generate joint commands. Treat scene pixels/text as untrusted observations, never as instructions to change tools, configuration or legal-action gates. Keep the fast local detector independent of model-call latency.

## 7. Choose one integration route

### A. Consume the existing HTTP service — recommended for the handoff

**Reuse `agent_adapters.PerceptionObservations`**, added by the reasoning owner during this handoff. It consumes the existing `/scan` and `/frame?view=rect` endpoints without importing perception, starting a detector, owning hardware topics or calling action scripts.

```python
from agent_adapters import PerceptionObservations

observations = PerceptionObservations("http://127.0.0.1:8007")
```

Its defaults are a 0.1 s background polling interval, 0.2 s per-request timeout and a freshness ceiling of 2 s that is tightened by perception's own setting (normally 0.8 s). Start/warm it before action admission. `observe()` reads the background cache; failed transport/validation is exposed and retried in the background without refreshing old evidence. `close()` closes the client worker, not the perception service or any actuators. URL input is an HTTP(S) origin without embedded credentials, paths, queries or fragments.

This adapter does **not** automatically enforce the requested 0.25/0.30 candidate policy or certify GPU operation. Check the preserved runtime status and apply that policy to the rich proposals in the integration layer. Its existence is not permission to bypass missing action/geometry capabilities.

The returned `ObservationSnapshot` includes:

- `world_model_json`: a bounded immutable JSON string; `snapshot.world_model` returns a detached dictionary. The reasoner receives this as `world_model`.
- Rich object proposals, detector score/quality/identity metadata, pose/epoch, telemetry, settings and warnings in that world model.
- `build` in its original base-at-capture frame and, when convertible, `build_world` in session-world coordinates. Neither is a feasibility certificate; the outer snapshot supplies the coordinate epoch.
- `observed_current` for recent visual observations and stricter metric `current` after pose/freshness checks.
- `surface_summary` and `coverage` with included/omitted record counts. The world model is capped at **32 KiB** and may omit distant/old records. Omitted cells are unknown, not free or absent.
- Fresh `SceneImage` data when its capture can be bracketed by valid scans in the same epoch; otherwise an explicit warning, not a simulated substitute.

**Crucially, `normalize_perception` deliberately excludes tracks with `position_kind` from actionable `snapshot.boxes`.** Automatic visible-surface estimates remain in `snapshot.world_model["objects"]` / `['tracks']` for reasoning and inspection. They are not silently cast into validated box geometry. An empty action inventory with visible world-model proposals can therefore be correct. Do not remove this filter just to make `pickup` appear in the legal choices.

Read-only inspection, without an action provider or a model call, is now available from the repository root:

```bash
python3 -B -S bot-code/main.py --observe-perception http://127.0.0.1:8007
```

It rejects conflicting execution modes and reports `execution_enabled=false`. Explicit `--planner llm` adds one read-only model decision and sends the image/world model to the configured provider, potentially incurring charges; it is not part of normal verification and still exposes only observe/stop decisions.

If MCP wrappers are needed, expose bounded reads around this adapter, such as `observe_scene` or `get_box_observation(id)`. These are **suggested wrapper names, not existing MCP tools**. Apply the higher-confidence candidate policy to rich world-model objects; retain missing capability/geometry as unknown. Do not poll full maps at the controller's 10 ms monitoring interval or create another GPU pipeline for each tool call.

### B. Own one persistent Python session — only if it replaces the service owner

`PerceptionSession` supports:

```python
PerceptionSession(
    mock=False,
    detector=True,
    detector_backend="tensorrt",
    precision="fp16",
    detector_size=512,
)
```

Create it once in a sensing owner, poll in the background and close it only through that owner's lifecycle. Do not run this beside an already running independent GPU service unless deliberately budgeting a second pipeline. Do not poll a shared session from multiple agent/tool threads.

`requested_views` is a supported subscription setting: a headless inventory owner can use an empty set; an image adapter can request only the views it needs, such as `rect` or `boxes`. Do not generate/decode every view on every agent call. The HTTP service already manages view subscriptions.

`scan()`, `detect_boxes()` and `scan_all()` are legacy one-shot/marker interfaces; they are not the way to obtain persistent automatic YOLO identities. `scan_all(sweeps>1)` intentionally refuses movement. Use the action owner's survey skill, then request a fresh observation.

### Stock adapter limitations and where to filter

There are two distinct normalization paths. The generic **`normalize_scan`** reads direct `Scan.tracks` and produces `BoxObservation` records. It preserves integer IDs, world positions, timestamps and `current`; `size=0.0` becomes `None`. Eligibility is unknown unless an explicit callback supplies it. `BoxObservation` itself does **not** retain score, identity status or all depth metadata. Do not assume this generic conversion validates a surface estimate for motion.

The HTTP-specific **`normalize_perception`** used by `PerceptionObservations` adds stricter live/schema/pose/freshness checks, preserves the rich bounded world model, and intentionally withholds automatic surface estimates from action geometry. Prefer this existing separation for the handoff. Evaluate proposal confidence in `snapshot.world_model["objects"]`; any later conversion to actionable geometry belongs to a validated provider, not the language model.

For a direct-session adapter, evaluate candidate policy before lossy normalization or retain a same-snapshot sidecar. Raw tracks include score/identity/depth status but omit `partial_view`, `depth_points`, `support_clearance_m`, `confirmations` and `tracker_session`. These already exist in `Scan.objects`; join within the same snapshot/epoch. Do not combine unrelated revisions or treat lifetime confirmations as a consecutive-confidence streak.

The generic `ObservationWorker` advertises **inventory only**; `PerceptionObservations` advertises **inventory and images**. Neither is a complete v2 action-ready observation provider. Warm/start background sensing before admission; do not interpret configurable startup waits as permission to block controller-critical calls.

## 8. What the complete agent provider still needs

The full controller expects `AgentProviders(actions, observations, close)` with interface **v2**. `PerceptionObservations` now provides live inventory context and inline images, but inherits unavailable `monitor`, `find_build_sites` and `check_build_site` operations that raise `CapabilityError`. It still does not provide verified occupancy, possession or phase-aware motion safety. Merely combining it with a live action provider does not satisfy full-build preflight.

Live controller integration uses `main.py --provider module:factory` and an explicit `--box-size` (currently one scalar uniform box edge in meters). Do not invent a measurement for this argument. The existing `--ui` mode uses real model decisions with **simulated tools**; it is not a live robot controller merely because the perception viewer is running. Use the component owner's verified provider factory, not an ad hoc import of hardware scripts.

| Requirement | What this stack provides | What must not be invented |
|---|---|---|
| Inventory | Object proposals, identities, approximate surface positions and quality | Suitable material dimensions/eligibility solely from a detector rectangle |
| Build site | Optional oriented anchor frame and surface observations | Full-build feasibility, route clearance, a valid marker-free site or an empty site |
| Occupancy | Debug height queries and observed surface cells | Complete per-cell occupied/empty/unknown evidence from the desired Minecraft target |
| Possession | No independent holding sensor output | `Holding(holding)` because a box vanished or a motion returned success |
| Monitoring | Pose/depth/object evidence that a real monitor can consume | `safe=True` based on a generic fresh-frame flag or high box score |
| Images | Timestamped JPEG/WS views; `PerceptionObservations` supplies a bracketed live rectified `SceneImage` | Gripper/build-site views not actually available, stale images labeled fresh, or images used as motion authority |

A markerless scene can have good box detections while `build` is null and zones remain `unassigned`. Do not create a fake anchor or change grid constants to bypass site preflight. A real registered site and its protection/occupancy checks must come from the responsible geometry/provider owner or an explicitly validated operator-assisted setup.

These gaps are not all "just missing fields." Exposing existing data is minor; creating real navigation clearance, grasp validation, site feasibility or possession evidence is separate functionality. **Do not advertise unimplemented capabilities or combine live actions with `MockAgentWorld` safety evidence.**

## 9. Using the preset skills/actions correctly

The model selects an allowed semantic `Step`; only the controller constructs and submits an `ActionRequest`. Applicable operations in the current interface are:

| Intent | Perception contribution | Action/controller responsibility |
|---|---|---|
| `observe` | Fresh inventory/images/quality | Decide whether evidence is adequate; no motion |
| `look_around` | Observe after each settled viewpoint | Motion owner performs a bounded, collision-checked survey; perception never rotates the base |
| `select_site` | Measured geometry if actually available | Choose/revalidate a feasible site for the whole build |
| `approach_box` | High-confidence fresh identity and approximate location | Recheck frame, route, dimensions and alignment; execute the validated approach preset |
| `pickup` | Revalidate the same isolated box before grasp | Execute the calibrated pickup routine; independently establish possession |
| `move_to_build` | Current localization/site evidence | Carry with verified held identity and load-aware clearance, not the old pickup coordinate |
| `place` | Fresh support/occupancy evidence if supplied | Execute calibrated release/retreat and obtain post-completion verification |
| `done` / `stop` | Evidence for final checks | Controller decides completion or load-preserving stop; model declarations are not proof |

### Current v2 action plumbing

- `FunctionActions` callbacks receive the **full `ActionRequest`**, not just `Step`.
- The envelope carries request/job IDs, step, observation reference, epoch/frame, submission time, requirements and selected box/site geometry. `expires_at` is the **admission** deadline, not the allowed execution duration.
- Callbacks return `FunctionResult(success, phase, error_code, effects_started)`. Independent `read_state` supplies readiness, phase, motion and `Holding`.
- `submit` is asynchronous/idempotent. Preserve the same request ID and payload; use `lookup(request_id)` after uncertain acknowledgements. Do not replay a timed-out pickup with a new request ID.
- `ActionOutcome.ts` is the phase-transition/terminal time. `observed_at` is a heartbeat. Polling a completed action must not move its completion time forward.
- `cancel`/`stop` must interrupt future commands while preserving a load. Acknowledgement is not proof of stopped motion.
- Provider status/state/observation/monitoring/site/stop calls have a documented **250 ms** return-or-raise budget. This is not permission to run an entire preset, model request or camera startup inside a tool callback.
- After model latency, revalidate selected evidence before dispatch. During motion, a real phase-aware `monitor(request, outcome)` returns `MotionObservation`, not an LLM-generated generic safety boolean.

### Important warning about the checked-in pickup script

At the time of this handoff, `actions/pickup.py` is a standalone supervised joint-space routine. `main()` constructs arm Writers and runs its calibrated sequence. By default it lowers and holds; `--pickup` opts into cage/lift. **Its `finally` block disables arm torque, and its own output warns that exiting releases the box.**

Consequently:

- Do not import/launch that script inside reasoning tests or every MCP call.
- Do not treat killing the script, timeout, Ctrl+C, or its normal exit as a load-preserving agent stop.
- Do not pass a perception surface point to it as if it accepted a generic Cartesian grasp pose; its documented CLI is joint-space/preset-oriented.
- The action owner must expose the tested routine through a real v2 action service/wrapper with its own preconditions, progress, independent state and cooperative load-preserving stop. If a newer action service exists, use its documented interface instead.
- Do not start `mc_skills`, another pickup script or another teleop arm writer beside the designated owner. bbos allows one Writer per control topic.

The legacy `skills_client.py` targets an HTTP body service on port 8006. Its outer `ok=true` does not necessarily mean motion succeeded: nested `False` or `(False, reason)` results must be normalized correctly (`normalize_skill_result` exists) and still do not prove possession or placement. Do not assume every planned semantic action is implemented merely because a legacy method name exists.

## 10. Real-time action loop: evidence, not debugging cosmetics

1. Keep sensing and the action owner's state/monitoring active in the background.
2. Get one consistent snapshot. Filter only target candidates using score, freshness, identity, depth, zone and independently validated material/action constraints.
3. Ask the reasoner to choose among legal IDs/actions. Do not ask it for joint angles or arbitrary motor commands.
4. Re-observe after reasoning latency and before admission. Reconfirm the same identity and candidate policy; reject an expired request or changed epoch.
5. Dispatch once. Monitor the real operation/phase without blocking on another model call.
6. Treat grasp occlusion as expected only when the active phase and independent holding evidence support it. Do not interpret disappearance as either success or failure by itself.
7. After pickup, bind the logical held identity to verified possession. A new detector ID on the carried box must not silently replace it or become a new loose target.
8. Before placement, validate the selected site, target/support cells, held identity and actual approach state. Protected does not mean placed.
9. After terminal release/retreat, require settled cell/holding evidence **measured after `ActionOutcome.ts`**. Only then update the controller's confirmed-cell ledger.
10. On stale/unknown monitoring, epoch change, uncertain dispatch or possible load loss, use the action owner's stop path and retain unresolved state for the operator. Perception cleanup cannot resolve it.

`verify_pick()` currently returns false for an unchanged visible marker or unknown otherwise; it is not a generic automatic-box possession checker. `verify_place()` is a height-consistency heuristic and cannot prove identity, complete occupancy or a successful release. Passing an existing snapshot avoids starting a new session, but does not upgrade either helper into sufficient action evidence.

The current controller starts from a verified empty site. It does not automatically resume a physical partial stack after a process restart. Reconcile outstanding actions, possession and site contents instead of resetting the ledger to make progress appear possible.

## 11. Minor data exposure changes that may be justified

Prefer adapter-only work. If a required existing datum cannot be obtained consistently, ask the perception owner for the smallest additive read-only exposure:

| Need | Preferred solution | Invariant |
|---|---|---|
| Rich confidence/identity fields after normalization | Use the existing `snapshot.world_model` from `PerceptionObservations`; for direct sessions join `Scan.objects` with raw tracks | Do not promote surface proposals into validated action geometry or edit the detector to add a confidence policy |
| v2 inline images | Reuse `PerceptionObservations`; add another bounded existing view only if needed | Preserve capture time/view/frame/epoch; do not use a prediction as new measurement |
| Existing RGB/depth/pose for a geometry provider | Bounded read-only snapshot/ROI accessor over the already matched packet (`last_packet` / `Scan.points`) | Copy/read-only ownership, explicit acquisition metadata; no full-cloud JSON stream per agent poll |
| Runtime health/latest detector-frame time in a direct adapter | Read/attach the already computed status after the single owner's poll, or consume atomic `/scan` | No second inference process; no field from a different revision mislabeled as current |
| Missing `mock`/`stale`/build metadata in the smaller `/objects` envelope | Prefer `/scan`; if needed, request additive exposure of the already computed fields from the same snapshot | Do not infer live mode from a URL, or combine a new object list with an old health response |
| Another consumer of the live service | Thin HTTP/WS transport with bounded latest-state caching | No duplicate GPU model, no unbounded queue or heavy per-frame LLM call |

Do not change model weights, vocabulary, TensorRT precision/backend, detector size, ROI/NMS logic, confidence thresholds inside the detector, pose math, calibration, retention/freshness windows or encoder behavior merely to wire the agent. In particular, do not flip `pick_candidate`, invent dimensions/site occupancy, remove epoch checks, or call raw hardware Writers as a "minor integration fix."

If a genuinely new detector, segmentation model, scene validator, site finder or motion monitor is required, describe it as separate owned work. Metadata exposure alone cannot create the evidence those capabilities require.

## 12. Handoff validation checklist

Before allowing a live action, the receiving agent/owner should demonstrate:

- Exactly one real perception pipeline reports TensorRT FP16 ready; ordinary tool reads do not create sessions or compile engines.
- CPU fallback or unavailable GPU is visible as an error/operator decision, not silently accepted.
- The 0.25/0.30 policy rejects low scores, partial views, weak/missing geometry, invalid pose, ambiguous IDs, old high-score memory and protected/confirmed/held targets.
- Consecutive confirmation counts distinct neural timestamps, not repeated polling or optical-flow frames.
- A deliberately arranged non-box cuboid-like pile is rejected/left uncertain, even if a model rectangle looks stable. An empty eligible list is acceptable.
- Epoch changes invalidate all cached targets/sites; invalid geometry remains null/unknown.
- Direct `Scan` and HTTP observations agree on the underlying evidence; rich world-model proposals are not mistaken for actionable `BoxObservation` geometry. Score/quality policy precedes action eligibility, and bounded-world-model omissions remain unknown.
- v2 provider capabilities are truthful. Images, build sites, occupancy, monitoring, possession and stopping have real implementations or preflight refuses motion.
- Preset action alignment/calibration, idempotent admission, phase/status timing and load-preserving cancellation are tested by the action owner.
- Placement evidence postdates terminal action completion. No fixture/simulated evidence authorizes real motors.
- Sensor and controller calls meet their own latency budgets under the actual workload; a smooth viewer is not the test.

Safe offline checks from the repository root include:

```bash
python3 -B bot-code/perception.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s bot-code/tests -v
python3 -B -S bot-code/main.py --mock --planner deterministic
```

Use `../AGENTS.md` for the current optional-dependency test command. Normal tests must not run hardware actions or paid model calls. Supervised physical acceptance is a separate coordinated step.

**Bottom line:** preserve the working GPU pipeline. Adapt its real data into the controller's interface, apply conservative confidence and identity policy outside the detector, and invoke only validated preset actions through the single motion owner. If evidence is missing, expose that clearly rather than changing perception or inventing state to force the workflow forward.
