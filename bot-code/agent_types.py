"""Robot-free component interface v2. See README.md's agent interface reference.

Cells and dimensions use schematic (x, layer, z) order. Physical vectors use
right-handed world XYZ in meters, with +Z up; angles are radians. Evidence uses
Unix seconds on a shared clock. Deadlines inside the agent use a monotonic clock.
"""

from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol

INTERFACE_VERSION = 2
MAX_IMAGE_BYTES = 512 * 1024
MAX_SCENE_IMAGES = 3
MAX_WORLD_MODEL_BYTES = 32 * 1024
Cell = tuple[int, int, int]
Vector = tuple[float, float, float]
Operation = Literal["observe", "look_around", "select_site", "approach_box", "pickup",
                    "move_to_build", "place", "done", "stop"]
MOTION_OPS = frozenset({"look_around", "approach_box", "pickup", "move_to_build", "place"})


@dataclass(frozen=True)
class AgentConfig:
    voxel_size: Vector = (0.3, 0.3, 0.3)
    max_blocks: int = 128
    max_extent: int = 32
    max_steps: int = 512
    max_actions: int = 512
    max_no_progress: int = 16
    max_searches: int = 8
    max_retries: int = 2
    observation_attempts: int = 3
    observation_timeout: float = 1.0
    fresh_s: float = 2.0
    poll_s: float = 0.01
    history_limit: int = 20

    def __post_init__(self):
        if not vector_valid(self.voxel_size) or min(self.voxel_size) <= 0:
            raise ValueError("voxel_size must contain three finite positive dimensions")
        for name in ("max_blocks", "max_extent", "max_steps", "max_actions",
                     "max_no_progress", "max_searches", "observation_attempts", "history_limit"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if not all(math.isfinite(v) and v > 0 for v in (self.fresh_s, self.poll_s, self.observation_timeout)):
            raise ValueError("freshness, observation and polling intervals must be finite and positive")


def vector_valid(value):
    return (isinstance(value, (tuple, list)) and len(value) == 3
            and all(type(v) in (float, int) and math.isfinite(v) for v in value))


def cell_valid(value):
    return isinstance(value, (tuple, list)) and len(value) == 3 and all(type(v) is int for v in value)


@dataclass(frozen=True)
class BuildRequirements:
    """Normalized target occupancy and physical voxel dimensions, not motor goals."""

    cells: tuple[Cell, ...]
    voxel_size: Vector
    extents: Cell

    @property
    def dimensions(self) -> Vector:
        return tuple(n * size for n, size in zip(self.extents, self.voxel_size))


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    requirements: BuildRequirements
    original_cells: tuple[Cell, ...]


@dataclass(frozen=True)
class Holding:
    """Independent measured possession. Missing markers and motor success are not proof."""

    status: Literal["empty", "holding", "unknown"] = "unknown"
    box_id: int | None = None
    ts: float = 0.0
    source: str = "unavailable"


@dataclass(frozen=True)
class BoxObservation:
    """Stable identity within a map epoch; remembered boxes must have current=False."""

    id: int
    position: Vector
    size: Vector | None
    last_seen: float
    current: bool = True
    eligible: bool | None = None
    valid: bool = True


@dataclass(frozen=True)
class BuildSite:
    """Measured frame: origin is cell (0,0,0)'s bottom center; col/row are unit axes.

    Feasibility covers the entire requested build, including approach and loaded
    navigation, not merely an empty rectangle. Dimensions are (width, height, depth).
    """

    id: str
    origin: Vector
    col: Vector
    row: Vector
    dimensions: Vector
    ts: float
    epoch: int
    frame_id: str = "world"
    valid: bool = False
    floor_valid: bool = False
    clearance_valid: bool = False
    feasible: bool = False
    cost: float = 0.0

    def cell_center(self, cell: Cell, voxel_size: Vector) -> Vector:
        up = (self.col[1]*self.row[2]-self.col[2]*self.row[1],
              self.col[2]*self.row[0]-self.col[0]*self.row[2],
              self.col[0]*self.row[1]-self.col[1]*self.row[0])
        return tuple(self.origin[i] + self.col[i]*cell[0]*voxel_size[0]
                     + self.row[i]*cell[2]*voxel_size[2]
                     + up[i]*(cell[1]+.5)*voxel_size[1] for i in range(3))


@dataclass(frozen=True)
class CellObservation:
    """ts is this cell's measurement time, never the containing snapshot's refresh time."""

    cell: Cell
    status: Literal["empty", "occupied", "unknown"]
    ts: float
    box_id: int | None = None


@dataclass(frozen=True)
class SceneImage:
    """Bounded inline PNG/JPEG; no paths or remote URLs. Pixels are not motion authority."""

    view: str
    data_url: str = field(repr=False)
    captured_at: float
    epoch: int
    frame_id: str = "world"
    simulated: bool = False
    description: str = ""

    def __post_init__(self):
        if not isinstance(self.view, str) or not 0 < len(self.view) <= 64:
            raise ValueError("image view must be a bounded name")
        if not isinstance(self.description, str) or len(self.description) > 512:
            raise ValueError("image description must be bounded")
        if (type(self.captured_at) not in (int, float) or not math.isfinite(self.captured_at)
                or type(self.epoch) is not int or not isinstance(self.frame_id, str) or not self.frame_id
                or type(self.simulated) is not bool):
            raise ValueError("image requires a timestamp, frame, epoch and simulation flag")
        if not isinstance(self.data_url, str) or len(self.data_url) > 4*((MAX_IMAGE_BYTES+2)//3)+32:
            raise ValueError("image exceeds the inline byte budget")
        header, separator, encoded = self.data_url.partition(",")
        if not separator or header not in {"data:image/png;base64", "data:image/jpeg;base64"}:
            raise ValueError("images must be inline base64 PNG or JPEG, not remote URLs")
        data = base64.b64decode(encoded, validate=True)
        signature = b"\x89PNG\r\n\x1a\n" if header == "data:image/png;base64" else b"\xff\xd8\xff"
        if len(data) > MAX_IMAGE_BYTES or not data.startswith(signature):
            raise ValueError("invalid or oversized encoded image")


@dataclass(frozen=True)
class ObservationSnapshot:
    """Immutable sensor snapshot. Refreshing metadata must not refresh cached evidence.

    Occupancy includes every cell in the requested bounding envelope and any extra
    obstruction. Occluded/unobserved cells remain unknown. Images, boxes and holding
    retain their own capture times. All geometry shares frame_id and epoch.
    """

    revision: str
    captured_at: float
    received_at: float
    epoch: int
    frame_id: str = "world"
    valid: bool = False
    pose_valid: bool = False
    boxes: tuple[BoxObservation, ...] = ()
    holding: Holding | None = None
    site_id: str | None = None
    occupancy: tuple[CellObservation, ...] = ()
    occupancy_complete: bool = False
    search_exhausted: bool = False
    warnings: tuple[str, ...] = ()
    base_position: Vector | None = None
    base_yaw: float | None = None
    images: tuple[SceneImage, ...] = ()
    world_model_json: str = field(default="{}", repr=False)

    def __post_init__(self):
        if (not isinstance(self.world_model_json, str)
                or len(self.world_model_json.encode("utf-8")) > MAX_WORLD_MODEL_BYTES):
            raise ValueError("world model exceeds its bounded JSON budget")
        world = self.world_model
        if not isinstance(world, dict):
            raise ValueError("world model must be a JSON object")
        json.dumps(world, allow_nan=False)

    @property
    def world_model(self):
        return json.loads(self.world_model_json)


@dataclass(frozen=True)
class Capabilities:
    operations: frozenset[str] = frozenset()
    api_version: int = INTERFACE_VERSION
    status: bool = False
    cancellation: bool = False
    idempotency: bool = False
    possession: bool = False
    carrying: bool = False
    max_height: float = 0.0
    max_box_size: Vector = (0.0, 0.0, 0.0)
    action_timeout: float = 30.0
    emergency_stop: bool = False


@dataclass(frozen=True)
class PerceptionCapabilities:
    inventory: bool = False
    sites: bool = False
    occupancy: bool = False
    possession: bool = False
    api_version: int = INTERFACE_VERSION
    monitoring: bool = False
    images: bool = False


@dataclass(frozen=True)
class ExecutorState:
    """Fresh independent device state; ready only when a new motion may be admitted."""

    holding: Holding
    ts: float
    ready: bool = False
    motion: Literal["stopped", "running", "unknown"] = "unknown"
    active_action: str | None = None
    phase: str = "unknown"


@dataclass(frozen=True)
class Step:
    """Model-selected intent. Only the controller may create an ActionRequest."""

    operation: Operation
    reason: str = ""
    box_id: int | None = None
    site_id: str | None = None
    cell: Cell | None = None
    search: Literal["materials", "sites"] | None = None

    def key(self):
        return self.operation, self.box_id, self.site_id, self.cell, self.search


@dataclass(frozen=True)
class ActionRequest:
    """Idempotent admission envelope with validated geometry, not an LLM motor command.

    Agent-generated requests include requirements and, when applicable, site/box.
    expires_at is the latest admission time, not an execution deadline. The action
    owner must recheck localization and physical preconditions before moving.
    """

    request_id: str
    job_id: str
    step: Step
    observation_ref: str
    epoch: int
    frame_id: str
    submitted_at: float
    requirements: BuildRequirements | None = None
    site: BuildSite | None = None
    box: BoxObservation | None = None
    expires_at: float = 0.0

    def validate_admission(self, now: float):
        """Shared envelope checks; the action owner still validates live physical state."""
        if self.step.operation not in MOTION_OPS:
            raise ValueError("only semantic motion operations may be submitted")
        if any(not isinstance(v, str) or not 0 < len(v) <= 128
               for v in (self.request_id, self.job_id, self.observation_ref, self.frame_id)):
            raise ValueError("action identities must be bounded nonempty strings")
        if (type(self.epoch) is not int or not isinstance(self.requirements, BuildRequirements)
                or not math.isfinite(self.submitted_at) or not math.isfinite(self.expires_at)
                or self.submitted_at > now+.05 or self.expires_at <= now):
            raise ValueError("action admission is expired or missing job/frame context")
        if self.step.box_id is not None and (self.box is None or self.box.id != self.step.box_id):
            raise ValueError("action box geometry does not match the selected identity")
        if self.step.site_id is not None and (self.site is None or self.site.id != self.step.site_id):
            raise ValueError("action build geometry does not match the selected site")
        if self.site is not None and (self.site.epoch, self.site.frame_id) != (self.epoch, self.frame_id):
            raise ValueError("action site uses a different frame or epoch")
        if self.step.cell is not None and self.step.cell not in self.requirements.cells:
            raise ValueError("action cell is not in the admitted job")


@dataclass(frozen=True)
class ActionReceipt:
    request_id: str
    action_id: str


@dataclass(frozen=True)
class ActionOutcome:
    """ts is the last transition/completion time; observed_at is the status heartbeat.

    Polling must never move a terminal ts forward. A succeeded motion is not proof
    of possession or placement. Effects/phase must describe actual execution.
    """

    request_id: str
    action_id: str
    status: Literal["running", "succeeded", "failed", "cancelled", "unknown"]
    ts: float
    phase: str = "unknown"
    effects_started: Literal["yes", "no", "unknown"] = "unknown"
    motion: Literal["stopped", "running", "unknown"] = "unknown"
    error_code: str | None = None
    holding: Holding | None = None
    observed_at: float = 0.0


@dataclass(frozen=True)
class MotionObservation:
    """Read-only, phase-aware monitoring of one action. None means safety is unknown.

    safe=True requires fresh evidence for the current operation/phase. Expected
    grasp occlusion is not automatically a lost box; unexplained loss of a carried
    load is unsafe. The controller requests a stop on false, unknown or stale data.
    """

    request_id: str
    action_id: str
    phase: str
    snapshot: ObservationSnapshot
    safe: bool | None
    reason: str = ""


@dataclass(frozen=True)
class CancellationReceipt:
    """Acknowledgement is not proof that motion stopped or a held load is safe."""

    action_id: str | None
    acknowledged: bool
    stopped: bool = False


class ActionProvider(Protocol):
    """Single motion owner. Calls are bounded; submit returns before motion completes.

    Keep request IDs queryable throughout the job, reject changed payloads for an
    existing ID, and never replay on a transport timeout. Stop/cancel must preserve
    loads and prevent the interrupted routine from issuing later motor commands.
    """

    def capabilities(self) -> Capabilities: ...
    def submit(self, request: ActionRequest) -> ActionReceipt: ...
    def lookup(self, request_id: str) -> ActionReceipt | None: ...
    def status(self, action_id: str) -> ActionOutcome: ...
    def cancel(self, action_id: str) -> CancellationReceipt: ...
    def stop(self) -> CancellationReceipt: ...
    def state(self) -> ExecutorState: ...


class ObservationProvider(Protocol):
    """Read-only sensing owner. Maintain sensors in the background; never move hardware.

    observe returns the latest snapshot, not fabricated fresh measurements. monitor
    is called while actions run and must use the supplied action phase. Site checks
    cover the complete requirements. Unknown geometry must not become free space.
    """

    def capabilities(self) -> PerceptionCapabilities: ...
    def observe(self, site_id: str | None = None) -> ObservationSnapshot: ...
    def monitor(self, request: ActionRequest, outcome: ActionOutcome) -> MotionObservation: ...
    def find_build_sites(self, requirements: BuildRequirements) -> tuple[BuildSite, ...]: ...
    def check_build_site(self, site_id: str, requirements: BuildRequirements) -> BuildSite | None: ...


class Reasoner(Protocol):
    """Receives bounded state, SceneImage dictionaries, and legal choices; has no hardware access."""

    def decide(self, context: dict, choices: tuple[Step, ...]) -> Step: ...


@dataclass
class BuildState:
    job: JobSpec
    phase: str = "INVENTORY"
    inventory: dict[int, BoxObservation] = field(default_factory=dict)
    confirmed: dict[Cell, int] = field(default_factory=dict)
    candidate_sites: tuple[BuildSite, ...] = ()
    site: BuildSite | None = None
    target_box: int | None = None
    target_cell: Cell | None = None
    stage: str = "approach_box"
    holding: Holding = field(default_factory=Holding)
    snapshot: ObservationSnapshot | None = None
    executor: ExecutorState | None = None
    active_request: ActionRequest | None = None
    receipt: ActionReceipt | None = None
    last_outcome: ActionOutcome | None = None
    last_monitor: MotionObservation | None = None
    quarantined: set[int] = field(default_factory=set)
    retries: dict[tuple, int] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    searches: int = 0
    actions: int = 0
    steps: int = 0
    no_progress: int = 0
    reason: str = ""


@dataclass(frozen=True)
class RunResult:
    job_id: str
    status: str
    reason: str
    placed: int
    required: int
    actions: int
    steps: int
    events: tuple[dict, ...]

    @property
    def success(self):
        return self.status == "COMPLETED"


EventSink = Callable[[dict], None]
