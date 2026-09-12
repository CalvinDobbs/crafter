from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol

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
        if not all(math.isfinite(v) and v > 0 for v in (self.fresh_s, self.poll_s)):
            raise ValueError("freshness and polling intervals must be finite and positive")


def vector_valid(value):
    return (isinstance(value, (tuple, list)) and len(value) == 3
            and all(type(v) in (float, int) and math.isfinite(v) for v in value))


def cell_valid(value):
    return isinstance(value, (tuple, list)) and len(value) == 3 and all(type(v) is int for v in value)


@dataclass(frozen=True)
class BuildRequirements:
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
    status: Literal["empty", "holding", "unknown"] = "unknown"
    box_id: int | None = None
    ts: float = 0.0
    source: str = "unavailable"


@dataclass(frozen=True)
class BoxObservation:
    id: int
    position: Vector
    size: Vector | None
    last_seen: float
    current: bool = True
    eligible: bool | None = None
    valid: bool = True


@dataclass(frozen=True)
class BuildSite:
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


@dataclass(frozen=True)
class CellObservation:
    cell: Cell
    status: Literal["empty", "occupied", "unknown"]
    ts: float
    box_id: int | None = None


@dataclass(frozen=True)
class ObservationSnapshot:
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


@dataclass(frozen=True)
class Capabilities:
    operations: frozenset[str] = frozenset()
    api_version: int = 1
    status: bool = False
    cancellation: bool = False
    idempotency: bool = False
    possession: bool = False
    carrying: bool = False
    max_height: float = 0.0
    max_box_size: Vector = (0.0, 0.0, 0.0)
    action_timeout: float = 30.0


@dataclass(frozen=True)
class PerceptionCapabilities:
    inventory: bool = False
    sites: bool = False
    occupancy: bool = False
    possession: bool = False


@dataclass(frozen=True)
class ExecutorState:
    holding: Holding
    ts: float
    ready: bool = False
    motion: Literal["stopped", "running", "unknown"] = "unknown"
    active_action: str | None = None


@dataclass(frozen=True)
class Step:
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
    request_id: str
    job_id: str
    step: Step
    observation_ref: str
    epoch: int
    frame_id: str
    submitted_at: float


@dataclass(frozen=True)
class ActionReceipt:
    request_id: str
    action_id: str


@dataclass(frozen=True)
class ActionOutcome:
    request_id: str
    action_id: str
    status: Literal["running", "succeeded", "failed", "cancelled", "unknown"]
    ts: float
    phase: str = "unknown"
    effects_started: Literal["yes", "no", "unknown"] = "unknown"
    motion: Literal["stopped", "running", "unknown"] = "unknown"
    error_code: str | None = None
    holding: Holding | None = None


@dataclass(frozen=True)
class CancellationReceipt:
    action_id: str
    acknowledged: bool
    stopped: bool = False


class ActionProvider(Protocol):
    def capabilities(self) -> Capabilities: ...
    def submit(self, request: ActionRequest) -> ActionReceipt: ...
    def status(self, action_id: str) -> ActionOutcome: ...
    def cancel(self, action_id: str) -> CancellationReceipt: ...
    def state(self) -> ExecutorState: ...


class ObservationProvider(Protocol):
    def capabilities(self) -> PerceptionCapabilities: ...
    def observe(self, site_id: str | None = None) -> ObservationSnapshot: ...
    def find_build_sites(self, requirements: BuildRequirements) -> tuple[BuildSite, ...]: ...
    def check_build_site(self, site_id: str, requirements: BuildRequirements) -> BuildSite | None: ...


class Reasoner(Protocol):
    def decide(self, context: dict, choices: tuple[Step, ...]) -> Step: ...


@dataclass
class BuildState:
    job: JobSpec
    phase: str = "INVENTORY"
    inventory: dict[int, BoxObservation] = field(default_factory=dict)
    confirmed: dict[Cell, int] = field(default_factory=dict)
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
