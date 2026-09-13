"""Interface-v2 action provider: the designated owner of both arms and drive.ctrl.

Run it through the agent with ``--provider actions.provider:build_providers``. It must not be
started beside mc_skills or actions/pickup.py, which open the same single-writer topics.

Only operations that are actually implemented are registered, so Capabilities.operations stays
truthful and Agent.preflight fails with a clear list rather than a provider that lies about what
it can do. Today that is ``place``; look_around, approach_box, pickup and move_to_build follow.

Possession is reported as ``unknown`` unless an independent evidence source is supplied. Contact
tracking error and "the command completed" are not possession evidence, and inventing possession
alongside real motion is forbidden. Unknown is the truthful answer and the agent treats it as
blocking, which is the intended behaviour.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

from agent import CapabilityError
from agent_adapters import AgentProviders, FunctionActions, FunctionResult
from agent_types import ExecutorState, Holding

try:                        # flat, when actions/ is on the path (tests, direct runs)
    import armctl
    from geometry import Geometry
except ImportError:         # as actions.provider, which is how load_providers imports it
    from actions import armctl
    from actions.geometry import Geometry
Cancelled = armctl.Cancelled

# place: reverses where a two-arm cradle grasp leaves the box. TUNE on the robot.
LOWER_SPEED = 0.4       # turns/s on J0 descending to the layer height
UNCRADLE_SPEED = 0.05   # turns/s easing the elbows back to 90 deg, setting the front edge down
GRIP_SPEED = 0.4        # turns/s opening the jaws
RELEASE_SPEED = 0.08    # turns/s swinging J2 out of the squeeze
RELEASE_TRAVEL = 0.12   # turns of J2 outward travel that clears the box
RETREAT_SPEED = 0.6     # turns/s lifting J0 clear once the box is free
RETREAT_TURNS = 1.0     # turns of J0 lift after release
SETTLE_S = 0.5          # let each stage stop moving before the next one starts
SETTLE_AFTER_EFFECT_S = 1.0  # > perception's 0.8 s freshness, so the new state is observed
                             # before the action reports terminal

# pickup: mirrors actions/pickup.py's stage order. That script stays the prototype and the place
# these values get tuned; it cannot be imported (it installs a SIGINT handler at import time) and
# must not be edited, so the sequence is rebuilt here on armctl primitives. Re-read its docstring
# after any upstream change — the prose here is a copy, and copies drift.
INITIALIZE_SPEED = 0.08  # turns/s returning J1/J2/J4/J5/J6 to their configured home angles
J0_SPEED = 0.4           # turns/s along the lift
J0_BOTTOM_MARGIN = 0.80  # turns back toward the top from the calibrated bottom; NOT floor clearance
ELBOW_SPEED = 0.15       # turns/s bending the elbow
SPREAD_SPEED = 0.15      # turns/s swinging the whole arm out (J2 carries the arm, so ease it)
PINCH_SPEED = 0.05       # turns/s J2 creep toward the box
PINCH_CONTACT_ERR = 0.015  # turns of J2 tracking error that counts as touching the box
PINCH_SQUEEZE = 0.03     # turns commanded past contact: a steady spring squeeze. TUNE on box
HOOK_SPEED = 0.08        # turns/s wrist creep, hand turning in across the front of the box
HOOK_MAX_TRAVEL = 0.22   # turns of wrist travel when nothing stops the hand earlier
HOOK_CONTACT_ERR = 0.015
HOOK_SQUEEZE = 0.01      # turns past the hook contact point. TUNE on box
CRADLE_TILT = 0.03       # turns of extra elbow flex past 90 deg, lifting the box's front edge. TUNE
CRADLE_SPEED = 0.05      # turns/s; slow so the box rolls back onto the forearms, not out of them
LIFT_SPEED = 1.2         # turns/s for the final shoot-up
HOME_JOINTS = (1, 2, 4, 5, 6)   # returned to configured home while raised, during initialize

# Driving. The standoff is what leaves the box inside the forearm spread when pickup starts, so
# it is geometry, not preference: MEASURE it against the arms rather than trusting this default.
APPROACH_STANDOFF = 0.45    # m from base centre to box centre when the approach completes. TUNE
BUILD_STANDOFF = 0.55       # m from base centre to the target cell. TUNE
CARRY_SPEED = 0.05          # m/s while loaded; slower than an empty approach
CARRY_OMEGA = 0.10          # rad/s while loaded; a cradled box is held by squeeze, not a grip

# look_around: an in-place survey, stationed rather than continuous. Perception rejects frames
# older than its 0.8 s freshness window, so a frame grabbed mid-rotation ages out before it can be
# used; the robot has to actually stop to buy a usable observation.
SURVEY_STATIONS = 8     # yaw stops around a full turn
SURVEY_YAW_RATE = 0.4   # rad/s, well under drive.max_angular_vel (0.9)
SURVEY_SETTLE_S = 1.2   # dwell per station, comfortably past perception's 0.8 s freshness budget
SURVEY_YAW_SIGN = 1.0   # drive.ctrl twist[1] is DOCUMENTED as +CCW but has never been verified on
                        # the robot. Confirm with the smallest possible rotation before trusting it.


class Executor:
    """Shared state between the motion routines and the provider's bounded status calls.

    Holds the cancel event and the current phase. Every read here must return promptly:
    status/state are bounded at 250 ms and must never block behind a motion routine.
    """

    def __init__(self, rig, holding_source=None, log=print):
        self.rig = rig
        self.log = log
        self._holding_source = holding_source
        self._lock = threading.Lock()
        self._phase = "idle"
        self._moving = False
        self.cancel = threading.Event()

    def phase(self, name):
        with self._lock:
            self._phase = name
        self.log(f"[provider] phase {name}")

    def begin(self):
        """Arm a fresh motion. Raises if a cancel is still latched from the previous one."""
        with self._lock:
            self._moving = True
        self.cancel.clear()

    def end(self, phase):
        with self._lock:
            self._moving, self._phase = False, phase

    def read_state(self):
        with self._lock:
            moving, phase = self._moving, self._phase
        return ExecutorState(holding=self.holding(), ts=time.time(), ready=not moving,
                             motion="running" if moving else "stopped", phase=phase)

    def holding(self):
        if self._holding_source is None:
            return Holding(status="unknown", ts=time.time(), source="no-possession-sensor")
        return self._holding_source()

    def stop(self):
        """Load-preserving stop: interrupt the routine, zero the twist, keep torque on.

        The publisher thread keeps commanding the arms' current pose, so a held box stays held.
        Nothing here closes a writer or disables torque, either of which would drop the load.
        """
        self.cancel.set()
        self.rig.hold()
        self.log("[provider] stop: twist zeroed, arm targets frozen, torque untouched")


def look_around(executor, request):
    """Survey in place: rotate through a full turn in stations, dwelling at each one.

    The dwell is the point. A twist is a velocity that expires in 100 ms, so the rig's publisher
    keeps it alive while turning, but perception cannot use anything measured mid-rotation â€” every
    usable frame comes from a station. The base is zeroed on every exit path, including cancel;
    the daemon's command timeout is the backstop, not the brake.

    Requires empty grippers. The agent enforces that, and this does not grasp or carry.
    """
    rig, cancel = executor.rig, executor.cancel
    stations = int(request.get("stations", SURVEY_STATIONS))
    step_rad = 2.0 * np.pi / stations
    turn_s = step_rad / SURVEY_YAW_RATE
    executor.phase("surveying")
    try:
        for i in range(stations):
            rig.set_twist(0.0, SURVEY_YAW_SIGN * SURVEY_YAW_RATE)
            armctl.dwell(turn_s, cancel)
            rig.stop_base()
            executor.log(f"[provider] survey station {i + 1}/{stations}")
            armctl.dwell(SURVEY_SETTLE_S, cancel)
    finally:
        rig.stop_base()
    executor.phase("settling")
    armctl.dwell(SURVEY_SETTLE_S, cancel)
    return "completed"


def approach_box(executor, request):
    """Drive until the box sits at the grasp standoff, steering to where it is seen now.

    Empty grippers only. The drive re-observes every cycle, so a box that stops being visible
    ends the approach rather than letting the robot continue toward a remembered position.
    """
    executor.phase("approaching")
    armctl.drive_to_standoff(executor.rig, request["target_fn"], APPROACH_STANDOFF,
                             cancel=executor.cancel, log=executor.log)
    executor.phase("settling")
    armctl.dwell(SURVEY_SETTLE_S, executor.cancel)   # buy perception a usable post-motion frame
    return "completed"


def move_to_build(executor, request):
    """Carry the held box to the build site.

    Same drive loop as the approach, but loaded: slower, gentler, and the arms hold the cradle
    they were left in. Nothing here touches the arms, because anything that moves them mid-carry
    risks the squeeze that is the only thing holding the box.
    """
    executor.phase("carrying")
    armctl.drive_to_standoff(executor.rig, request["target_fn"], BUILD_STANDOFF,
                             cancel=executor.cancel, log=executor.log,
                             speed=CARRY_SPEED, omega=CARRY_OMEGA)
    executor.phase("settling")
    armctl.dwell(SURVEY_SETTLE_S, executor.cancel)
    return "completed"


def pickup(executor, request):
    """Cage a box between both forearms and lift it.

    Mirrors actions/pickup.py: initialize to a calibrated reference pose, spread the arms wider
    than the box, turn the wrists in while still raised, lower, squeeze to contact, grip, cradle,
    and lift. Contact is a rise in joint tracking error — which says something resisted the joint,
    not that a box is held. Possession comes from the executor's evidence source, never from here.
    """
    arms, cancel = executor.rig.arms, executor.cancel
    log = executor.log

    executor.phase("grasping")
    # 1. initialize: the same calibrated reference pose every run, while raised and unloaded.
    armctl.ramp_joint(arms, armctl.ELBOW, [a.elbow_90 for a in arms], ELBOW_SPEED,
                      ease=armctl.smootherstep, cancel=cancel)
    armctl.settle_joint(arms, armctl.ELBOW, cancel=cancel, log=log)
    armctl.ramp_joint(arms, armctl.GRIPPER, [a.grip_open for a in arms], GRIP_SPEED, cancel=cancel)
    armctl.ramp_joint(arms, armctl.J0, [a.top for a in arms], J0_SPEED, cancel=cancel)
    armctl.settle_joint(arms, armctl.J0, cancel=cancel, log=log)
    for joint in HOME_JOINTS:
        armctl.ramp_joint(arms, joint, [float(a.cfg.home[joint]) for a in arms],
                          INITIALIZE_SPEED, ease=armctl.smootherstep, cancel=cancel)
        armctl.settle_joint(arms, joint, cancel=cancel, log=log)
    armctl.dwell(SETTLE_S, cancel)

    # 2. spread wider than the box, out to each arm's own calibrated edge.
    out = [float(a.edge(armctl.SWING, armctl.spread_direction(a)[armctl.SWING])) for a in arms]
    armctl.ramp_joint(arms, armctl.SWING, out, SPREAD_SPEED, ease=armctl.smootherstep, cancel=cancel)
    armctl.settle_joint(arms, armctl.SWING, cancel=cancel, log=log)
    armctl.dwell(SETTLE_S, cancel)

    # 3. turn the wrists in while still at the top: yaw alone can change hand height, so this
    #    must not happen down at box level.
    armctl.creep_to_contact(arms, armctl.hook_direction, HOOK_SPEED, HOOK_CONTACT_ERR,
                            HOOK_SQUEEZE, HOOK_MAX_TRAVEL, "hook", cancel=cancel, log=log)
    armctl.dwell(SETTLE_S, cancel)

    # 4. down to the calibrated bottom, held back by a lift offset that does not sense the floor.
    armctl.ramp_joint(arms, armctl.J0, [armctl.j0_low_target(a, J0_BOTTOM_MARGIN) for a in arms],
                      J0_SPEED, cancel=cancel)
    armctl.settle_joint(arms, armctl.J0, cancel=cancel, log=log)
    armctl.dwell(SETTLE_S, cancel)

    # 5. squeeze until each forearm independently meets its side of the box, so an off-centre
    #    box is still held from both sides.
    armctl.creep_to_contact(arms, armctl.pinch_direction, PINCH_SPEED, PINCH_CONTACT_ERR,
                            PINCH_SQUEEZE, float("inf"), "pinch", cancel=cancel, log=log)
    armctl.dwell(SETTLE_S, cancel)

    armctl.ramp_joint(arms, armctl.GRIPPER, [a.grip_closed for a in arms], GRIP_SPEED, cancel=cancel)
    armctl.dwell(SETTLE_S, cancel)

    # 6. cradle: the elbows flex past 90 deg so the box tilts back and its weight rests on the
    #    forearms rather than on the squeeze alone.
    armctl.ramp_joint(arms, armctl.ELBOW, [a.cradle(CRADLE_TILT) for a in arms], CRADLE_SPEED,
                      ease=armctl.smootherstep, cancel=cancel)
    armctl.settle_joint(arms, armctl.ELBOW, cancel=cancel, log=log)
    armctl.dwell(SETTLE_S, cancel)

    executor.phase("lifting")
    armctl.ramp_joint(arms, armctl.J0, [a.top for a in arms], LIFT_SPEED, cancel=cancel)
    armctl.settle_joint(arms, armctl.J0, cancel=cancel, log=log)
    # Let perception actually see the grasp before reporting terminal. Possession is measured from
    # the latest scan, so returning the instant the lift finishes reports possession from a frame
    # taken before the box was held, and that stale answer contradicts the next one.
    armctl.dwell(SETTLE_AFTER_EFFECT_S, cancel)
    return "lifted"


def place(executor, request):
    """Lower the cradled box onto its cell, release it, and retreat.

    The inverse of a two-arm cradle grasp: J0 down to the layer height, elbows back to 90 deg so
    the box's front edge sets down first, jaws open, J2 out of the squeeze, J0 clear. Torque stays
    on throughout and the publisher keeps commanding, so cancelling mid-place leaves the box held
    rather than dropped.

    ``target_z_drop`` is how far the box has to descend, in metres, resolved by the caller from
    the site frame against a fresh pose.
    """
    rig, arms, cancel = executor.rig, executor.rig.arms, executor.cancel
    drop_turns = arms[0].j0_meters_to_turns(request["z_drop"])

    executor.phase("lowering")
    targets = [float(np.clip(a.cmd[armctl.J0] - drop_turns * np.sign(a.top - a.bottom),
                             min(a.lo[armctl.J0], a.hi[armctl.J0]) + armctl.RANGE_MARGIN,
                             max(a.lo[armctl.J0], a.hi[armctl.J0]) - armctl.RANGE_MARGIN))
               for a in arms]
    armctl.ramp_joint(arms, armctl.J0, targets, LOWER_SPEED, cancel=cancel)
    armctl.settle_joint(arms, armctl.J0, cancel=cancel, log=executor.log)
    armctl.dwell(SETTLE_S, cancel)

    executor.phase("releasing")
    # Un-cradle first: the elbows carry the box's weight, so easing them back to 90 deg sets the
    # front edge down while the forearms still have it. Quintic, because a jerk here tips the box.
    armctl.ramp_joint(arms, armctl.ELBOW, [a.elbow_90 for a in arms], UNCRADLE_SPEED,
                      ease=armctl.smootherstep, cancel=cancel)
    armctl.settle_joint(arms, armctl.ELBOW, cancel=cancel, log=executor.log)
    armctl.dwell(SETTLE_S, cancel)

    armctl.ramp_joint(arms, armctl.GRIPPER, [a.grip_open for a in arms], GRIP_SPEED, cancel=cancel)
    armctl.dwell(SETTLE_S, cancel)

    # Both arms swing out together; releasing one side first would shove the box off its cell.
    out = [float(a.cmd[armctl.SWING] + armctl.spread_direction(a)[armctl.SWING] * RELEASE_TRAVEL)
           for a in arms]
    out = [float(np.clip(t, a.lo[armctl.SWING] + armctl.RANGE_MARGIN,
                         a.hi[armctl.SWING] - armctl.RANGE_MARGIN)) for a, t in zip(arms, out)]
    armctl.ramp_joint(arms, armctl.SWING, out, RELEASE_SPEED, ease=armctl.smootherstep, cancel=cancel)
    armctl.settle_joint(arms, armctl.SWING, cancel=cancel, log=executor.log)
    armctl.dwell(SETTLE_S, cancel)

    executor.phase("retreating")
    lift = [float(a.cmd[armctl.J0] + RETREAT_TURNS * np.sign(a.top - a.bottom)) for a in arms]
    lift = [float(np.clip(t, min(a.lo[armctl.J0], a.hi[armctl.J0]) + armctl.RANGE_MARGIN,
                          max(a.lo[armctl.J0], a.hi[armctl.J0]) - armctl.RANGE_MARGIN))
            for a, t in zip(arms, lift)]
    armctl.ramp_joint(arms, armctl.J0, lift, RETREAT_SPEED, cancel=cancel)
    armctl.settle_joint(arms, armctl.J0, cancel=cancel, log=executor.log)
    armctl.dwell(SETTLE_AFTER_EFFECT_S, cancel)   # let perception see the release before reporting
    return "retreated"


def _wrap(executor, routine, resolve):
    """Adapt a motion routine to the FunctionActions callable contract.

    ``resolve`` turns the ActionRequest's world-frame geometry into the routine's base-frame
    arguments against a fresh pose, and raises if the geometry is stale or unreachable. It runs
    before any effect, never at admission time.
    """
    def call(request):
        executor.begin()
        try:
            args = resolve(request)
        except Exception as exc:
            executor.end("rejected")
            return FunctionResult(False, "rejected", type(exc).__name__, effects_started="no")
        try:
            phase = routine(executor, args)
        except Cancelled:
            executor.end("cancelled")
            return FunctionResult(False, "cancelled", "blocked_motion", effects_started="unknown")
        except armctl.Stuck:
            # A known failure with a known end state: the drive gave up and zeroed the base in its
            # own finally. Reporting it as blocked rather than letting it escape as an unknown
            # outcome is what lets the agent retry instead of stopping for an operator.
            executor.end("blocked")
            return FunctionResult(False, "blocked", "blocked_motion", effects_started="unknown")
        executor.end(phase)
        return FunctionResult(True, phase, effects_started="yes")
    return call


def build_providers(rig=None, observations=None, geometry=None, holding_source=None,
                    voxel_size=None, log=print):
    """Factory for ``--provider actions.provider:build_providers``.

    ``load_providers`` calls this with no arguments, so a bare call has to stand up the whole live
    stack itself: the rig, perception in this same process, the geometry seam and possession. The
    arguments exist for tests and for callers that already hold those pieces.

    The box edge has to be stated, never guessed: eligibility compares a detected box's size
    against it within 1%, so a wrong value silently makes every box ineligible. It must match the
    ``--box-size`` the agent is run with.
    """
    if voxel_size is None:
        edge = os.environ.get("CRAFTER_BOX_SIZE")
        if edge is None:
            raise CapabilityError(
                "set CRAFTER_BOX_SIZE to the box edge in metres, matching --box-size; "
                "it is a measurement, not a default")
        voxel_size = (float(edge),) * 3
    if observations is None:
        from observations import RobotObservations
        observations = RobotObservations(
            voxel_size=voxel_size,
            mock=os.environ.get("CRAFTER_PERCEPTION_MOCK") == "1")
    rig = rig or armctl.Rig(log=log)
    rig.start()
    if geometry is None and observations is not None:
        geometry = Geometry(observations, carry_height=rig.carry_height)
    if holding_source is None and observations is not None and voxel_size is not None:
        try:                    # perception supplies possession; without it, unknown stands
            from observations import CarryVolume
        except ImportError:
            CarryVolume = None
        if CarryVolume is not None:
            holding_source = CarryVolume(observations, rig, voxel_size)
    executor = Executor(rig, holding_source=holding_source, log=log)

    # look_around needs no target geometry: it surveys where it stands.
    functions = {"look_around": _wrap(executor, look_around, lambda r: {}),
                 "pickup": _wrap(executor, pickup, lambda r: {})}
    if geometry is not None:
        functions["place"] = _wrap(executor, place, geometry.place)
        functions["approach_box"] = _wrap(executor, approach_box, geometry.approach_box)
        functions["move_to_build"] = _wrap(executor, move_to_build, geometry.move_to_build)

    actions = FunctionActions(
        functions, read_state=executor.read_state, stop=executor.stop,
        max_box_size=(0.25, 0.25, 0.25), max_height=1.0, action_timeout=60.0)

    def close():
        """Transports and sensors only. Never parks, releases, or torques off a loaded robot."""
        executor.stop()
        shut = getattr(observations, "close", None)
        if callable(shut):
            shut()

    return AgentProviders(actions=actions, observations=observations, close=close)
