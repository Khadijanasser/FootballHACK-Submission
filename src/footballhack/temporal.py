"""Deterministic possession continuity from heuristic evidence, not probabilities.

No file I/O, fitting, match-specific constants, interpolation, or global state.
See docs/temporal.md for confirmation, restart and retrospective-placement rules.
"""
from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .control import ControlEvidence
from .dead_ball import DeadBallEvidence


@dataclass(frozen=True)
class TemporalConfig:
    """Untuned engineering assumptions; all time settings are in seconds.

    A team score of .30 is reachable at 1 m with known matching motion (.375).
    A lower hold threshold preserves established ownership with weaker evidence.
    Raw conflict thresholds veto mutually strong evidence before net thresholds.
    Offline mode changes placement only, never confirmation decisions.
    """
    mode: str = "causal"
    team_threshold: float = 0.30
    team_hold_threshold: float = 0.15
    control_conflict_threshold: float = 0.50
    live_threshold: float = 0.40
    dead_threshold: float = 0.40
    phase_conflict_threshold: float = 0.40
    restart_live_threshold: float = 0.60
    tag_threshold: float = 0.50
    acquisition_s: float = 0.12
    turnover_s: float = 0.20
    phase_confirmation_s: float = 0.20
    team_hold_s: float = 2.0
    dead_hold_s: float = 2.0
    gap_reset_s: float = 2.0

    def __post_init__(self):
        if self.mode not in ("causal", "offline"):
            raise ValueError("mode must be 'causal' or 'offline'")
        for field in fields(self):
            if field.name == "mode":
                continue
            value = getattr(self, field.name)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{field.name} must be positive and finite")
            if not field.name.endswith("_s") and value > 1:
                raise ValueError(f"{field.name} must be in (0, 1]")
        if self.team_hold_threshold > self.team_threshold:
            raise ValueError("team_hold_threshold must not exceed team_threshold")
        if self.restart_live_threshold < self.live_threshold:
            raise ValueError("restart_live_threshold must not be below live_threshold")


@dataclass(frozen=True)
class PossessionResult:
    """One row per supplied frame; diagnostics describe the chosen placement.

    labels: H/A/D/X. phase: LIVE/DEAD/UNKNOWN. possession: H/A/NONE.
    support_age_s: age of the emitted team or D support; NaN for X.
    control_status: H/A/weak/conflict/missing at the acquisition threshold.
    phase_status: LIVE/DEAD/weak/conflict/missing before temporal reasoning.
    confirmation_frame: frame that confirmed an assignment, including future
        frames in offline mode; -1 on ordinary holds, expiry and reset rows.
    reset: explicit period/reset boundaries, including the first row. Long gaps
        reset continuity but are not period boundaries for downstream scoring.
    """
    frame: NDArray
    labels: NDArray
    reason: NDArray
    phase: NDArray
    possession: NDArray
    support_age_s: NDArray
    control_status: NDArray
    phase_status: NDArray
    confirmation_frame: NDArray
    reset: NDArray


def _frames(value, name):
    array = np.asarray(value)
    if (array.ndim != 1 or array.dtype.kind not in "iu"
            or np.any(array < 0) or np.any(array[1:] <= array[:-1])):
        raise ValueError(f"{name} must be strictly increasing nonnegative integer frames")
    # confirmation_frame uses signed int64 and -1 as its absence sentinel.
    if np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f"{name} exceeds the supported int64 range")
    return array


def _validate(control, dead_ball):
    if not isinstance(control, ControlEvidence) or not isinstance(dead_ball, DeadBallEvidence):
        raise TypeError("expected ControlEvidence and DeadBallEvidence")
    c = {f.name: np.asarray(getattr(control, f.name)) for f in fields(control)}
    d = {f.name: np.asarray(getattr(dead_ball, f.name)) for f in fields(dead_ball)}
    frames = _frames(c["frame"], "control.frame")
    _frames(d["frame"], "dead_ball.frame")
    if not np.array_equal(frames, d["frame"]):
        raise ValueError("evidence frame arrays must match exactly")
    n = len(frames)

    def check(values, name, shape, kind):
        a = values[name]
        if a.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if kind == "mask":
            valid = a.dtype.kind == "b"
        elif kind == "count":
            valid = a.dtype.kind in "iu" and np.all(a >= 0)
        else:
            valid = a.dtype.kind in "iuf" and not np.any(np.isinf(a))
            if valid and kind in ("weight", "nullable_weight"):
                finite = np.isfinite(a)
                valid = (kind == "nullable_weight" or finite.all()) and np.all(
                    (a[finite] >= 0) & (a[finite] <= 1))
            if valid and kind == "scores":
                valid = (np.isfinite(a).all() and np.all((a >= 0) & (a <= 1))
                         and np.allclose(a.sum(axis=1), 1, rtol=0, atol=1e-8))
        if not valid:
            raise ValueError(f"invalid {name} {kind}")

    for values in (c, d):
        check(values, "scores", (n, 3), "scores")
        check(values, "ball_available", (n,), "mask")
    check(c, "team_support", (n, 2), "nullable_weight")
    check(c, "candidate_count", (n, 2), "count")
    check(c, "team_available", (n, 2), "mask")
    observed = c["ball_available"][:, None] & c["team_available"]
    if not np.array_equal(np.isfinite(c["team_support"]), observed):
        raise ValueError("team_support must be finite exactly where ball/team are available")
    check(d, "support", (n, 2), "weight")
    for name in ("motion_available", "tag_available"):
        check(d, name, (n,), "mask")
    for name in ("set_piece_prior", "live_phase_prior", "restart_prior"):
        check(d, name, (n,), "weight")
    check(d, "boundary_clearance_m", (n,), "nullable")
    if (not np.array_equal(c["ball_available"], d["ball_available"])
            or np.any(d["motion_available"] & ~d["ball_available"])
            or not np.array_equal(np.isfinite(d["boundary_clearance_m"]), d["ball_available"])):
        raise ValueError("inconsistent ball availability diagnostics")
    return ControlEvidence(**c), DeadBallEvidence(**d)


def decode_possession(
    control: ControlEvidence,
    dead_ball: DeadBallEvidence,
    *,
    fps: float = 25.0,
    config: TemporalConfig | None = None,
    period_ids: ArrayLike | None = None,
    reset_mask: ArrayLike | None = None,
) -> PossessionResult:
    """Decode aligned evidence without filling frames or mutating inputs.

    period_ids: optional (T,) nonnegative integers; every change resets history.
    reset_mask: optional (T,) booleans; True resets before processing that row.
    Frames remain globally increasing. For per-period frame numbering, call
    separately, and always split upstream feature extraction at identity/period
    resets so its velocity pairs cannot cross those boundaries.
    """
    cfg = config if config is not None else TemporalConfig()
    if not isinstance(cfg, TemporalConfig):
        raise TypeError("config must be TemporalConfig")
    if (isinstance(fps, (bool, np.bool_)) or not np.isscalar(fps)
            or not isinstance(fps, (int, float, np.integer, np.floating))
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError("fps must be positive and finite")
    c, d = _validate(control, dead_ball)
    frames = c.frame
    n = len(frames)
    reset = np.zeros(n, dtype=bool)
    if reset_mask is not None:
        mask = np.asarray(reset_mask)
        if mask.shape != (n,) or mask.dtype.kind != "b":
            raise ValueError("reset_mask must be a boolean vector aligned to frames")
        reset |= mask
    if period_ids is not None:
        periods = np.asarray(period_ids)
        if periods.shape != (n,) or periods.dtype.kind not in "iu" or np.any(periods < 0):
            raise ValueError("period_ids must be nonnegative integers aligned to frames")
        reset[1:] |= periods[1:] != periods[:-1]
    if n:
        reset[0] = True

    labels = np.full(n, "X", dtype="U1")
    reasons = np.full(n, "unresolved", dtype="U40")
    phases = np.full(n, "UNKNOWN", dtype="U7")
    owners = np.full(n, "NONE", dtype="U4")
    ages = np.full(n, np.nan)
    control_status = np.full(n, "weak", dtype="U8")
    phase_status = np.full(n, "weak", dtype="U8")
    confirmation = np.full(n, -1, dtype=np.int64)
    # Confirmations record retrospective placements; inference never reads outputs.
    placements = []
    phase, owner, restart_required = "UNKNOWN", None, False
    owner_last = dead_last = phase_last = None
    live_run = dead_run = team_run = live_since = None
    team_side, team_kind = None, None

    def age(i, last):
        return (int(frames[i]) - int(frames[last])) / fps

    def confirmed(i, start, seconds):
        return start is not None and age(i, start) >= seconds

    def side(i, threshold, conflict):
        if conflict or not (c.ball_available[i] and c.team_available[i].all()):
            return None
        for col, name in enumerate(("H", "A")):
            other = 1 - col
            if (c.scores[i, col] >= threshold and c.scores[i, col] > c.scores[i, other]
                    and c.team_support[i, col] > c.team_support[i, other]):
                return name
        return None

    for i in range(n):
        event = None
        if reset[i]:
            phase, owner, restart_required = "UNKNOWN", None, False
            owner_last = dead_last = phase_last = None
            live_run = dead_run = team_run = live_since = None
            team_side = team_kind = None
            event = "sequence_start" if i == 0 else "period_reset"
        elif int(frames[i]) != int(frames[i - 1]) + 1:
            live_run = dead_run = team_run = None
            team_side = team_kind = None
            if age(i, i - 1) > cfg.gap_reset_s:
                phase, owner = "UNKNOWN", None
                owner_last = dead_last = phase_last = live_since = None
                # A gap does not prove that a previously confirmed stoppage ended.
                event = "gap_reset"

        # Expire history before current evidence can refresh a stale state.
        if owner is not None and age(i, owner_last) > cfg.team_hold_s:
            owner, owner_last = None, None
            event = "team_support_expired"
        if phase == "DEAD" and age(i, dead_last) > cfg.dead_hold_s:
            phase, dead_last = "UNKNOWN", None
            dead_run = None
            event = "dead_support_expired"
        if phase == "LIVE" and age(i, phase_last) > cfg.team_hold_s:
            phase, owner = "UNKNOWN", None
            owner_last = phase_last = live_since = None
            live_run = team_run = None
            team_side = team_kind = None
            event = "live_support_expired"

        available = c.ball_available[i] and c.team_available[i].all()
        control_conflict = bool(available and np.all(
            c.team_support[i] >= cfg.control_conflict_threshold))
        candidate = side(i, cfg.team_threshold, control_conflict)
        holder = side(i, cfg.team_hold_threshold, control_conflict)
        control_status[i] = ("missing" if not available else "conflict" if control_conflict
                             else candidate if candidate is not None else "weak")
        phase_conflict = bool(np.all(d.support[i] >= cfg.phase_conflict_threshold))
        dead_ok = bool(not phase_conflict and d.ball_available[i]
                       and d.scores[i, 1] >= cfg.dead_threshold
                       and d.support[i, 1] > d.support[i, 0])
        live_ok = bool(not phase_conflict and d.scores[i, 0] >= cfg.live_threshold
                       and d.support[i, 0] > d.support[i, 1]
                       and (d.motion_available[i]
                            or d.live_phase_prior[i] >= cfg.tag_threshold))
        motion_restart = bool(live_ok and d.motion_available[i]
                              and d.scores[i, 0] >= cfg.restart_live_threshold)
        tag_restart = bool(live_ok and candidate is not None
                           and d.live_phase_prior[i] >= cfg.tag_threshold
                           and d.restart_prior[i] >= cfg.tag_threshold)
        open_ok = live_ok and (not restart_required or motion_restart or tag_restart)
        phase_status[i] = ("conflict" if phase_conflict else "DEAD" if dead_ok
                           else "LIVE" if live_ok else "missing"
                           if not d.ball_available[i] and not d.tag_available[i] else "weak")

        dead_run = (i if dead_run is None else dead_run) if dead_ok else None
        live_run = (i if live_run is None else live_run) if open_ok else None
        phase_event = None
        if phase != "DEAD" and confirmed(i, dead_run, cfg.phase_confirmation_s):
            phase, owner, restart_required = "DEAD", None, True
            owner_last = phase_last = live_since = None
            dead_last = i
            team_run = live_run = None
            team_side = team_kind = None
            phase_event = "dead_confirmed"
            placements.append((dead_run, i, phase, owner, phase_event))
        elif phase != "LIVE" and confirmed(i, live_run, cfg.phase_confirmation_s):
            phase = "LIVE"
            phase_last = i
            dead_last = dead_run = None
            live_since = live_run
            phase_event = "restart_confirmed" if restart_required else "live_confirmed"
            restart_required = False
            placements.append((live_run, i, phase, None, phase_event))
        if phase == "DEAD" and dead_ok:
            dead_last = i
        if phase == "LIVE" and live_ok:
            phase_last = i

        # Team runs may accumulate alongside an opening/restart run, but cannot
        # establish ownership until LIVE is confirmed. DEAD and conflicts veto.
        eligible = (not dead_ok and not phase_conflict
                    and (phase == "LIVE" or live_run is not None))
        if eligible and holder == owner and owner is not None:
            owner_last = phase_last = i
        team_event = None
        if not eligible or candidate is None or candidate == owner:
            team_run = None
            team_side = team_kind = None
        else:
            if candidate != team_side:
                team_run, team_side = i, candidate
                team_kind = "turnover" if owner is not None else "control"
            required = cfg.turnover_s if team_kind == "turnover" else cfg.acquisition_s
            if phase == "LIVE" and confirmed(i, team_run, required):
                owner, owner_last, phase_last = candidate, i, i
                team_event = f"{team_kind}_confirmed"
                placements.append((max(team_run, live_since), i, phase, owner, team_event))
                team_run = None
                team_side = team_kind = None
        # A simultaneous team confirmation is the more specific label reason.
        event = team_event or phase_event or event
        phases[i] = phase
        owners[i] = owner if owner is not None else "NONE"
        labels[i] = "D" if phase == "DEAD" else owner if owner is not None else "X"
        if labels[i] != "X":
            ages[i] = age(i, dead_last if labels[i] == "D" else owner_last)
        if event is not None:
            reasons[i] = event
        elif phase == "DEAD":
            reasons[i] = "dead_supported" if dead_ok else "dead_continuity_hold"
        elif owner is not None:
            if dead_ok:
                reasons[i] = "dead_pending"
            elif phase_conflict:
                reasons[i] = "continuity_conflict"
            elif eligible and holder == owner:
                reasons[i] = "control_supported"
            elif eligible and candidate is not None:
                reasons[i] = "turnover_pending"
            else:
                reasons[i] = f"continuity_{control_status[i]}"
        else:
            reasons[i] = ("restart_pending" if restart_required else
                          "control_pending" if phase == "LIVE" and candidate is not None else
                          "live_pending" if live_run is not None else "unresolved")
        if event is not None and event.endswith("_confirmed"):
            confirmation[i] = frames[i]

    if cfg.mode == "offline":
        for start, end, new_phase, new_owner, reason in placements:
            interval = slice(start, end + 1)
            phases[interval] = new_phase
            owners[interval] = new_owner if new_owner is not None else "NONE"
            labels[interval] = "D" if new_phase == "DEAD" else new_owner or "X"
            ages[interval] = np.nan if new_phase == "LIVE" and new_owner is None else 0.0
            reasons[start:end] = f"{reason}_retrospective"
            reasons[end] = reason
            confirmation[interval] = frames[end]
    return PossessionResult(frames.copy(), labels, reasons, phases, owners, ages,
                            control_status, phase_status, confirmation, reset)
