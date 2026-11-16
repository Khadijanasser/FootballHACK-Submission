"""Independent, frame-local live/dead evidence from tracking and automatic tags."""
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .features import extract_features

SET_PIECES = frozenset({"throw_in", "corner", "free_kick", "goal_kick", "kick_off", "penalty"})
TAG_KINDS = SET_PIECES | {"live_phase", "restart", "unknown"}


@dataclass(frozen=True)
class SmartTag:
    """Automatic tag on [start_frame, end_frame), in tactical video frames.

    kind: throw_in/corner/free_kick/goal_kick/kick_off/penalty, live_phase,
    restart, or unknown. A restart tag is only a phase prior, not a known touch.
    confidence is an upstream quality weight, NOT a fitted probability.
    No team label or ground truth is accepted. Map source codes explicitly;
    unrecognized or ambiguous codes should be mapped to unknown.
    """
    start_frame: int
    end_frame: int
    kind: str
    confidence: float = 1.0

    def __post_init__(self):
        for value in (self.start_frame, self.end_frame):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError("tag frames must be nonnegative integers")
        if self.end_frame <= self.start_frame:
            raise ValueError("tag interval must be nonempty and half-open")
        if self.kind not in TAG_KINDS:
            raise ValueError("unrecognized kind; map ambiguous source tags to unknown")
        if not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class DeadBallConfig:
    """Untuned physical tolerances and conservative evidence weights.

    Boundary buffer tolerates sub-metre localization/ball-extent uncertainty.
    Slow <=0.5 m/s is only corroboration for set-piece preparation; moving
    >=1.5 m/s is only live support safely inside the pitch. Between is uncertain.
    Weights are engineering assumptions, not probabilities or optimal settings.
    """
    boundary_buffer_m: float = 0.5
    slow_speed_mps: float = 0.5
    moving_speed_mps: float = 1.5
    outside_weight: float = 0.9
    moving_weight: float = 0.65
    live_tag_weight: float = 0.45
    preparation_weight: float = 0.45
    restart_weight: float = 0.75

    def __post_init__(self):
        for name in ("boundary_buffer_m", "slow_speed_mps", "moving_speed_mps"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.moving_speed_mps <= self.slow_speed_mps:
            raise ValueError("moving_speed_mps must exceed slow_speed_mps")
        for name in ("outside_weight", "moving_weight", "live_tag_weight",
                     "preparation_weight", "restart_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class DeadBallEvidence:
    """Temporal boundary: scores (T,3) ordered LIVE, DEAD, UNKNOWN.

    Scores are heuristic weights summing to one, not calibrated probabilities or
    final submission labels. support (T,2) holds raw LIVE/DEAD strengths; opposing
    support is canceled into UNKNOWN. Other arrays have shape (T,).
    boundary_clearance_m is positive inside, negative outside, NaN if missing.
    Missing tracking can retain weak independent tag evidence, never geometry or
    speed evidence. tag_available excludes unknown/zero-confidence tags.
    """
    frame: NDArray
    scores: NDArray
    support: NDArray
    ball_available: NDArray
    motion_available: NDArray
    boundary_clearance_m: NDArray
    tag_available: NDArray
    set_piece_prior: NDArray
    live_phase_prior: NDArray
    restart_prior: NDArray
    score_columns = ("LIVE", "DEAD", "UNKNOWN")


def estimate_dead_ball(
    frame_ids: ArrayLike, player_xy: ArrayLike, ball_xy: ArrayLike, teams: ArrayLike,
    *, tags: Iterable[SmartTag] = (), fps: float = 25.0,
    pitch_m: tuple[float, float] = (105.3, 68.0), normalized: bool = True,
    config: DeadBallConfig | None = None,
) -> DeadBallEvidence:
    """Estimate frame-local evidence; never calls or depends on team control.

    Coordinates must use a corner origin with pitch [0,length] x [0,width],
    normalized or metres as in features.py. Center-origin inputs need conversion.
    Velocities inherit features.py gap/reset semantics. No history is smoothed.
    Tag overlaps use maximum confidence per category, so duplicates do not vote.
    """
    cfg = config if config is not None else DeadBallConfig()
    f = extract_features(frame_ids, player_xy, ball_xy, teams, fps=fps,
                         pitch_m=pitch_m, normalized=normalized)
    frames = f["frame"]
    n = len(frames)
    piece, phase, restart = (np.zeros(n) for _ in range(3))
    for tag in tags:
        if not isinstance(tag, SmartTag):
            raise TypeError("tags must contain SmartTag instances")
        active = (frames >= tag.start_frame) & (frames < tag.end_frame)
        target = piece if tag.kind in SET_PIECES else phase if tag.kind == "live_phase" else restart if tag.kind == "restart" else None
        if target is not None:
            target[active] = np.maximum(target[active], tag.confidence)
    xy = np.array(ball_xy, dtype=float, copy=True)
    if normalized:
        xy *= np.asarray(pitch_m)
    clearance = np.min(np.column_stack((xy[:, 0], xy[:, 1],
                                       pitch_m[0] - xy[:, 0], pitch_m[1] - xy[:, 1])), axis=1)
    clearance[~f["ball_available"]] = np.nan
    inside = clearance > cfg.boundary_buffer_m
    outside = clearance < -cfg.boundary_buffer_m
    speed = f["ball_speed_mps"]
    moving = np.isfinite(speed) & (speed >= cfg.moving_speed_mps)
    slow = np.isfinite(speed) & (speed <= cfg.slow_speed_mps)
    live = np.maximum(phase * cfg.live_tag_weight,
                      inside * moving * np.maximum(cfg.moving_weight, restart * cfg.restart_weight))
    # Only observed slow balls corroborate set-piece preparation; tags alone
    # and stationary balls without tags never assert DEAD.
    dead = np.maximum(outside * cfg.outside_weight,
                      slow * f["ball_available"] * piece * cfg.preparation_weight)
    delta = live - dead
    scores = np.column_stack((np.maximum(delta, 0), np.maximum(-delta, 0), 1 - np.abs(delta)))
    return DeadBallEvidence(frames, scores, np.column_stack((live, dead)),
                            f["ball_available"], np.isfinite(speed), clearance,
                            (piece > 0) | (phase > 0) | (restart > 0), piece, phase, restart)
