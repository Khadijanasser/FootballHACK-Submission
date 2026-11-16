"""Local deterministic prediction, with explicit coverage and period contracts."""
from dataclasses import dataclass, field, fields, replace

import numpy as np
from numpy.typing import NDArray

from .control import ControlConfig, ControlEvidence, estimate_control
from .dead_ball import DeadBallConfig, DeadBallEvidence, SmartTag, estimate_dead_ball
from .io import Period, TrackingData
from .io._xml import positive_number
from .temporal import PossessionResult, TemporalConfig, decode_possession


OFFICIAL_N_FRAMES = 199_319


@dataclass(frozen=True)
class PipelineConfig:
    n_frames: int
    periods: tuple[Period, ...] = ()
    fps: float | None = None
    pitch_m: tuple[float, float] | None = None
    gap_policy: str = "error"
    evidence_chunk_size: int = 4096
    # Export slots have no persistent identities. Use full proximity support
    # without player velocity terms; legacy ControlConfig defaults are unchanged.
    control: ControlConfig = field(default_factory=lambda: ControlConfig(
        use_player_motion=False, unknown_motion_weight=1.0))
    dead_ball: DeadBallConfig = field(default_factory=DeadBallConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)

    def __post_init__(self):
        for name in ("n_frames", "evidence_chunk_size"):
            value = getattr(self, name)
            if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
                    or value <= 0 or value > np.iinfo(np.int64).max):
                raise ValueError(f"{name} must be a positive integer")
        if self.gap_policy not in ("error", "unknown"):
            raise ValueError("gap_policy must be 'error' or 'unknown'")
        if self.fps is not None:
            positive_number(self.fps, "fps")
        if self.pitch_m is not None:
            if not isinstance(self.pitch_m, tuple) or len(self.pitch_m) != 2:
                raise ValueError("pitch_m must be a pair of metric dimensions")
            for value in self.pitch_m:
                positive_number(value, "pitch_m")
        if not isinstance(self.periods, tuple) or any(not isinstance(p, Period) for p in self.periods):
            raise ValueError("periods must be a tuple of Period objects")
        _validate_periods(self.periods, self.n_frames)
        for name, cls in (("control", ControlConfig), ("dead_ball", DeadBallConfig), ("temporal", TemporalConfig)):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"{name} must be {cls.__name__}")


def _validate_periods(periods, n_frames):
    seen, previous_end = set(), 0
    for period in periods:
        if period.id in seen or period.start_frame < previous_end or period.end_frame > n_frames:
            raise ValueError("periods must be ordered, nonoverlapping, uniquely identified and within coverage")
        seen.add(period.id)
        previous_end = period.end_frame


def official_pipeline_config(**overrides) -> PipelineConfig:
    """Published DATA.md video windows, not inferred labels or GT counters.

    These constants remain at the profile boundary. Generic tests and other
    datasets supply their own dimensions, frame counts and period intervals.
    """
    values = dict(n_frames=OFFICIAL_N_FRAMES, fps=25.0, pitch_m=(105.3, 68.0),
                  periods=(Period(1, 11185, 90652), Period(2, 110808, 190198)))
    values.update(overrides)
    return PipelineConfig(**values)


@dataclass(frozen=True)
class PredictionResult:
    possession: PossessionResult
    control: ControlEvidence
    dead_ball: DeadBallEvidence
    observed_frame: NDArray
    period_ids: NDArray
    pitch_m: tuple[float, float]
    fps: float

    @property
    def frame(self):
        return self.possession.frame

    @property
    def labels(self):
        return self.possession.labels


def _concatenate(parts, cls):
    return cls(**{f.name: np.concatenate([getattr(part, f.name) for part in parts]) for f in fields(cls)})


def predict_possession(
    tracking: TrackingData, tags=(), *, config: PipelineConfig, reset_frames=(),
) -> PredictionResult:
    """Reuse existing estimators (and their feature extraction) and decoder.

    Evidence chunks have one-frame overlap within a reset segment, preserving
    causal differences without carrying velocity across period/identity resets.
    Chunk size changes resource use only, never prediction semantics.
    """
    if not isinstance(tracking, TrackingData) or not isinstance(config, PipelineConfig):
        raise TypeError("expected TrackingData and PipelineConfig")
    source = np.asarray(tracking.frame)
    n = config.n_frames
    players, ball, teams = np.asarray(tracking.player_xy), np.asarray(tracking.ball_xy), np.asarray(tracking.teams)
    if (source.ndim != 1 or source.dtype.kind not in "iu" or not len(source)
            or np.any(source < 0) or np.any(source >= n) or np.any(source[1:] <= source[:-1])):
        raise ValueError("tracking frames must be nonempty, ordered, unique and within configured coverage")
    if (players.shape != (len(source), len(tracking.metadata.slots), 2)
            or ball.shape != (len(source), 2)
            or teams.shape != (len(tracking.metadata.slots),)
            or not np.array_equal(teams, [slot.side for slot in tracking.metadata.slots])):
        raise ValueError("tracking arrays do not match metadata slot/team order")
    if (players.dtype.kind not in "iuf" or ball.dtype.kind not in "iuf"
            or np.isinf(players).any() or np.isinf(ball).any()):
        raise ValueError("tracking observations must be numeric finite values or NaN")
    fps = tracking.metadata.fps if config.fps is None else config.fps
    fps = positive_number(fps, "fps")
    if fps != tracking.metadata.fps:
        raise ValueError("pipeline FPS disagrees with ATD metadata; do not rescale the timeline silently")
    pitch = tracking.metadata.pitch_m if config.pitch_m is None else config.pitch_m
    pitch = tuple(positive_number(value, "pitch_m") for value in pitch)
    periods = config.periods or tracking.metadata.periods
    if not periods:
        raise ValueError("explicit period boundaries are required; supply periods or EPTS period metadata")
    _validate_periods(periods, n)
    frames = np.arange(n, dtype=np.int64)
    observed = np.zeros(n, dtype=bool)
    observed[source] = True
    if not observed.all():
        if config.gap_policy == "error":
            raise ValueError("incomplete ATD coverage; use gap_policy='unknown' to insert NaN observations explicitly")
        dense_players = np.full((n, len(teams), 2), np.nan)
        dense_ball = np.full((n, 2), np.nan)
        dense_players[source], dense_ball[source] = players, ball
        players, ball = dense_players, dense_ball
    period_ids = np.zeros(n, dtype=np.int64)
    for period in periods:
        period_ids[period.start_frame:period.end_frame] = period.id
    reset = np.zeros(n, dtype=bool)
    reset[0] = True
    reset[1:] = period_ids[1:] != period_ids[:-1]
    for frame in reset_frames:
        if (not isinstance(frame, (int, np.integer)) or isinstance(frame, (bool, np.bool_))
                or not 0 <= frame < n):
            raise ValueError("reset_frames must contain zero-based integer frames within coverage")
        reset[frame] = True
    tags = tuple(tags)
    if any(not isinstance(tag, SmartTag) for tag in tags):
        raise TypeError("tags must contain SmartTag objects")
    if any(tag.end_frame > n for tag in tags):
        raise ValueError("Smart Tagging exceeds the configured tactical timeline")
    c_parts, d_parts = [], []
    boundaries = list(np.flatnonzero(reset)) + [n]
    for start, end in zip(boundaries, boundaries[1:]):
        for chunk_start in range(start, end, config.evidence_chunk_size):
            chunk_end = min(end, chunk_start + config.evidence_chunk_size)
            halo = max(start, chunk_start - 1)
            selection = slice(halo, chunk_end)
            kwargs = dict(fps=fps, pitch_m=pitch, normalized=tracking.metadata.normalized)
            c = estimate_control(frames[selection], players[selection], ball[selection], teams,
                                 config=config.control, **kwargs)
            # Dead-ball evidence needs only ball tracking. Avoid duplicating full
            # player-feature tensors in this independent estimator.
            active_tags = [tag for tag in tags if tag.start_frame < chunk_end and tag.end_frame > halo]
            d = estimate_dead_ball(frames[selection], np.empty((chunk_end - halo, 0, 2)),
                                   ball[selection], [], tags=active_tags,
                                   config=config.dead_ball, **kwargs)
            offset = chunk_start - halo
            c_parts.append(replace(c, **{f.name: getattr(c, f.name)[offset:] for f in fields(c)}))
            d_parts.append(replace(d, **{f.name: getattr(d, f.name)[offset:] for f in fields(d)}))
    control = _concatenate(c_parts, ControlEvidence)
    dead_ball = _concatenate(d_parts, DeadBallEvidence)
    decoded = decode_possession(control, dead_ball, fps=fps, config=config.temporal,
                                period_ids=period_ids, reset_mask=reset)
    outside = period_ids == 0
    decoded.labels[outside] = "X"
    decoded.phase[outside] = "UNKNOWN"
    decoded.possession[outside] = "NONE"
    decoded.reason[outside] = "outside_period"
    decoded.support_age_s[outside] = np.nan
    decoded.confirmation_frame[outside] = -1
    return PredictionResult(decoded, control, dead_ball, observed, period_ids, pitch, fps)
