"""Frame-local, rule-based control evidence; no dead-ball labels or smoothing.

Conceptually inspired by tracking-based control detection, not a reproduction of
Vidal-Codina et al. (2022). No paper-specific thresholds are claimed.
"""
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .features import extract_features


@dataclass(frozen=True)
class ControlConfig:
    """Untuned physical heuristics, independent of competition observations.

    radius_m: 2 m is a generous immediate-touch neighborhood, not proof of control.
    relative_speed_scale_mps: 5 m/s halves motion support when relative motion is
        substantial compared with a controlled dribble; this is a soft scale.
    unknown_motion_weight: 0.5 retains reduced proximity evidence when a backward
        velocity is unavailable (first frame, gap or reacquisition).
    max_players_per_team: 3 allows nearby teammates to contribute while bounding
        crowd-size effects. Duplicate tracklets can still inflate evidence.
    ambiguity_margin: 0.2 requires a nontrivial team-support advantage.
    use_player_motion: True retains the original relative-motion rule. False
        always uses unknown_motion_weight, for sources without slot identity.
    These defaults are conservative design assumptions, not optimal parameters.
    """
    radius_m: float = 2.0
    relative_speed_scale_mps: float = 5.0
    unknown_motion_weight: float = 0.5
    max_players_per_team: int = 3
    ambiguity_margin: float = 0.2
    use_player_motion: bool = True

    def __post_init__(self):
        if not isinstance(self.use_player_motion, (bool, np.bool_)):
            raise ValueError("use_player_motion must be boolean")
        for name in ("radius_m", "relative_speed_scale_mps"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isfinite(self.unknown_motion_weight) or not 0 <= self.unknown_motion_weight <= 1:
            raise ValueError("unknown_motion_weight must be in [0, 1]")
        if not np.isfinite(self.ambiguity_margin) or not 0 <= self.ambiguity_margin < 1:
            raise ValueError("ambiguity_margin must be in [0, 1)")
        if (isinstance(self.max_players_per_team, bool)
                or not isinstance(self.max_players_per_team, (int, np.integer))
                or self.max_players_per_team < 1):
            raise ValueError("max_players_per_team must be a positive integer")


@dataclass(frozen=True)
class ControlEvidence:
    """Temporal-model boundary; all arrays preserve input frame order.

    scores: (T, 3) columns H, A, Unknown, finite in [0,1], rows sum to 1.
        Heuristic evidence, NOT probabilities. Unknown is neither D nor a final X.
    team_support: (T, 2) raw H/A support in [0,1]; NaN when the ball or that
        team's players are unobserved. Zero means observed but no local support.
    candidate_count: (T, 2) number of observed H/A players strictly within radius,
        before the top-k cap; zero when ball tracking is missing.
    ball_available: (T,) and team_available: (T,2) observation diagnostics.
        A team is available if at least one player is seen, not if all are seen.
    """
    frame: NDArray
    scores: NDArray
    team_support: NDArray
    candidate_count: NDArray
    ball_available: NDArray
    team_available: NDArray
    score_columns = ("H", "A", "Unknown")


def estimate_control(
    frame_ids: ArrayLike, player_xy: ArrayLike, ball_xy: ArrayLike, teams: ArrayLike,
    *, fps: float = 25.0, pitch_m: tuple[float, float] = (105.3, 68.0),
    normalized: bool = True, config: ControlConfig | None = None,
) -> ControlEvidence:
    """Reuse feature validation, units, gaps and missingness semantics.

    Player support = max(0, 1 - distance/radius) * motion_weight.
    Motion weight = 1 / (1 + (relative_speed/scale)**2), or the configured
    unknown weight when velocity is unavailable or use_player_motion is False.
    Relative speed uses the vector
    difference, so equally fast opposite motion is discounted.
    Team support = 1 - product(1 - top_k_player_support), a bounded pooling rule,
    not an independence assumption or probability calculation.
    Team evidence = max(0, abs(H-A)-margin)/(1-margin), assigned to the stronger
    side; Unknown receives the remainder. Equal support never favors either team.
    Both teams and ball must be observed to emit directional evidence.

    No frame history is pooled. Motion uses only the backward differences from
    features.py; callers must split at period/track-identity resets as documented
    there. No API performs file I/O, fitting, or data-dependent calibration.
    """
    cfg = config if config is not None else ControlConfig()
    f = extract_features(frame_ids, player_xy, ball_xy, teams, fps=fps,
                         pitch_m=pitch_m, normalized=normalized)
    sides = np.asarray(teams)
    distance = f["ball_player_distance_m"]
    relative = np.linalg.norm(f["player_velocity_mps"]
                              - f["ball_velocity_mps"][:, None, :], axis=-1)
    motion = np.full_like(relative, cfg.unknown_motion_weight)
    valid_motion = np.isfinite(relative) & cfg.use_player_motion
    with np.errstate(over="ignore"):
        motion[valid_motion] = 1 / (1 + (relative[valid_motion]
                                       / cfg.relative_speed_scale_mps) ** 2)
    proximity = np.where(np.isfinite(distance),
                         np.clip(1 - distance / cfg.radius_m, 0, 1), 0)
    player_support = proximity * motion
    n = len(f["frame"])
    support = np.full((n, 2), np.nan)
    counts = np.zeros((n, 2), dtype=int)
    available = np.zeros((n, 2), dtype=bool)
    for col, side in enumerate(("H", "A")):
        mask = sides == side
        available[:, col] = f["player_available"][:, mask].any(axis=1)
        counts[:, col] = ((distance[:, mask] < cfg.radius_m)
                          & np.isfinite(distance[:, mask])).sum(axis=1)
        strongest = np.sort(player_support[:, mask], axis=1)[:, -cfg.max_players_per_team:]
        pooled = 1 - np.prod(1 - strongest, axis=1)
        observed = f["ball_available"] & available[:, col]
        support[observed, col] = pooled[observed]
    scores = np.zeros((n, 3))
    scores[:, 2] = 1
    usable = f["ball_available"] & available.all(axis=1)
    delta = support[usable, 0] - support[usable, 1]
    strength = np.clip((np.abs(delta) - cfg.ambiguity_margin)
                       / (1 - cfg.ambiguity_margin), 0, 1)
    scores[usable, 0] = np.where(delta > 0, strength, 0)
    scores[usable, 1] = np.where(delta < 0, strength, 0)
    scores[usable, 2] = 1 - strength
    return ControlEvidence(f["frame"], scores, support, counts,
                           f["ball_available"], available)
