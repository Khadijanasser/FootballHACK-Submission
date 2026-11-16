"""Four frozen prediction methods; no reference-label imports or parameters."""
import numpy as np

from footballhack.control import ControlEvidence
from footballhack.features import extract_features
from footballhack.pipeline import PipelineConfig, predict_possession
from footballhack.temporal import decode_possession

METHODS = ('nearest_raw', 'nearest_smooth50', 'nearest_temporal', 'footballhack_causal')
SMOOTH_HALF_WIDTH = 50


def majority_vote(codes):
    """Published smooth50 procedure, independently applied within each half.

    NOBODY=-1 is not a vote. Home wins vote ties; an empty window emits D.
    This deliberately retains the baseline's missing=>D convention. It is not
    applied to FootballHACK and uses future frames (+/-50), unlike causal mode.
    """
    index = np.arange(len(codes))
    low, high = np.maximum(0, index - SMOOTH_HALF_WIDTH), np.minimum(len(codes), index + SMOOTH_HALF_WIDTH + 1)
    votes = []
    for team in (0, 1):
        cumulative = np.concatenate(([0], np.cumsum(codes == team)))
        votes.append(cumulative[high] - cumulative[low])
    home, away = votes
    return np.where(home + away == 0, 'D', np.where(home >= away, 'H', 'A'))


def predict_methods(half, *, chunk_size=4096):
    """Prediction boundary accepts a SportecHalf containing positions only.

    Frozen PipelineConfig defaults are used verbatim. Native identities do not
    enable new motion terms. No Smart Tags, ball-status flags or references.
    """
    data = half.tracking
    cfg = PipelineConfig(n_frames=len(data.frame), evidence_chunk_size=chunk_size)
    result = predict_possession(data, config=cfg)
    f = extract_features(data.frame, data.player_xy, data.ball_xy, data.teams,
                         fps=data.metadata.fps, pitch_m=data.metadata.pitch_m, normalized=False)
    distances = np.where(np.isfinite(f['ball_player_distance_m']), f['ball_player_distance_m'], np.inf)
    best = np.argmin(distances, axis=1)
    visible = np.isfinite(distances[np.arange(len(best)), best])
    # Raw and published baselines use first-in-metadata order on exact distance
    # ties, just like the published strict '<' nearest-player loop.
    codes = np.where(visible, np.where(data.teams[best] == 'H', 0, 1), -1)
    raw = np.where(codes == 0, 'H', np.where(codes == 1, 'A', 'D'))
    # Shared-decoder evidence instead marks equal team distances as uncertain.
    both = result.control.team_available.all(axis=1) & result.control.ball_available
    home_distance = np.min(np.where(data.teams[None, :] == 'H', distances, np.inf), axis=1)
    away_distance = np.min(np.where(data.teams[None, :] == 'A', distances, np.inf), axis=1)
    directional = both & (home_distance != away_distance)
    scores = np.zeros((len(best), 3))
    scores[:, 2] = 1
    for team in (0, 1):
        mask = directional & (codes == team)
        scores[mask, team], scores[mask, 2] = 1, 0
    support = np.where(result.control.team_available & result.control.ball_available[:, None],
                       scores[:, :2], np.nan)
    counts = scores[:, :2].astype(int)
    evidence = ControlEvidence(data.frame, scores, support, counts, result.control.ball_available,
                               result.control.team_available)
    decoded = decode_possession(evidence, result.dead_ball, fps=data.metadata.fps,
                                config=cfg.temporal)
    return dict(zip(METHODS, (raw, majority_vote(codes), decoded.labels, result.labels)))
