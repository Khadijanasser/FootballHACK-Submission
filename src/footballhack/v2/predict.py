"""Competition inference with the frozen V2 model. Local files only.

    python -m footballhack.v2.predict --data-dir BUNDLE --output TEAM.csv

Reads the tactical ATD tracking (no Smart Tagging, video or labels), predicts
each published period independently and writes X outside the periods.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import decode, model

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "frozen.json"
MODEL = HERE / "model.joblib"


def load_frozen():
    import joblib
    frozen = json.loads(FROZEN.read_text())
    digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if digest != frozen["model_sha256"]:
        raise ValueError("model.joblib does not match frozen.json")
    bundle = joblib.load(MODEL)
    return bundle, frozen


def predict_labels(player_xy, ball_xy, teams, *, periods, pitch_m, fps, normalized, n_frames, frames=None):
    """H/A/D inside each period, X elsewhere. Periods are decoded independently."""
    bundle, frozen = load_frozen()
    players, ball = np.asarray(player_xy, dtype=float), np.asarray(ball_xy, dtype=float)
    if frames is not None and (len(frames) != n_frames or not np.array_equal(frames, np.arange(n_frames))):
        dense_p = np.full((n_frames, players.shape[1], 2), np.nan)
        dense_b = np.full((n_frames, 2), np.nan)
        dense_p[frames], dense_b[frames] = players, ball
        players, ball = dense_p, dense_b
    labels = np.full(n_frames, "X", dtype="U1")
    penalty = frozen["decoder"]
    for period in periods:
        window = slice(period.start_frame, period.end_frame)
        X, Xs, names = model.half_features(players[window], ball[window], teams, pitch_m, fps=fps, normalized=normalized)
        if names != bundle["feature_names"]:
            raise ValueError("feature definition differs from the frozen model")
        proba = model.predict_proba(bundle["models"], X, Xs)
        labels[window] = decode.viterbi(proba, **penalty)
    return labels


def main(argv=None):
    from footballhack.cli import resolve_input_paths
    from footballhack.io import load_atd
    from footballhack.pipeline import official_pipeline_config
    from footballhack.submission import write_submission
    parser = argparse.ArgumentParser(prog="footballhack.v2.predict")
    parser.add_argument("--data-dir", type=Path, help="bundle root containing tactical/; or set MTAH_DATA_DIR")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    cfg = official_pipeline_config()
    metadata, raw, _ = resolve_input_paths(data_dir=args.data_dir, no_tagging=True)
    tracking = load_atd(metadata, raw)
    if tracking.metadata.fps != cfg.fps:
        raise ValueError("unexpected frame rate")
    labels = predict_labels(tracking.player_xy, tracking.ball_xy, np.asarray(tracking.teams), periods=cfg.periods,
                            pitch_m=cfg.pitch_m, fps=cfg.fps, normalized=tracking.metadata.normalized,
                            n_frames=cfg.n_frames, frames=np.asarray(tracking.frame))
    write_submission(args.output, np.arange(cfg.n_frames), labels, n_expected=cfg.n_frames, overwrite=args.overwrite)
    print(f"Wrote validated CSV: {cfg.n_frames} rows (V2 frozen model, offline decoding)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
