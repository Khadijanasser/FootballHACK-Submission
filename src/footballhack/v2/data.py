"""Sportec halves as compact arrays. Reference labels are returned separately
and are used only as training targets / evaluation references, never as features."""
import argparse
import json
from pathlib import Path

import numpy as np

SPLIT = json.loads((Path(__file__).resolve().parents[1] / "external" / "sportec_split.json").read_text())
PARTITIONS = SPLIT["assignments"]


def cache_match(match, data_dir, cache_dir):
    """Parse one native match with the existing audited adapter and save arrays."""
    from footballhack.external.references import load_references
    from footballhack.io.sportec import load_sportec_metadata, load_sportec_tracking
    if match in PARTITIONS["holdout"]:
        raise ValueError("holdout is sealed")
    entry = next(m for m in SPLIT["matches"] if m["match_id"] == match)
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    metadata = load_sportec_metadata(data_dir / entry["metadata"]["name"])
    positions = data_dir / entry["positions"]["name"]
    halves = load_sportec_tracking(positions, metadata)
    references = load_references(positions, halves)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for index, (half, reference) in enumerate(zip(halves, references)):
        t = half.tracking
        np.savez_compressed(cache_dir / f"{match}_h{index + 1}.npz",
                            player_xy=t.player_xy.astype(np.float32), ball_xy=t.ball_xy.astype(np.float32),
                            teams=np.asarray(t.teams), labels=np.asarray(reference.labels),
                            pitch_m=np.asarray(t.metadata.pitch_m, dtype=float), fps=float(t.metadata.fps))


def load_half(cache_dir, match, half):
    """Return (tracking dict, labels). Tracking holds positions only."""
    with np.load(Path(cache_dir) / f"{match}_h{half}.npz") as z:
        tracking = dict(player_xy=z["player_xy"].astype(np.float64), ball_xy=z["ball_xy"].astype(np.float64),
                        teams=z["teams"], pitch_m=tuple(z["pitch_m"]), fps=float(z["fps"]))
        return tracking, z["labels"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("match")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    cache_match(args.match, args.data_dir, args.cache_dir)
    print("cached", args.match)
