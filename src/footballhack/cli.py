"""Explicit local-file entry point. No network clients or AI-service calls."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

from .io import Period, TaggingConfig, load_atd, load_smart_tagging
from .pipeline import OFFICIAL_N_FRAMES, PipelineConfig, official_pipeline_config, predict_possession
from .submission import validate_submission_csv, write_submission


def resolve_input_paths(*, metadata=None, tracking=None, tagging=None, data_dir=None,
                        stem="BAR-ATH_tactical", no_tagging=False):
    """Resolve only, without globbing, inspecting directories or opening files."""
    if not stem or Path(stem).name != stem or stem in (".", ".."):
        raise ValueError("stem must be a filename stem without directory components")
    if no_tagging and tagging is not None:
        raise ValueError("--tagging and --no-tagging cannot be combined")
    root = data_dir or os.environ.get("MTAH_DATA_DIR")
    suffixes = ("_atd.xml", "_atd.txt", "_smart_tagging.xml")
    paths = []
    for explicit, suffix in zip((metadata, tracking, tagging), suffixes):
        if suffix == "_smart_tagging.xml" and no_tagging:
            paths.append(None)
        elif explicit is not None:
            paths.append(Path(explicit))
        elif root:
            paths.append(Path(root) / "tactical" / (stem + suffix))
        else:
            raise ValueError("provide explicit input paths, --data-dir, or MTAH_DATA_DIR; omit tags with --no-tagging")
    return tuple(paths)


def _period(text):
    try:
        values = text.split(":")
        if len(values) != 3:
            raise ValueError
        return Period(*(int(value) for value in values))
    except ValueError as error:
        raise argparse.ArgumentTypeError("period must be ID:START:END in half-open video frames") from error


def _configuration(args):
    raw = {}
    if args.config is not None:
        raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("configuration must be a JSON object")
    raw = raw.copy()
    tagging_values = raw.pop("tagging", {})
    if not isinstance(tagging_values, dict):
        raise ValueError("tagging configuration must be an object")
    tagging_values = tagging_values.copy()
    if "live_codes" in tagging_values:
        if not isinstance(tagging_values["live_codes"], list):
            raise ValueError("live_codes must be a JSON array")
        tagging_values["live_codes"] = tuple(tagging_values["live_codes"])
    if "set_piece_types" in tagging_values:
        pairs = tagging_values["set_piece_types"]
        tagging_values["set_piece_types"] = tuple(pairs.items()) if isinstance(pairs, dict) else tuple(tuple(p) for p in pairs)
    tagging_cfg = TaggingConfig(**tagging_values)
    reset_frames = raw.pop("reset_frames", [])
    rows = args.rows if args.rows is not None else raw.get("n_frames")
    if args.profile == "official":
        cfg = official_pipeline_config()
        if rows is not None and rows != OFFICIAL_N_FRAMES:
            raise ValueError("official profile requires 199319 frames; use --profile generic for other datasets")
    else:
        if rows is None:
            raise ValueError("generic profile requires --rows or n_frames in configuration")
        cfg = PipelineConfig(n_frames=rows)
    for name in ("control", "dead_ball", "temporal"):
        if name in raw:
            values = raw.pop(name)
            if not isinstance(values, dict):
                raise ValueError(f"{name} configuration must be an object")
            cfg = replace(cfg, **{name: replace(getattr(cfg, name), **values)})
    if "periods" in raw:
        raw["periods"] = tuple(Period(**p) for p in raw["periods"])
    if "pitch_m" in raw and raw["pitch_m"] is not None:
        raw["pitch_m"] = tuple(raw["pitch_m"])
    # Unknown keys raise TypeError instead of silently disabling validation.
    cfg = replace(cfg, **raw)
    overrides = {}
    for arg, name in (("rows", "n_frames"), ("gap_policy", "gap_policy"),
                      ("chunk_size", "evidence_chunk_size")):
        value = getattr(args, arg)
        if value is not None:
            overrides[name] = value
    if args.period is not None:
        overrides["periods"] = tuple(args.period)
    if args.pitch is not None:
        overrides["pitch_m"] = tuple(args.pitch)
    if args.mode is not None:
        overrides["temporal"] = replace(cfg.temporal, mode=args.mode)
    cfg = replace(cfg, **overrides)
    if args.profile == "official" and cfg.n_frames != OFFICIAL_N_FRAMES:
        raise ValueError("official profile requires 199319 frames")
    return cfg, tagging_cfg, list(reset_frames) + args.reset_frame


def _parser():
    parser = argparse.ArgumentParser(prog="footballhack", description="Local possession prediction and CSV validation")
    subcommands = parser.add_subparsers(dest="command", required=True)
    predict = subcommands.add_parser("predict", help="load local ATD/Smart Tagging and write a validated CSV")
    predict.add_argument("--data-dir", type=Path, help="bundle root containing tactical/; overrides MTAH_DATA_DIR")
    predict.add_argument("--stem", default="BAR-ATH_tactical")
    predict.add_argument("--metadata", type=Path)
    predict.add_argument("--tracking", type=Path)
    predict.add_argument("--tagging", type=Path)
    predict.add_argument("--no-tagging", action="store_true")
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--overwrite", action="store_true")
    predict.add_argument("--profile", choices=("official", "generic"), default="official")
    predict.add_argument("--rows", type=int, help="coverage length; official profile is fixed to 199319")
    predict.add_argument("--period", type=_period, action="append", help="ID:START:END; repeat for each half/period")
    predict.add_argument("--reset-frame", type=int, action="append", default=[])
    predict.add_argument("--pitch", type=float, nargs=2, metavar=("LENGTH_M", "WIDTH_M"))
    predict.add_argument("--mode", choices=("causal", "offline"))
    predict.add_argument("--gap-policy", choices=("error", "unknown"))
    predict.add_argument("--chunk-size", type=int)
    predict.add_argument("--config", type=Path, help="JSON overrides for all prediction and adapter settings")
    predict.add_argument("--manifest", type=Path, help="optional effective configuration JSON (contains no observations)")
    validate = subcommands.add_parser("validate", help="check CSV format and coverage without loading any match data")
    validate.add_argument("csv", type=Path)
    validate.add_argument("--rows", type=int, default=OFFICIAL_N_FRAMES)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            count = validate_submission_csv(args.csv, n_expected=args.rows)
            print(f"Valid CSV: {count} rows, frames 0..{count - 1}")
            return 0
        cfg, tag_cfg, reset_frames = _configuration(args)
        metadata, raw, tagging = resolve_input_paths(
            metadata=args.metadata, tracking=args.tracking, tagging=args.tagging,
            data_dir=args.data_dir, stem=args.stem, no_tagging=args.no_tagging)
        inputs = [path.resolve() for path in (metadata, raw, tagging, args.config) if path is not None]
        if args.output.resolve() in inputs:
            raise ValueError("output must not overwrite an input")
        if args.manifest is not None and args.manifest.resolve() in inputs + [args.output.resolve()]:
            raise ValueError("manifest must be distinct from inputs and CSV output")
        for output in (args.output, args.manifest):
            if output is not None and output.exists() and not args.overwrite:
                raise FileExistsError("output exists; choose a new path or pass --overwrite")
        tracking = load_atd(metadata, raw)
        tags = () if tagging is None else load_smart_tagging(tagging, fps=tracking.metadata.fps, config=tag_cfg).tags
        result = predict_possession(tracking, tags, config=cfg, reset_frames=reset_frames)
        write_submission(args.output, result.frame, result.labels, n_expected=cfg.n_frames, overwrite=args.overwrite)
        if args.manifest is not None:
            payload = dict(pipeline=asdict(cfg), tagging=asdict(tag_cfg), reset_frames=reset_frames,
                           effective_fps=result.fps, effective_pitch_m=result.pitch_m,
                           metadata_pitch_m=tracking.metadata.pitch_m,
                           team_ids=tracking.metadata.team_ids, team_names=tracking.metadata.team_names)
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Wrote validated CSV: {cfg.n_frames} rows, {int((~result.observed_frame).sum())} inserted observations")
        print(f"Mode: {cfg.temporal.mode}; FPS: {result.fps:g}; pitch: {result.pitch_m[0]:g} x {result.pitch_m[1]:g} m")
        return 0
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"footballhack: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
