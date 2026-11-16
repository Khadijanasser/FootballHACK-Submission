"""FootballHACK: possession analytics developed with synthetic data."""

from .features import extract_features
from .control import ControlConfig, ControlEvidence, estimate_control

from .dead_ball import DeadBallConfig, DeadBallEvidence, SmartTag, estimate_dead_ball
from .temporal import TemporalConfig, PossessionResult, decode_possession
from .io import Period, TrackingData, load_atd, load_atd_metadata, TaggingConfig, load_smart_tagging
from .pipeline import PipelineConfig, PredictionResult, official_pipeline_config, predict_possession
from .submission import validate_submission_arrays, validate_submission_csv, write_submission

__all__ = ["extract_features", "ControlConfig", "ControlEvidence", "estimate_control",
           "DeadBallConfig", "DeadBallEvidence", "SmartTag", "estimate_dead_ball",
           "TemporalConfig", "PossessionResult", "decode_possession",
           "Period", "TrackingData", "load_atd", "load_atd_metadata", "TaggingConfig", "load_smart_tagging",
           "PipelineConfig", "PredictionResult", "official_pipeline_config", "predict_possession",
           "validate_submission_arrays", "validate_submission_csv", "write_submission"]
