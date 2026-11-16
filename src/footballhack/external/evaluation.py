"""Unchanged official scoring plus explicitly separate external diagnostics."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from footballhack.pipeline import PipelineConfig


def protocol():
    return json.loads(Path(__file__).with_name('provenance.json').read_text())


def verify_frozen_sources():
    manifest = protocol()
    package = Path(__file__).resolve().parents[1]
    for name, expected in manifest['model_sha256'].items():
        if sha256((package / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen model source changed: {name}")
    actual_config = json.loads(json.dumps(asdict(PipelineConfig(n_frames=1))))
    if actual_config != manifest['pipeline_defaults']:
        raise ValueError("frozen pipeline defaults changed")
    path = Path(__file__).with_name('_official') / 'possession_metrics.py'
    if sha256(path.read_bytes()).hexdigest() != manifest['official_scorer_sha256']:
        raise ValueError("official scorer checksum mismatch")
    split_path = Path(__file__).with_name('sportec_split.json')
    if sha256(split_path.read_bytes()).hexdigest() != manifest['split_manifest_sha256']:
        raise ValueError("frozen split manifest changed")
    return manifest


def _scorer():
    verify_frozen_sources()
    try:
        from ._official import possession_metrics
    except ImportError as error:
        raise ValueError("install requirements-external.lock to run the official scorer") from error
    return possession_metrics


def audit_match(halves, references):
    """Coverage audit only. No predictions or accuracy scores are computed."""
    gate = protocol()['reference_coverage_gate']
    if len(halves) != 2 or len(references) != 2:
        raise ValueError("both halves required")
    results = []
    for half, reference in zip(halves, references):
        _check_alignment(half, reference)
        coverage = reference.coverage
        adequate = coverage['valid_fraction'] >= gate['minimum_valid_fraction_per_half']
        if gate['require_H_A_D_each_half']:
            adequate &= all(coverage['label_counts'][k] > 0 for k in 'HAD')
        data = half.tracking
        results.append(dict(section=half.section, reference=coverage,
                            fps=data.metadata.fps, n_frames=len(data.frame),
                            duration_s=len(data.frame) / data.metadata.fps,
                            team_ids=data.metadata.team_ids, team_names=data.metadata.team_names,
                            coordinate_system='corner-origin metres; provider centre-origin translated by pitch/2',
                            ball_xy_range_m=_xy_range(data.ball_xy),
                            player_xy_range_m=_xy_range(data.player_xy),
                            timestamp_start_us=int(half.timestamp_us[0]),
                            timestamp_end_inclusive_us=int(half.timestamp_us[-1]),
                            pitch_m=data.metadata.pitch_m,
                            native_frame_start=int(half.native_frame[0]),
                            native_frame_end_inclusive=int(half.native_frame[-1]),
                            missing_ball_frames=int((~np.isfinite(data.ball_xy).all(axis=1)).sum()),
                            missing_player_observations=int((~np.isfinite(data.player_xy).all(axis=2)).sum()),
                            coverage_gate_passed=bool(adequate)))
    return dict(match_id=halves[0].match_id, halves=results,
                coverage_gate_passed=all(r['coverage_gate_passed'] for r in results))


def _xy_range(values):
    flat = np.asarray(values).reshape(-1, 2)
    return {axis: [float(finite.min()), float(finite.max())] if len(finite) else None
            for i, axis in enumerate(('x', 'y'))
            for finite in (flat[np.isfinite(flat[:, i]), i],)}


def check_native_reference_isolation(path, metadata, halves):
    """Remove all non-position attributes and verify identical prediction inputs.

    No native predictions or scores are computed. Together with deterministic
    synthetic prediction tests, equality proves references cannot change model
    output. Retain the normal loader's schema validation on the original pass.
    """
    from footballhack.io.sportec import _tracking_from_frames, iter_sportec_frames
    frames = ((context, {k: v for k, v in attrs.items() if k in ('N', 'T', 'X', 'Y')})
              for context, attrs in iter_sportec_frames(path))
    shadow = _tracking_from_frames(frames, metadata, 250_000)
    if len(shadow) != len(halves):
        raise ValueError('reference isolation changed half coverage')
    for original, stripped in zip(halves, shadow):
        pairs = [(original.native_frame, stripped.native_frame), (original.timestamp_us, stripped.timestamp_us)]
        pairs.extend((getattr(original.tracking, k), getattr(stripped.tracking, k))
                     for k in ('frame', 'player_xy', 'ball_xy'))
        if (original.tracking.metadata != stripped.tracking.metadata
                or not np.array_equal(original.tracking.teams, stripped.tracking.teams)
                or any(not np.array_equal(a, b, equal_nan=True) for a, b in pairs)):
            raise ValueError('reference fields changed prediction inputs')
    return dict(passed=True, check='identical timelines, positions and team metadata after removing all non-position frame fields',
                native_predictions_computed=False)


def _check_alignment(half, reference):
    n = len(half.native_frame)
    if (half.match_id != reference.match_id or half.section != reference.section
            or not np.array_equal(half.native_frame, reference.native_frame)
            or np.asarray(reference.labels).shape != (n,)):
        raise ValueError("reference/prediction half or native frame alignment mismatch")
    if (half.tracking.metadata.fps != 25 or n < 2
            or not np.array_equal(half.tracking.frame, np.arange(n))
            or not np.all(np.diff(half.native_frame) == 1)
            or len(half.timestamp_us) != n or not np.all(np.diff(half.timestamp_us) == 40_000)):
        raise ValueError("official scoring requires verified dense 25 Hz halves")


def _transition_summary(pm, gt, pred):
    result = pm.transition_scores(gt, pred, 25, typed='pair')
    return {key: value for key, value in result.items() if key not in ('pairs', 'missed', 'spurious')}


def score_half(half, reference, predictions):
    """Use the exact official evaluator on one half, no resampling/mask edits.

    Reference X runs are left to official valid_runs. Frame scores charge raw
    predicted X; temporal scores use official fill_abstentions within GT runs.
    """
    _check_alignment(half, reference)
    pm = _scorer()
    gt = pm.label_codes(reference.labels)
    pred = pm.label_codes(predictions)
    if len(pred) != len(gt):
        raise ValueError("prediction frame count mismatch")
    if not np.any(gt != pm.X):
        raise ValueError("no valid reference frames")
    result = pm.evaluate(gt, pred)
    # Return aggregate latency/count diagnostics, never framewise pairs.
    for values in result['transitions'].values():
        for key in ('pairs', 'missed', 'spurious'):
            values.pop(key)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    g_trans, p_trans = pm.transitions(gt, runs), pm.transitions(filled, runs)
    turnovers = lambda trans: [t for t in trans if {t[1], t[2]} == {pm.H, pm.A}]
    to_score = _transition_summary(pm, turnovers(g_trans), turnovers(p_trans))
    stoppages = lambda trans: [t for t in trans if t[2] == pm.D]
    restarts = lambda trans: [t for t in trans if t[1] == pm.D and t[2] in (pm.H, pm.A)]
    missing = ~np.isfinite(half.tracking.ball_xy).all(axis=1)
    valid_missing = missing & (gt != pm.X)
    cm = pm.confusion(gt[valid_missing], pred[valid_missing])
    missing_classes = {}
    for i, label in enumerate(pm.SCORED):
        p, r, f = pm.per_class_prf(pm.scored_confusion(cm), i)
        missing_classes[label] = dict(precision=p, recall=r, f1=f, support=int(cm[i].sum()))
    result['diagnostics'] = dict(
        turnover_timing_25_frames=to_score,
        false_possession_switches=to_score['n_spurious'],
        false_switches_per_90_valid_minutes=to_score['n_spurious'] * 90 * 60 * 25 / int(np.count_nonzero(gt != pm.X)),
        stoppage_entry_timing_25_frames=_transition_summary(pm, stoppages(g_trans), stoppages(p_trans)),
        restart_timing_25_frames=_transition_summary(pm, restarts(g_trans), restarts(p_trans)),
        dead_ball_precision_recall=result['per_class']['D'],
        missing_ball=dict(n_frames=int(missing.sum()), n_valid_reference=int(valid_missing.sum()),
                          confusion=cm.tolist(), per_class=missing_classes,
                          accuracy=float(np.trace(cm[:3, :3]) / valid_missing.sum()) if valid_missing.any() else None),
        duration_s=dict(
            reference_possession=sum(b-a for a, b, k in pm.possession_spells(gt, runs)) / 25,
            predicted_possession=sum(b-a for a, b, k in pm.possession_spells(filled, runs)) / 25,
            reference_dead=sum(b-a for a, b, k in pm.segments(gt, runs) if k == pm.D) / 25,
            predicted_dead=sum(b-a for a, b, k in pm.segments(filled, runs) if k == pm.D) / 25))
    return result


def aggregate_halves(results):
    """Explicit equal-half means; no artificial spell concatenation across halves."""
    if not results:
        raise ValueError("no half results to aggregate")
    pm = _scorer()
    cm = sum((np.array(r['confusion'], dtype=np.int64) for r in results), np.zeros((4, 4), dtype=np.int64))
    per_class = {}
    for index, label in enumerate(pm.SCORED):
        p, r, f = pm.per_class_prf(pm.scored_confusion(cm), index)
        per_class[label] = dict(precision=p, recall=r, f1=f, support=int(cm[index].sum()))
    summary = {key: float(np.mean([r[key] for r in results])) for key in ('spell_f1_mean', 'macro_f1')}
    summary.update(aggregation='equal-half means', n_halves=len(results), per_class_pooled=per_class,
                   confusion_pooled=cm.tolist())
    for key in ('n_frames', 'n_graded', 'n_gt_spells', 'n_pred_spells', 'n_abstain_on_graded'):
        summary[key] = sum(r[key] for r in results)
    for who, count_key, mean_key in [('reference', 'n_gt_spells', 'gt_mean_spell_duration_s'),
                                      ('prediction', 'n_pred_spells', 'mean_spell_duration_s')]:
        count = summary[count_key]
        summary[f'{who}_mean_spell_duration_s'] = sum(r[count_key] * r[mean_key] for r in results) / count if count else 0.0
    summary['false_possession_switches'] = sum(r['diagnostics']['false_possession_switches'] for r in results)
    summary['duration_s'] = {key: sum(r['diagnostics']['duration_s'][key] for r in results)
                             for key in results[0]['diagnostics']['duration_s']}
    return summary


def evaluate_match(halves, references, *, chunk_size=4096):
    from .splits import load_manifest
    sealed = {'DFL-MAT-' + m for m in load_manifest()['assignments']['holdout']}
    if any(h.match_id in sealed for h in halves):
        raise ValueError("holdout is sealed; preregister a separate release before evaluation")
    audit = audit_match(halves, references)
    if not audit['coverage_gate_passed']:
        raise ValueError("reference coverage audit failed; review provider fields before scoring")
    from .methods import METHODS, predict_methods
    by_method = {m: [] for m in METHODS}
    for half, reference in zip(halves, references):
        predictions = predict_methods(half, chunk_size=chunk_size)
        for name, labels in predictions.items():
            by_method[name].append(score_half(half, reference, labels))
    methods = {m: dict(halves=values, summary=aggregate_halves(values)) for m, values in by_method.items()}
    comparisons = {baseline: {m: {k: values['summary'][k] - methods[baseline]['summary'][k]
                                  for k in ('spell_f1_mean', 'macro_f1')}
                              for m, values in methods.items()}
                   for baseline in ('nearest_raw', 'nearest_smooth50')}
    return dict(match_id=halves[0].match_id, audit=audit, methods=methods, delta_vs_baselines=comparisons)
