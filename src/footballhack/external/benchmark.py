"""Development-only frozen benchmark capture; no prediction/scoring rule changes.

Run in a fresh process per match for honest process-peak RSS measurements.
Stage timing observes the unchanged bundled methods; it is not isolated cost.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch

import numpy as np

from .__main__ import outside_git, run_partition
from .evaluation import _scorer, verify_frozen_sources
from . import methods

APPROVED = ('J03WN1', 'J03WOY', 'J03WR9')
_FROZEN_PREDICT_METHODS = methods.predict_methods
FROZEN_METHODS_SHA256 = '69bcfbb6f89733514265eb7e601687953447643f38070e16318354361f26f0e9'
LONG_X_S = 2.0  # Reporting cut only; never affects model or scoring.
DEAD_INTERVAL_IOU = .5  # Supplemental interval diagnostic, not spellF1.


def _peak():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
               * (1 if sys.platform == 'darwin' else 1024))


def observe_methods(half, *, chunk_size=4096):
    """Call the original method runner exactly once, wrapping calls only to time.

    The original callables receive the exact original arguments and their
    unchanged return values flow through the original runner. No references.
    """
    import footballhack.pipeline as pipeline
    stages = {}
    def timed(name, fn):
        def call(*args, **kwargs):
            start = time.perf_counter()
            value = fn(*args, **kwargs)
            item = stages.setdefault(name, dict(wall_s=0.0, calls=0))
            item['wall_s'] += time.perf_counter() - start
            item['calls'] += 1
            item['process_peak_rss_bytes_at_exit'] = _peak()
            return value
        return call
    original = _FROZEN_PREDICT_METHODS
    start = time.perf_counter()
    with patch.object(methods, 'predict_possession', timed('footballhack_pipeline', methods.predict_possession)), \
         patch.object(methods, 'extract_features', timed('nearest_shared_features', methods.extract_features)), \
         patch.object(methods, 'decode_possession', timed('nearest_temporal_decoder', methods.decode_possession)), \
         patch.object(methods, 'majority_vote', timed('majority_vote', methods.majority_vote)), \
         patch.object(pipeline, 'estimate_dead_ball', timed('shared_dead_evidence_nested_in_pipeline', pipeline.estimate_dead_ball)):
        predictions = original(half, chunk_size=chunk_size)
    elapsed = time.perf_counter() - start
    residual = max(0.0, elapsed - sum(stages[k]['wall_s'] for k in (
        'footballhack_pipeline', 'nearest_shared_features', 'nearest_temporal_decoder', 'majority_vote')))
    stages['nearest_array_preparation_and_runner_overhead'] = dict(wall_s=residual)
    return predictions, dict(total_bundled_prediction_s=elapsed, stages=stages,
        note='Stage timers overlap for nested dead evidence. RSS is cumulative process peak including loading, not isolated method memory. Residual includes nearest arrays, evidence construction and runner overhead.')


def _latency(values):
    values = np.asarray(values, dtype=float)
    return dict(mean_latency_frames=float(values.mean()) if len(values) else None,
                median_latency_frames=float(np.median(values)) if len(values) else None,
                mean_abs_latency_frames=float(np.abs(values).mean()) if len(values) else None,
                p90_abs_latency_frames=float(np.percentile(np.abs(values), 90)) if len(values) else None)


def diagnostics(half_labels):
    """Aggregate only; each half retains its own official valid runs/X policy."""
    pm = _scorer()
    cm = np.zeros((4, 4), dtype=np.int64)
    filled_cm = cm.copy()
    unknown_lengths = []
    dead = dict(n_reference=0, n_predicted=0, matched_iou50=0, false_iou50=0, missed_iou50=0,
                no_overlap_reference_intervals=0, no_overlap_predicted_intervals=0)
    transitions = {}
    for kind in ('all', 'turnover', 'stoppage_entry', 'restart'):
        for tol in (12, 25):
            transitions[(kind, tol)] = dict(n_gt=0, n_pred=0, n_matched=0, n_missed=0, n_spurious=0, latency=[])
    for reference, prediction in half_labels:
        gt, pred = pm.label_codes(reference), pm.label_codes(prediction)
        if len(gt) != len(pred):
            raise ValueError('diagnostic frame alignment mismatch')
        runs = pm.valid_runs(gt)
        filled = pm.fill_abstentions(pred, runs)
        cm += pm.confusion(gt, pred)
        filled_cm += pm.confusion(gt, filled)
        unknown_lengths.extend(b-a for a, b, k in pm.segments(pred, runs) if k == pm.X)
        gseg, pseg = pm.segments(gt, runs), pm.segments(filled, runs)
        gd, pd = [s for s in gseg if s[2] == pm.D], [s for s in pseg if s[2] == pm.D]
        tp, fp, fn = pm.match_spells(pm.spell_overlaps(gd, pd), len(gd), len(pd), DEAD_INTERVAL_IOU)
        dead['n_reference'] += len(gd); dead['n_predicted'] += len(pd)
        dead['matched_iou50'] += tp; dead['false_iou50'] += fp; dead['missed_iou50'] += fn
        dead['no_overlap_reference_intervals'] += sum(not any(a < d and c < b for c,d,_ in pd) for a,b,_ in gd)
        dead['no_overlap_predicted_intervals'] += sum(not any(a < d and c < b for c,d,_ in gd) for a,b,_ in pd)
        gt_t, pred_t = pm.transitions(gt, runs), pm.transitions(filled, runs)
        filters = dict(all=lambda t: True,
                       turnover=lambda t: {t[1], t[2]} == {pm.H, pm.A},
                       stoppage_entry=lambda t: t[2] == pm.D,
                       restart=lambda t: t[1] == pm.D and t[2] in (pm.H, pm.A))
        for kind, accept in filters.items():
            g, p = [t for t in gt_t if accept(t)], [t for t in pred_t if accept(t)]
            for tol in (12, 25):
                pairs, missed, spurious = pm.match_transitions(g, p, tol, typed='to' if kind == 'all' else 'pair')
                v = transitions[(kind, tol)]
                for name, count in (('n_gt',len(g)),('n_pred',len(p)),('n_matched',len(pairs)),('n_missed',len(missed)),('n_spurious',len(spurious))):
                    v[name] += count
                v['latency'].extend(b[0]-a[0] for a,b in pairs)
    for value in transitions.values():
        value.update(_latency(value.pop('latency')))
        value['precision'] = value['n_matched']/value['n_pred'] if value['n_pred'] else 0.0
        value['recall'] = value['n_matched']/value['n_gt'] if value['n_gt'] else 0.0
    lengths = np.array(unknown_lengths, dtype=int)
    n_valid = int(cm[:3].sum())
    n_live = int(cm[:2].sum())
    return dict(
        n_valid=n_valid, confusion_raw=cm.tolist(), confusion_temporally_filled=filled_cm.tolist(),
        team_ownership_errors_raw=int(cm[pm.H,pm.A]+cm[pm.A,pm.H]),
        team_ownership_error_rate_on_reference_live=float((cm[pm.H,pm.A]+cm[pm.A,pm.H])/n_live) if n_live else None,
        team_ownership_errors_after_official_X_fill=int(filled_cm[pm.H,pm.A]+filled_cm[pm.A,pm.H]),
        false_dead_frames_on_reference_live=int(cm[:2,pm.D].sum()),
        missed_dead_frames_assigned_to_team=int(cm[pm.D,:2].sum()),
        dead_frames_abstained=int(cm[pm.D,pm.X]),
        predicted_X=dict(n_frames=int(cm[:3,pm.X].sum()), rate=float(cm[:3,pm.X].sum()/n_valid) if n_valid else None,
                         n_runs=len(lengths), longest_s=float(lengths.max()/25) if len(lengths) else 0.0,
                         long_run_cut_s=LONG_X_S, n_runs_at_least_cut=int((lengths>=LONG_X_S*25).sum()),
                         duration_at_least_cut_s=float(lengths[lengths>=LONG_X_S*25].sum()/25)),
        dead_intervals=dict(dead, iou_threshold=DEAD_INTERVAL_IOU),
        transitions={f'{kind}_{tol}':value for (kind,tol),value in transitions.items()})


def run_match(data_dir, output_dir, match):
    if match not in APPROVED:
        raise ValueError('only the three approved development matches may be benchmarked')
    output_dir = outside_git(output_dir)
    if output_dir.exists():
        raise ValueError('benchmark output directory already exists; refusing overwrite')
    before = verify_frozen_sources()
    method_path = Path(methods.__file__)
    method_hash = sha256(method_path.read_bytes()).hexdigest()
    if method_hash != FROZEN_METHODS_SHA256:
        raise ValueError('frozen four-method implementation changed')
    audit_path = Path.home()/'FootballHACK-external-results/sportec-v1'/f'{match}-audit.json'
    audit = json.loads(audit_path.read_text())
    if audit['protocol'] != before or not audit['matches'][0]['coverage_gate_passed']:
        raise ValueError('native audit/configuration checkpoint mismatch')
    output_dir.mkdir(parents=True)
    profiles, captured = [], []
    original_score = __import__('footballhack.external.evaluation', fromlist=['score_half']).score_half
    def capture_prediction(half, *, chunk_size=4096):
        prediction, profile = observe_methods(half, chunk_size=chunk_size)
        profiles.append(dict(section=half.section, **profile))
        captured.append(dict(half=half, predictions=prediction, reference=None, score_s={}))
        return prediction
    def capture_score(half, reference, prediction):
        record = next(r for r in captured if r['half'] is half)
        name = next(name for name, labels in record['predictions'].items() if labels is prediction)
        record['reference'] = reference
        start = time.perf_counter()
        result = original_score(half, reference, prediction)
        record['score_s'][name] = time.perf_counter() - start
        return result
    with patch.object(methods, 'predict_methods', capture_prediction), \
         patch('footballhack.external.evaluation.score_half', capture_score):
        report = run_partition(outside_git(data_dir), 'development', evaluate=True, match_id=match)
    assert verify_frozen_sources() == before and sha256(method_path.read_bytes()).hexdigest() == method_hash
    result = report['matches'][0]
    for name in methods.METHODS:
        result['methods'][name]['aggregate_diagnostics'] = diagnostics([
            (r['reference'].labels, r['predictions'][name]) for r in captured])
        result['methods'][name]['official_scoring_wall_s'] = sum(r['score_s'][name] for r in captured)
    for record in captured:
        half, reference = record['half'], record['reference']
        np.savez_compressed(output_dir/(half.section+'.npz'), frame=half.tracking.frame,
                            native_frame=half.native_frame, reference=reference.labels, **record['predictions'])
    result['prediction_profiles'] = profiles
    report['capture_provenance'] = dict(methods_sha256=method_hash,
        runner_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        diagnostics=dict(long_X_s=LONG_X_S, dead_interval_iou=DEAD_INTERVAL_IOU),
        report_only=True, predictions_and_scorer_unchanged=True,
        audit_sha256=sha256(audit_path.read_bytes()).hexdigest())
    (output_dir/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(f'{match}: frozen four-method benchmark complete; report and arrays saved locally.',flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', required=True, choices=APPROVED)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    run_match(args.data_dir,args.output_dir,args.match)


if __name__=='__main__':
    main()
