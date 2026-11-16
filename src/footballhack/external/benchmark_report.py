"""Aggregate saved approved development results; never reads tracking or videos."""
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .benchmark import APPROVED
from .evaluation import _scorer
from .methods import METHODS


def build_report(root):
    root = Path(root)
    reports = {m: json.loads((root/m/'report.json').read_text()) for m in APPROVED}
    result = {m: r['matches'][0] for m, r in reports.items()}
    assert all(r['evaluated'] and r['partition'] == 'development' for r in reports.values())
    assert all(r['protocol'] == reports[APPROVED[0]]['protocol'] for r in reports.values())
    assert all(result[m]['match_id'] == 'DFL-MAT-'+m for m in APPROVED)
    pm = _scorer()
    averages = {}
    for method in METHODS:
        values = [r['methods'][method] for r in result.values()]
        cm = sum((np.array(v['summary']['confusion_pooled'], dtype=np.int64) for v in values), np.zeros((4,4), dtype=np.int64))
        filled_cm = sum((np.array(v['aggregate_diagnostics']['confusion_temporally_filled'], dtype=np.int64) for v in values), np.zeros((4,4), dtype=np.int64))
        avg = {metric: float(np.mean([v['summary'][metric] for v in values]))
               for metric in ('spell_f1_mean', 'macro_f1')}
        avg.update(confusion_pooled=cm.tolist(), confusion_temporally_filled_pooled=filled_cm.tolist(),
                   per_class_pooled={label: dict(zip(('precision','recall','f1'), pm.per_class_prf(pm.scored_confusion(cm),k)))
                                     for k,label in enumerate('HAD')})
        for key in ('n_frames','n_graded','n_gt_spells','n_pred_spells','n_abstain_on_graded'):
            avg[key] = sum(v['summary'][key] for v in values)
        for who, number in (('prediction','n_pred_spells'), ('reference','n_gt_spells')):
            key = who+'_mean_spell_duration_s'
            avg[key] = sum(v['summary'][key] * v['summary'][number] for v in values)/avg[number] if avg[number] else 0.0
        avg['predicted_X_rate_pooled'] = avg['n_abstain_on_graded']/avg['n_graded']
        avg['long_X_runs_at_least_2s'] = sum(v['aggregate_diagnostics']['predicted_X']['n_runs_at_least_cut'] for v in values)
        avg['longest_X_s'] = max(v['aggregate_diagnostics']['predicted_X']['longest_s'] for v in values)
        avg['long_X_duration_s'] = sum(v['aggregate_diagnostics']['predicted_X']['duration_at_least_cut_s'] for v in values)
        avg['dead_intervals'] = {k: sum(v['aggregate_diagnostics']['dead_intervals'][k] for v in values)
                                 for k in ('n_reference','n_predicted','matched_iou50','false_iou50','missed_iou50','no_overlap_reference_intervals','no_overlap_predicted_intervals')}
        avg['transitions'] = {}
        for name in values[0]['aggregate_diagnostics']['transitions']:
            events = [v['aggregate_diagnostics']['transitions'][name] for v in values]
            totals = {key: sum(e[key] for e in events) for key in ('n_gt','n_pred','n_matched','n_missed','n_spurious')}
            totals['precision'] = totals['n_matched']/totals['n_pred'] if totals['n_pred'] else 0.0
            totals['recall'] = totals['n_matched']/totals['n_gt'] if totals['n_gt'] else 0.0
            for key in ('mean_latency_frames','mean_abs_latency_frames'):
                totals[key] = sum(e[key]*e['n_matched'] for e in events if e['n_matched'])/totals['n_matched'] if totals['n_matched'] else None
            avg['transitions'][name] = totals
        averages[method] = avg
    paired = {}
    for a,b in combinations(METHODS,2):
        per_match = {m: {key: result[m]['methods'][b]['summary'][key] - result[m]['methods'][a]['summary'][key]
                         for key in ('spell_f1_mean','macro_f1')} for m in APPROVED}
        paired[f'{b} minus {a}'] = dict(per_match=per_match,
            equal_match_mean={key: float(np.mean([v[key] for v in per_match.values()])) for key in ('spell_f1_mean','macro_f1')},
            positive_match_counts={key: sum(v[key]>0 for v in per_match.values()) for key in ('spell_f1_mean','macro_f1')})
    summary = dict(approved_matches=list(APPROVED), equal_match_average=averages,
                   paired_differences=paired, per_match=result,
                   aggregation='official scores per half; match = equal-half mean; overall = equal-match mean. Pooled class metrics/counts/durations/X rates explicitly separate.',
                   protocol=reports[APPROVED[0]]['protocol'],
                   resource_caveat='Shared bundled computation. Compute-path times include shared preparations; RSS is full process peak including loading, not isolated per-method memory. Core runtime excludes supplemental report processing and writing.',
                   diagnostic_cuts=dict(long_X_s=2.0, dead_interval_iou=.5),
                   limitations=['No missing ball XY; missing-ball robustness is unmeasured.',
                                'Provider control/referee-state reference definitions are not proven equivalent to Metrica continuity/boundary conventions.',
                                'Three development games, unavoidable club overlap; no validation/holdout inference.',
                                'Centered smooth50 uses future frames; FootballHACK and nearest-temporal are causal.',
                                'X penalized raw at frame level; official temporal metrics fill it within each half.'])
    return summary


def _table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])+'\n\n'


def render_report(summary):
    out = ['# Frozen Sportec development benchmark\n\n',
           'Only J03WN1, J03WOY and J03WR9. Exact official scorer; unchanged model and method configuration. '
           'All six halves scored independently, at verified 25 fps with 100% reference coverage. '
           'No tuning, alternative modes, video inspection, validation or holdout inference.\n\n',
           summary['aggregation']+'\n\n', '## Equal-match comparison\n\n']
    out.append(_table(['Method','spellF1','macroF1','Pred/reference possessions','Mean pred/reference duration (s)','Pooled X %'], [
        [name,f"{a['spell_f1_mean']:.6f}",f"{a['macro_f1']:.6f}",f"{a['n_pred_spells']}/{a['n_gt_spells']}",
         f"{a['prediction_mean_spell_duration_s']:.3f}/{a['reference_mean_spell_duration_s']:.3f}",f"{a['predicted_X_rate_pooled']*100:.3f}"]
        for name,a in summary['equal_match_average'].items()]))
    out.append('## Paired differences\n\nPositive means the named method improved over its comparator. No significance claims from three games.\n\n')
    out.append(_table(['Pair','WN1 Δspell / Δmacro','WOY Δspell / Δmacro','WR9 Δspell / Δmacro','Equal-match Δspell / Δmacro'], [
        [name]+[f"{v['per_match'][m]['spell_f1_mean']:+.6f} / {v['per_match'][m]['macro_f1']:+.6f}" for m in APPROVED]+
        [f"{v['equal_match_mean']['spell_f1_mean']:+.6f} / {v['equal_match_mean']['macro_f1']:+.6f}"]
        for name,v in summary['paired_differences'].items()]))
    for match,record in summary['per_match'].items():
        out += [f'## {match}\n\n']
        out.append(_table(['Method','spellF1','macroF1','Pred/ref possessions','Mean pred/ref duration (s)','X %'],[
            [name,f"{v['summary']['spell_f1_mean']:.6f}",f"{v['summary']['macro_f1']:.6f}",
             f"{v['summary']['n_pred_spells']}/{v['summary']['n_gt_spells']}",
             f"{v['summary']['prediction_mean_spell_duration_s']:.3f}/{v['summary']['reference_mean_spell_duration_s']:.3f}",
             f"{v['aggregate_diagnostics']['predicted_X']['rate']*100:.3f}"] for name,v in record['methods'].items()]))
        out += ['Class metrics pool confusion counts over the two halves; they are not a macroF1 reaggregation.\n\n']
        out.append(_table(['Method','Class','Precision','Recall','F1'],[[name,k]+[f'{cls[x]:.6f}' for x in ('precision','recall','f1')]
            for name,v in record['methods'].items() for k,cls in v['summary']['per_class_pooled'].items()]))
        out += ['Boundary diagnostics use the official helpers independently within each half. '
                'All transitions match destination; direct turnovers/stoppage entries/restarts require the same from/to pair. '
                'Latency statistics include matched events only; positive latency means late. 25 frames = 1 second.\n\n']
        out.append(_table(['Method','Event/tolerance frames','Reference','Predicted','Matched','Missed','Spurious','P / R','Mean signed / absolute lag (frames)','P90 absolute lag'],[
            [name,kind,d['n_gt'],d['n_pred'],d['n_matched'],d['n_missed'],d['n_spurious'],f"{d['precision']:.4f} / {d['recall']:.4f}",
             f"{d['mean_latency_frames']:.3f} / {d['mean_abs_latency_frames']:.3f}" if d['mean_latency_frames'] is not None else 'N/A',
             f"{d['p90_abs_latency_frames']:.3f}" if d['p90_abs_latency_frames'] is not None else 'N/A']
            for name,v in record['methods'].items() for kind,d in v['aggregate_diagnostics']['transitions'].items()]))
        out += ['Dead interval matching is a supplemental diagnostic at IoU >=0.5 using official overlap/matching helpers. '
                'Unmatched intervals can reflect boundary mismatches rather than wholly false events; zero-overlap counts distinguish this. '
                'X is filled only by the official temporal policy for interval diagnostics.\n\n']
        out.append(_table(['Method','D intervals ref / pred','Matched / unmatched pred / missed ref','No-overlap pred / ref','False D live frames','Missed D assigned-team / X frames'],[
            [name,f"{d['dead_intervals']['n_reference']} / {d['dead_intervals']['n_predicted']}",
             f"{d['dead_intervals']['matched_iou50']} / {d['dead_intervals']['false_iou50']} / {d['dead_intervals']['missed_iou50']}",
             f"{d['dead_intervals']['no_overlap_predicted_intervals']} / {d['dead_intervals']['no_overlap_reference_intervals']}",
             d['false_dead_frames_on_reference_live'],f"{d['missed_dead_frames_assigned_to_team']} / {d['dead_frames_abstained']}"]
            for name,v in record['methods'].items() for d in [v['aggregate_diagnostics']]]))
        out.append(_table(['Method','Wrong-team live frames raw / filled','Wrong-team / reference-live %','X frames / runs','X >=2s runs / duration(s)','Longest X(s)'],[
            [name,f"{d['team_ownership_errors_raw']} / {d['team_ownership_errors_after_official_X_fill']}",
             f"{d['team_ownership_error_rate_on_reference_live']*100:.3f}",f"{d['predicted_X']['n_frames']} / {d['predicted_X']['n_runs']}",
             f"{d['predicted_X']['n_runs_at_least_cut']} / {d['predicted_X']['duration_at_least_cut_s']:.2f}",f"{d['predicted_X']['longest_s']:.2f}"]
            for name,v in record['methods'].items() for d in [v['aggregate_diagnostics']]]))
        resources = record['resources']
        out += [f"Core benchmark wall time: {resources['elapsed_s']:.3f} s. Process peak RSS: {resources['process_peak_rss_bytes']/2**20:.2f} MiB.\n\n",
                summary['resource_caveat']+'\n\n']
        stage = {key: sum(h['stages'][key]['wall_s'] for h in record['prediction_profiles'])
                 for key in record['prediction_profiles'][0]['stages']}
        preparation = stage['nearest_shared_features'] + stage['nearest_array_preparation_and_runner_overhead']
        paths = dict(nearest_raw=preparation, nearest_smooth50=preparation+stage['majority_vote'],
                     nearest_temporal=preparation+stage['nearest_temporal_decoder']+stage['shared_dead_evidence_nested_in_pipeline'],
                     footballhack_causal=stage['footballhack_pipeline'])
        out.append(_table(['Method','Observed shared compute-path time(s)','Official scoring time(s)','Shared process peak(MiB)'],[
            [name,f'{paths[name]:.6f}',f"{record['methods'][name]['official_scoring_wall_s']:.6f}",f"{resources['process_peak_rss_bytes']/2**20:.2f}"] for name in METHODS]))
        out += ['Shared preparation includes nearest arrays/evidence and runner overhead. Dead-evidence timing is nested inside FootballHACK, '
                'and is reused by nearest-temporal; do not add all method-path times as an independent runtime total.\n\n']
    out += ['## Limitations and interpretation\n\n']+[f'- {text}\n' for text in summary['limitations']]
    out += ['\nSportec defines possession via the controlling player and BallStatus via referee-interrupted play. '
            'Metrica evaluates maximal H/A spells and fills prediction X within each half for temporal metrics, while charging raw X in frame metrics. '
            'Exact equivalence of pass/duel continuity and restart/stoppage boundaries is not established; scores here measure agreement with Sportec references. '
            'No comparison to a competition-match baseline is made.\n\n',
            'Sources: [Sportec provider paper](https://doi.org/10.1038/s41597-025-04505-y), '
            '[Metrica evaluation specification](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/EVALUATION.md).\n\n',
            '## Reproducibility\n\n'
            'Figshare v1 DOI 10.6084/m9.figshare.28196177.v1; CC BY 4.0; credit Bassek, Rein, Weber & Memmert (2025), DFL / Sportec Solutions. '
            'Inputs are unchanged canonical files verified against provider size/MD5. `run-manifest.json` records source hashes, git base, configuration, '
            'and `frozen-source.zip` checksum. Each match report records source/configuration/audit hashes, Python/NumPy/pandas versions and half metadata. '
            'Per-half label archives are retained locally and were not inspected to choose settings. '
            'The supplemental summaries use only saved aggregate JSON; no prediction arrays are modified.\n']
    return ''.join(out)


def main():
    import argparse
    from .__main__ import outside_git
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    args = parser.parse_args()
    root = outside_git(args.root)
    for name in ('summary.json','report.md'):
        if (root/name).exists():
            raise ValueError('refusing to overwrite benchmark report')
    summary = build_report(root)
    (root/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
    (root/'report.md').write_text(render_report(summary))
    print('Aggregate JSON and complete quantitative Markdown report saved locally.')


if __name__=='__main__':
    main()
