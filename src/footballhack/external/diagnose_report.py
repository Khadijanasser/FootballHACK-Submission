"""Pool aggregate diagnostics; no tracking, video or prediction modification."""
from collections import Counter
import json
from pathlib import Path


def aggregate(report):
    halves=[h for match in report['matches'].values() for h in match]
    def counter_at(*keys):
        result=Counter()
        for h in halves:
            value=h
            for key in keys:value=value[key]
            result.update(value)
        return dict(result)
    def sums_at(*keys):
        first=halves[0]
        for key in keys:first=first[key]
        return {k:sum(_at(h,(*keys,k)) for h in halves) for k in first}
    x={k:sum(h['X'][k] for h in halves) for k in (
        'n_frames','restart_required_frames','restart_required_and_motion_qualified_frames',
        'no_restart_required_live_without_owner_frames','no_restart_required_unknown_phase_frames',
        'missing_control_frames','strong_candidate_frames')}
    for key in ('reasons','phases','control_status','phase_status'):x[key]=counter_at('X',key)
    x['long_runs']={k:sum(h['X']['long_runs'][k] for h in halves) for k in ('n','n_frames','restart_required_frames')}
    x['long_runs']['longest_s']=max(h['X']['long_runs']['longest_s'] for h in halves)
    for key in ('start_reasons','end_reasons','control_status','phase_status'):
        x['long_runs'][key]=counter_at('X','long_runs',key)
    duration=sums_at('duration')
    # Means must be recomputed from summed counts, never summed or averaged.
    duration['raw_mean_team_spell_s']=duration['raw_team_frames']/duration['raw_team_spells']/25
    duration['filled_mean_team_spell_s']=duration['filled_team_frames']/duration['filled_team_spells']/25
    duration['reference_mean_team_spell_s']=duration['reference_team_frames']/duration['reference_spells']/25
    restarts={k:sum(h['restarts'][k] for h in halves) for k in ('n_gt','n_matched','n_missed','n_spurious')}
    for name in ('all','missed'):
        restarts[name]={k:counter_at('restarts',name,k) for k in ('counts','phase','control_status','output_labels')}
    return dict(n_frames=sum(h['n_frames'] for h in halves), X=x,
        dead_evidence=sums_at('dead_evidence'), duration=duration,
        restarts=restarts, correct_control_runs=sums_at('correct_control_runs'),
        prediction_only_evidence=sums_at('prediction_only_evidence'),
        hidden_boundaries={key:sums_at('hidden_boundaries',key) for key in ('turnover','dead_entry','restart')},
        by_reference={key:dict(n_frames=sum(h['by_reference'][key]['n_frames'] for h in halves),
             output_labels=counter_at('by_reference',key,'output_labels'),
             control_status=counter_at('by_reference',key,'control_status'),
             phase_status=counter_at('by_reference',key,'phase_status'),
             phase=counter_at('by_reference',key,'phase')) for key in 'HAD'},
        proof=dict(original_artifacts_unchanged=report['original_artifacts_unchanged'],
                   reproduced_labels_identical=report['reproduced_labels_identical']),
        note='Reference conditioning is evaluation-only. Boundary snapshots are descriptive, not causal attribution; X filling, missed D and missed turnovers overlap. Half-specific delay percentiles remain in diagnostics.json and are not averaged.')


def _at(value,keys):
    for key in keys:value=value[key]
    return value


def render(a):
    x,d,t,r=a['X'],a['dead_evidence'],a['duration'],a['restarts']
    missed=r['missed']['counts']; control=a['correct_control_runs']
    def listing(mapping):return '\n'.join(f'- {key}: {value}' for key,value in mapping.items())+'\n\n'
    return ('# Frozen development diagnostic findings\n\n'
        'Only J03WN1/J03WOY/J03WR9. All frozen labels reproduce exactly and original benchmark artifacts remain unchanged. '
        'References are opened only after inference. No new model, score, tag, parameter choice or dropout experiment is run.\n\n'
        '## X state and onset\n\n'+listing({k:x[k] for k in ('n_frames','restart_required_frames','missing_control_frames','strong_candidate_frames')})+
        'Current control status:\n\n'+listing(x['control_status'])+
        'Current phase:\n\n'+listing(x['phases'])+
        'Emitted X reasons (events may override persistent reasons):\n\n'+listing(x['reasons'])+
        'Long X run onset reasons:\n\n'+listing(x['long_runs']['start_reasons'])+
        'Long X runs:\n\n'+listing({k:v for k,v in x['long_runs'].items() if not isinstance(v,dict)})+
        f"X frames with a restart-required latch: {x['restart_required_frames']}. The latch is reconstructed from exported phase/reset and verified against "
        f"the unchanged decoder local state in synthetic tracing tests. Recorded dead-support-expiry events on reference D: {d['support_expiry_events_on_reference_dead']}. "
        'Interpret latch counts and long-run onset reasons together before attributing X to timeout/restart locks. '
        'Missing observations and conflicts are distinct from weak observed proximity support.\n\n'
        '## Dead-ball evidence\n\n'+listing(d)+
        'With no Smart Tags, DEAD evidence is available only when ball clearance is below -0.5 m. '
        'Stationary balls alone intentionally provide no DEAD evidence. Of missed reference-D frames, '
        f"{d['missed_inside_pitch_frames']} lie inside the buffered pitch, {d['missed_boundary_band_frames']} in the boundary band, and {d['missed_but_raw_dead_qualified_frames']} have qualifying raw DEAD evidence "
        f"but are not yet labelled D. Missing-ball misses: {d['missed_missing_ball_frames']}; misses with LIVE evidence: {d['missed_with_live_evidence_frames']}. "
        'These are availability limits of the frozen evidence rules, not probabilities.\n\n'
        '## Restarts\n\n'+listing({k:v for k,v in r.items() if not isinstance(v,dict)})+
        'Missed restart counts:\n\n'+listing(r['missed']['counts'])+
        'State at the reference restart among misses:\n\n'+listing(r['missed']['phase'])+
        f"For {missed.get('no_raw_D_in_preceding_reference_D_interval',0)} of {r['n_missed']} misses no raw D appears in the preceding reference stoppage. "
        f"Among misses, {r['missed']['phase'].get('UNKNOWN',0)} have UNKNOWN phase, {r['missed']['phase'].get('DEAD',0)} remain DEAD, and {missed.get('already_correct_raw_team_before_reference_restart',0)} already label the correct "
        'team before the reference restart. A missing D means no typed D-to-team transition can be recovered merely by '
        f"shortening team acquisition. Correct control is already qualified at {missed.get('correct_candidate_at_reference_restart',0)} missed restart instants, so control absence "
        'alone is not a sufficient explanation; continuous LIVE confirmation and prior D detection matter. '
        'These are state snapshots, not exclusive counterfactual causes.\n\n'
        '## Possession durations and official X filling\n\n'+listing(t)+
        'Raw H/A runs break at X and are descriptive only, not an alternative ranking score or modified prediction. '
        f"Official filling adds {t['X_filled_to_team_frames']} team-labelled frames, {t['X_filled_to_team_on_reference_dead']} of which are reference D. "
        f"Raw H/A mean duration is {t['raw_mean_team_spell_s']:.2f} s versus {t['reference_mean_team_spell_s']:.2f} s in the reference; official filling produces {t['filled_mean_team_spell_s']:.2f} s. "
        'Missing D evidence creates unknown stoppages that the official rule fills with preceding ownership. '
        f"Missed turnovers also merge spells: {t['filled_spells_overlapping_multiple_reference_possessions']} filled spells overlap multiple reference possessions and {t['filled_spells_completely_bridging_reference_D_intervals']} fully enclose "
        'a reference D interval. These mechanisms overlap and must not be interpreted as additive causal percentages.\n\n'
        'Hidden exact reference boundaries:\n\n'+listing(a['hidden_boundaries'])+
        'Correct-candidate run availability inside reference possessions:\n\n'+listing(a['correct_control_runs'])+
        f"{control['no_correct_candidate']+control['correct_candidate_but_no_acquisition_length_run']} of {control['reference_possessions']} reference possessions have no correct acquisition-length run ({control['no_correct_candidate']} never have a correct candidate; "
        f"{control['correct_candidate_but_no_acquisition_length_run']} have only shorter runs). At 25 fps acquisition needs four samples and turnover confirmation six samples. "
        'Shortening temporal confirmation cannot create support in zero-candidate cases.\n\n'
        '## Interpretation limits\n\n'
        'No Smart Tags were available, so set-piece corroboration, automatic live-phase support and tag-assisted stationary '
        'restarts were all inactive. Adding tags is not proven to solve the problem: an independent, licensed tag source and '
        'a separately approved experiment are required. Never derive tags from BallStatus/BallPossession. '
        'Native ball XY coverage is complete, so tracking-dropout robustness remains unmeasured. '
        'The scorer, causal mode, thresholds and all stored predictions remain frozen.\n')


def main():
    import argparse
    from .__main__ import outside_git
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory');args=parser.parse_args()
    root=outside_git(args.directory)
    if (root/'summary.json').exists() or (root/'report.md').exists():raise ValueError('refusing to overwrite diagnostic reports')
    a=aggregate(json.loads((root/'diagnostics.json').read_text()))
    (root/'summary.json').write_text(json.dumps(a,indent=2,sort_keys=True,allow_nan=False)+'\n')
    (root/'report.md').write_text(render(a))


if __name__=='__main__':main()
