"""Read-only aggregate explanations of frozen development predictions.

No alternative model settings, tags, scoring formula or prediction edits.
Recomputed labels must equal the saved benchmark labels before references are
used for analysis. Original benchmark artifact hashes are checked before/after.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from math import ceil
from pathlib import Path

import numpy as np

from footballhack.io.sportec import load_sportec_metadata, load_sportec_tracking
from footballhack.pipeline import PipelineConfig, predict_possession
from .__main__ import outside_git, verify_artifact
from .benchmark import APPROVED, FROZEN_METHODS_SHA256
from .evaluation import _scorer, verify_frozen_sources
from .splits import partition_matches


def _counts(values):
    return dict(sorted(Counter(str(v) for v in values).items()))


def _runs(mask):
    edges = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _longest(mask):
    return max((b-a for a,b in _runs(mask)), default=0)


def _delays(values, fps):
    a = np.asarray(values, dtype=float) / fps
    return dict(n=len(a), median_s=float(np.median(a)) if len(a) else None,
                p90_s=float(np.percentile(a,90)) if len(a) else None,
                maximum_s=float(a.max()) if len(a) else None)


def restart_required_mask(possession):
    """Reconstruct the latch from exported phase/reset, without new decoding.

    DEAD sets it, LIVE clears it, UNKNOWN preserves it. A support timeout does
    not clear a previously confirmed stoppage; explicit resets do clear it.
    Synthetic tracing verifies equality with the unchanged decoder's local latch.
    """
    latch, result = False, np.zeros(len(possession.labels), dtype=bool)
    for i,phase in enumerate(possession.phase):
        if possession.reset[i]:
            latch = False
        if phase == 'DEAD':
            latch = True
        elif phase == 'LIVE':
            latch = False
        result[i] = latch
    return result


def summarize_half(result, reference, config):
    """Evaluation-only conditioning, after prediction has finished unchanged."""
    pm = _scorer()
    labels = result.labels
    gt = pm.label_codes(reference)
    pred = pm.label_codes(labels)
    if len(gt) != len(labels):
        raise ValueError('diagnostic/reference alignment mismatch')
    fps, cfg = result.fps, config.temporal
    p, c, d = result.possession, result.control, result.dead_ball
    valid = gt != pm.X
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred,runs)
    latch = restart_required_mask(p)
    x = (labels == 'X') & valid
    missing_control = ~(c.ball_available & c.team_available.all(axis=1))
    live = p.phase_status == 'LIVE'
    dead = p.phase_status == 'DEAD'
    motion_restart = live & d.motion_available & (d.scores[:,0] >= cfg.restart_live_threshold)
    candidates = np.isin(p.control_status, ('H','A'))
    long_runs = [(a,b) for start,stop in runs for a,b in _runs(x[start:stop])
                 for a,b in [(a+start,b+start)] if (b-a)/fps >= 2]
    long_mask = np.zeros(len(x),dtype=bool)
    for a,b in long_runs:
        long_mask[a:b] = True
    long_start = Counter(p.reason[a] for a,b in long_runs)
    long_end = Counter(p.reason[b] if b<len(labels) else 'half_end' for a,b in long_runs)
    x_diagnostics = dict(n_frames=int(x.sum()), reasons=_counts(p.reason[x]), phases=_counts(p.phase[x]),
        control_status=_counts(p.control_status[x]), phase_status=_counts(p.phase_status[x]),
        restart_required_frames=int((x&latch).sum()),
        restart_required_and_motion_qualified_frames=int((x&latch&motion_restart).sum()),
        no_restart_required_live_without_owner_frames=int((x&~latch&(p.phase=='LIVE')).sum()),
        no_restart_required_unknown_phase_frames=int((x&~latch&(p.phase=='UNKNOWN')).sum()),
        missing_control_frames=int((x&missing_control).sum()),
        strong_candidate_frames=int((x&candidates).sum()),
        long_runs=dict(n=len(long_runs), n_frames=int(long_mask.sum()),
            longest_s=max(((b-a)/fps for a,b in long_runs),default=0),
            start_reasons=dict(long_start), end_reasons=dict(long_end),
            phase_status=_counts(p.phase_status[long_mask]), control_status=_counts(p.control_status[long_mask]),
            restart_required_frames=int((long_mask&latch).sum())))
    by_reference = {}
    for k,code in enumerate('HAD'):
        mask = gt == k
        by_reference[code] = dict(n_frames=int(mask.sum()), output_labels=_counts(labels[mask]),
            control_status=_counts(p.control_status[mask]), phase_status=_counts(p.phase_status[mask]),
            phase=_counts(p.phase[mask]),
            X_restart_required=int((mask&x&latch).sum()),
            X_missing_control=int((mask&x&missing_control).sum()))
    ref_dead = gt == pm.D
    missing_dead = ref_dead & (pred != pm.D)
    clearance = d.boundary_clearance_m
    dead_diagnostics = dict(n_reference_dead_frames=int(ref_dead.sum()),
        raw_dead_qualified_frames=int((ref_dead&dead).sum()), final_dead_frames=int((ref_dead&(pred==pm.D)).sum()),
        missed_frames=int(missing_dead.sum()),
        missed_but_raw_dead_qualified_frames=int((missing_dead&dead).sum()),
        missed_inside_pitch_frames=int((missing_dead&(clearance>config.dead_ball.boundary_buffer_m)).sum()),
        missed_boundary_band_frames=int((missing_dead&(np.abs(clearance)<=config.dead_ball.boundary_buffer_m)).sum()),
        missed_missing_ball_frames=int((missing_dead&~d.ball_available).sum()),
        missed_with_live_evidence_frames=int((missing_dead&live).sum()),
        support_expiry_events_on_reference_dead=int(np.count_nonzero(ref_dead&(p.reason=='dead_support_expired'))),
        tag_available_frames=int(d.tag_available.sum()),
        set_piece_prior_frames=int(np.count_nonzero(d.set_piece_prior)),
        live_phase_prior_frames=int(np.count_nonzero(d.live_phase_prior)),
        restart_prior_frames=int(np.count_nonzero(d.restart_prior)))

    gseg, pseg = pm.segments(gt,runs), pm.segments(pred,runs)
    fseg = pm.segments(filled,runs)
    gs, raw_spells, fs = (pm.possession_spells(a,runs) for a in (gt,pred,filled))
    raw_team = np.isin(pred,(pm.H,pm.A)) & valid
    filled_team = np.isin(filled,(pm.H,pm.A)) & valid
    duration = dict(reference_spells=len(gs), raw_team_spells=len(raw_spells), filled_team_spells=len(fs),
        reference_team_frames=sum(b-a for a,b,k in gs), raw_team_frames=int(raw_team.sum()),
        filled_team_frames=int(filled_team.sum()),
        raw_mean_team_spell_s=float(sum(b-a for a,b,k in raw_spells)/len(raw_spells)/fps) if raw_spells else 0.,
        filled_mean_team_spell_s=float(sum(b-a for a,b,k in fs)/len(fs)/fps) if fs else 0.,
        X_filled_to_team_frames=int((x&filled_team).sum()),
        X_filled_to_team_on_reference_dead=int((x&filled_team&ref_dead).sum()),
        X_filled_to_team_on_reference_live=int((x&filled_team&np.isin(gt,(pm.H,pm.A))).sum()),
        filled_team_frames_on_reference_dead=int((filled_team&ref_dead).sum()),
        filled_spells_overlapping_multiple_reference_possessions=sum(
            sum(a<d_ and c_<b for c_,d_,k_ in gs)>1 for a,b,k in fs),
        filled_spells_completely_bridging_reference_D_intervals=sum(
            any(a<c_ and d_<b for c_,d_,k_ in gseg if k_==pm.D) for a,b,k in fs))
    hidden = {}
    for name, condition in dict(turnover=lambda a,b:{a,b}=={pm.H,pm.A},
                                 dead_entry=lambda a,b:b==pm.D,
                                 restart=lambda a,b:a==pm.D and b in (pm.H,pm.A)).items():
        transitions=[t for t in pm.transitions(gt,runs) if condition(t[1],t[2])]
        carry = [t for t,a,b in transitions if filled[t-1]==filled[t] and filled[t] in (pm.H,pm.A)]
        hidden[name]=dict(n_reference=len(transitions), unchanged_filled_team_at_exact_boundary=len(carry),
                         unchanged_team_with_raw_X_at_boundary=int(sum(pred[t-1]==pm.X or pred[t]==pm.X for t in carry)))

    gt_restart=[t for t in pm.transitions(gt,runs) if t[1]==pm.D and t[2] in (pm.H,pm.A)]
    pr_restart=[t for t in pm.transitions(filled,runs) if t[1]==pm.D and t[2] in (pm.H,pm.A)]
    pairs,missed,spurious=pm.match_transitions(gt_restart,pr_restart,25,typed='pair')
    gt_by_start={a:(a,b,k) for a,b,k in gseg}
    prior_dead_by_end={b:(a,b,k) for a,b,k in gseg if k==pm.D}
    restart = dict(n_gt=len(gt_restart), n_matched=len(pairs), n_missed=len(missed), n_spurious=len(spurious))
    for label, events in (('all',gt_restart),('missed',missed)):
        counts=Counter();phase_counts=Counter();control_counts=Counter();output_counts=Counter()
        output_delay=[];candidate_delay=[];motion_delay=[]
        for t,_,team in events:
            a,b,_=prior_dead_by_end[t]
            stop=gt_by_start[t][1]
            counts['no_raw_D_in_preceding_reference_D_interval']+=not np.any(pred[a:b]==pm.D)
            counts['no_filled_D_in_preceding_reference_D_interval']+=not np.any(filled[a:b]==pm.D)
            counts['restart_required_at_reference_restart']+=bool(latch[t])
            counts['motion_restart_qualified_at_reference_restart']+=bool(motion_restart[t])
            counts['correct_candidate_at_reference_restart']+=p.control_status[t]=='HA'[team]
            counts['already_correct_raw_team_before_reference_restart']+=pred[t-1]==team and pred[t]==team
            phase_counts[p.phase[t]]+=1;control_counts[p.control_status[t]]+=1;output_counts[labels[t]]+=1
            for name,mask,delays in (
                ('correct_output',pred[t:stop]==team,output_delay),
                ('correct_candidate',p.control_status[t:stop]=='HA'[team],candidate_delay),
                ('motion_restart',motion_restart[t:stop],motion_delay)):
                seen=np.flatnonzero(mask)
                if len(seen):delays.append(int(seen[0]))
                else:counts['no_'+name+'_within_current_reference_possession']+=1
        restart[label]=dict(counts= {k:int(v) for k,v in counts.items()}, phase=dict(phase_counts),
            control_status=dict(control_counts), output_labels=dict(output_counts),
            first_correct_output_delay=_delays(output_delay,fps),
            first_correct_candidate_delay=_delays(candidate_delay,fps),
            first_motion_restart_qualification_delay=_delays(motion_delay,fps))
    support = Counter()
    for a,b,team in gs:
        correct=p.control_status[a:b]=='HA'[team]
        longest=_longest(correct)
        support['reference_possessions']+=1
        support['no_correct_candidate']+=longest==0
        support['correct_candidate_but_no_acquisition_length_run']+=0<longest<ceil(cfg.acquisition_s*fps)+1
        support['has_acquisition_length_run']+=longest>=ceil(cfg.acquisition_s*fps)+1
        support['has_turnover_length_run']+=longest>=ceil(cfg.turnover_s*fps)+1
        support['never_correct_candidate_in_live_phase']+=not np.any(correct&(p.phase[a:b]=='LIVE'))
    return dict(n_frames=len(labels), fps=fps, X=x_diagnostics, by_reference=by_reference,
                dead_evidence=dead_diagnostics, restarts=restart, duration=duration,
                hidden_boundaries=hidden, correct_control_runs={k:int(v) for k,v in support.items()},
                prediction_only_evidence=dict(missing_control_frames=int(missing_control.sum()),
                    control_conflict_frames=int(np.count_nonzero(p.control_status=='conflict')),
                    phase_conflict_frames=int(np.count_nonzero(p.phase_status=='conflict'))))


def run(data_dir, benchmark_dir, output_dir):
    data_dir, benchmark_dir, output_dir = map(outside_git,(data_dir,benchmark_dir,output_dir))
    if output_dir.exists():raise ValueError('diagnostic output already exists; refusing overwrite')
    original=json.loads((benchmark_dir/'analysis-provenance.json').read_text())['artifact_sha256']
    for relative,digest in original.items():
        if sha256((benchmark_dir/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError('original benchmark artifact changed')
    frozen=verify_frozen_sources()
    if sha256(Path(__file__).with_name('methods.py').read_bytes()).hexdigest()!=FROZEN_METHODS_SHA256:
        raise ValueError('frozen method implementation changed')
    output_dir.mkdir(parents=True)
    matches={}
    for record in partition_matches('development'):
        match=record['match_id']
        if match not in APPROVED:raise ValueError('unapproved development match')
        for kind in ('metadata','positions'):verify_artifact(data_dir/record[kind]['name'],record[kind])
        meta=load_sportec_metadata(data_dir/record['metadata']['name'])
        halves=load_sportec_tracking(data_dir/record['positions']['name'],meta)
        summaries=[]
        for half in halves:
            cfg=PipelineConfig(n_frames=len(half.tracking.frame))
            # Prediction receives positions only. References are opened afterward.
            result=predict_possession(half.tracking,config=cfg)
            with np.load(benchmark_dir/match/(half.section+'.npz'),allow_pickle=False) as saved:
                if (not np.array_equal(result.labels,saved['footballhack_causal'])
                        or not np.array_equal(result.frame,saved['frame'])
                        or not np.array_equal(half.native_frame,saved['native_frame'])):
                    raise ValueError('frozen label/frame reproduction mismatch; stop diagnostics')
                reference=saved['reference'].copy()
            summaries.append(dict(section=half.section, **summarize_half(result,reference,cfg)))
        matches[match]=summaries
        print(match+': labels reproduced exactly; aggregate diagnostics complete.',flush=True)
    assert verify_frozen_sources()==frozen
    for relative,digest in original.items():
        assert sha256((benchmark_dir/relative).read_bytes()).hexdigest()==digest
    report=dict(matches=matches, frozen_protocol=frozen,
        original_artifacts_unchanged=True, reproduced_labels_identical=True,
        reference_isolation='references opened only after frozen prediction; evaluation conditioning only',
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        reporting_cuts=dict(long_X_s=2, restart_tolerance_frames=25),
        note='Raw vs official-filled segmentation is explanatory only; no alternative labels or scores are submitted or saved.')
    (output_dir/'diagnostics.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--benchmark-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    run(args.data_dir,args.benchmark_dir,args.output_dir)


if __name__=='__main__':main()
