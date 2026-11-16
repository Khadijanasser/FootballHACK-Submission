"""Synthetic checks for observation-only frozen-prediction explanations."""
from dataclasses import replace
import inspect
import json
import sys

import numpy as np
import pytest

from footballhack.pipeline import PipelineConfig, predict_possession
from footballhack.temporal import decode_possession
from footballhack.external.diagnose import restart_required_mask, summarize_half
from test_sportec import sportec_factory, load_pair


def test_diagnostic_counts_conserve_frames_and_serialize(sportec_factory):
    halves,refs=load_pair(sportec_factory())
    cfg=PipelineConfig(n_frames=120)
    result=predict_possession(halves[0].tracking,config=cfg)
    summary=summarize_half(result,refs[0].labels,cfg)
    assert summary['n_frames']==120
    assert sum(v['n_frames'] for v in summary['by_reference'].values())==120
    assert sum(summary['X']['reasons'].values())==np.count_nonzero(result.labels=='X')
    assert summary['dead_evidence']['tag_available_frames']==0
    assert summary['restarts']['n_gt']==summary['restarts']['n_matched']+summary['restarts']['n_missed']
    json.dumps(summary,allow_nan=False)


def test_reconstructed_restart_latch_equals_original_decoder_locals(sportec_factory):
    halves,_=load_pair(sportec_factory(n=220))
    data=halves[0].tracking
    ball=np.column_stack((np.full(220,50.),np.full(220,30.)))
    ball[:65,0]+=np.arange(65)*.1
    ball[65:75,0]=-2
    ball[180:,0]+=np.arange(40)*.1
    players=np.stack((ball.copy(),ball+[10,0]),axis=1)
    cfg=PipelineConfig(n_frames=220)
    result=predict_possession(replace(data,ball_xy=ball,player_xy=players),config=cfg)
    assert 'dead_support_expired' in result.possession.reason
    source,start=inspect.getsourcelines(decode_possession)
    line=start+next(i for i,text in enumerate(source) if 'phases[i] = phase' in text)
    values=[]
    def trace(frame,event,arg):
        if frame.f_code is decode_possession.__code__ and event=='line' and frame.f_lineno==line:
            values.append(frame.f_locals['restart_required'])
        return trace
    old=sys.gettrace()
    try:
        sys.settrace(trace)
        decoded=decode_possession(result.control,result.dead_ball,fps=25,config=cfg.temporal)
    finally:
        sys.settrace(old)
    np.testing.assert_array_equal(restart_required_mask(decoded),values)
    assert restart_required_mask(decoded)[150]
    assert not restart_required_mask(decoded)[200]


def test_reference_conditioning_never_mutates_prediction_or_evidence(sportec_factory):
    halves,_=load_pair(sportec_factory())
    cfg=PipelineConfig(n_frames=120)
    result=predict_possession(halves[0].tracking,config=cfg)
    before=(result.labels.copy(),result.control.scores.copy(),result.possession.phase.copy())
    h=summarize_half(result,np.full(120,'H'),cfg)
    d=summarize_half(result,np.full(120,'D'),cfg)
    assert h['by_reference']['D']['n_frames']==0
    assert d['by_reference']['D']['n_frames']==120
    for actual,expected in zip((result.labels,result.control.scores,result.possession.phase),before):
        np.testing.assert_array_equal(actual,expected)


def test_raw_vs_filled_segmentation_is_explanatory_only(sportec_factory):
    halves,_=load_pair(sportec_factory())
    cfg=PipelineConfig(n_frames=120)
    result=predict_possession(halves[0].tracking,config=cfg)
    labels=np.array(['H']*10+['X']*50+['H']*10+['D']*50)
    result=replace(result,possession=replace(result.possession,labels=labels))
    reference=np.array(['H']*70+['D']*50)
    before=labels.copy()
    d=summarize_half(result,reference,cfg)['duration']
    assert d['raw_team_spells']==2 and d['filled_team_spells']==1
    assert d['raw_mean_team_spell_s']==pytest.approx(.4)
    assert d['filled_mean_team_spell_s']==pytest.approx(2.8)
    assert d['X_filled_to_team_frames']==50
    np.testing.assert_array_equal(result.labels,before)


def test_tracking_missingness_separate_from_weak_observed_control(sportec_factory):
    halves,refs=load_pair(sportec_factory(ball_missing=(15,16,17)))
    cfg=PipelineConfig(n_frames=120)
    result=predict_possession(halves[0].tracking,config=cfg)
    s=summarize_half(result,refs[0].labels,cfg)
    assert s['prediction_only_evidence']['missing_control_frames']==3
    assert s['prediction_only_evidence']['phase_conflict_frames']==0


def test_mismatched_reference_length_is_rejected(sportec_factory):
    halves,_=load_pair(sportec_factory())
    cfg=PipelineConfig(n_frames=120)
    result=predict_possession(halves[0].tracking,config=cfg)
    with pytest.raises(ValueError,match='alignment'):
        summarize_half(result,np.full(119,'H'),cfg)


def test_pooled_diagnostic_means_use_summed_counts(sportec_factory):
    from footballhack.external.diagnose_report import aggregate
    halves,refs=load_pair(sportec_factory())
    cfg=PipelineConfig(n_frames=120)
    summaries=[summarize_half(predict_possession(h.tracking,config=cfg),r.labels,cfg)
               for h,r in zip(halves,refs)]
    result=aggregate(dict(matches={'invented':summaries},original_artifacts_unchanged=True,reproduced_labels_identical=True))
    assert result['n_frames']==240
    d=result['duration']
    assert d['filled_mean_team_spell_s']==pytest.approx(d['filled_team_frames']/d['filled_team_spells']/25)
    assert sum(result['X']['reasons'].values())==result['X']['n_frames']
    assert result['restarts']['n_gt']==result['restarts']['n_matched']+result['restarts']['n_missed']


def test_report_is_generated_from_synthetic_counts_not_native_results(sportec_factory):
    from footballhack.external.diagnose_report import aggregate,render
    halves,refs=load_pair(sportec_factory())
    cfg=PipelineConfig(n_frames=120)
    summary=summarize_half(predict_possession(halves[0].tracking,config=cfg),refs[0].labels,cfg)
    a=aggregate(dict(matches={'invented':[summary]},original_artifacts_unchanged=True,reproduced_labels_identical=True))
    report=render(a)
    assert f"Official filling adds {a['duration']['X_filled_to_team_frames']} team-labelled frames" in report
    assert f"For {a['restarts']['missed']['counts'].get('no_raw_D_in_preceding_reference_D_interval',0)} of {a['restarts']['n_missed']} misses" in report
    a['duration']['X_filled_to_team_frames']=1234567  # invented counter, not a native result
    assert 'Official filling adds 1234567 team-labelled frames' in render(a)
