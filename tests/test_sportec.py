"""Invented native Sportec files only. Never downloads or opens match assets."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import md5, sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from footballhack.io.sportec import load_sportec_metadata, load_sportec_tracking
from footballhack.external.references import load_references, reference_label
from footballhack.external.methods import METHODS, majority_vote, predict_methods
from footballhack.external.evaluation import (aggregate_halves, audit_match, evaluate_match,
                                             protocol, score_half, verify_frozen_sources)
from footballhack.external.splits import (assign_splits, club_overlap, load_manifest,
                                         partition_matches)


@pytest.fixture
def sportec_factory(tmp_path):
    def build(*, n=120, pitch=(100, 60), native=(10001, 20001), fps=25,
              gaps=(), ball_missing=(), namespace=False):
        metadata, positions = tmp_path / 'invented-metadata.xml', tmp_path / 'invented-positions.xml'
        root = ET.Element('PutDataRequest', {'xmlns': 'urn:invented:sportec'} if namespace else {})
        match = ET.SubElement(root, 'MatchInformation')
        ET.SubElement(match, 'General', MatchId='DFL-MAT-SYNTH', CompetitionId='SYNTHETIC',
                      HomeTeamId='home-7', GuestTeamId='away-2', Season='invented')
        ET.SubElement(match, 'Environment', PitchX=str(pitch[0]), PitchY=str(pitch[1]))
        teams = ET.SubElement(match, 'Teams')
        # Away team first: XML ordering must not determine H/A.
        for side, tid, pid in [('guest', 'away-2', 'person-2'), ('home', 'home-7', 'person-9')]:
            team = ET.SubElement(teams, 'Team', TeamId=tid, Role=side, TeamName='Synthetic ' + side)
            ET.SubElement(ET.SubElement(team, 'Players'), 'Player', PersonId=pid, ShirtNumber='1')
        ET.ElementTree(root).write(metadata, encoding='utf-8', xml_declaration=True)
        root = ET.Element('PutDataRequest', {'xmlns': 'urn:invented:sportec'} if namespace else {})
        container = ET.SubElement(root, 'Positions')
        expected = []
        for half_index, section in enumerate(('firstHalf', 'secondHalf')):
            t = np.arange(n)
            ball = np.column_stack((t * .1, np.zeros(n)))
            ball[65:85, 0] = -pitch[0] / 2 - 2
            # Second half starts with opposite ownership; no state may carry.
            owner = np.where((t >= 35) & (t < 85), 'A', 'H')
            if half_index == 1:
                owner = np.where(owner == 'H', 'A', 'H')
            players = np.stack((ball + [10, 0], ball + [12, 0]), axis=1)
            players[owner == 'H', 0] = ball[owner == 'H']
            players[owner == 'A', 1] = ball[owner == 'A']
            labels = np.where((t >= 65) & (t < 85), 'D', owner)
            expected.append((players.copy(), ball.copy(), labels))
            ball[list(ball_missing)] = np.nan
            origin = datetime(2024, 1, 1, hour=half_index, tzinfo=timezone.utc)
            # Ball last, so the adapter cannot rely on ball being parsed first.
            for obj, tid, pid in [(0, 'home-7', 'person-9'), (1, 'away-2', 'person-2'), (-1, 'BALL', 'ball')]:
                fs = ET.SubElement(container, 'FrameSet', GameSection=section, MatchId='DFL-MAT-SYNTH',
                                   TeamId=tid, PersonId=pid)
                for i in range(n):
                    if i in gaps:
                        continue
                    xy = ball[i] if obj == -1 else players[i, obj]
                    attrs = dict(N=str(native[half_index] + i), T=(origin + timedelta(seconds=i/fps)).isoformat(),
                                 X=str(xy[0]), Y=str(xy[1]), S='999', D='999', A='999')
                    if obj == -1:
                        attrs.update(BallStatus='0' if labels[i] == 'D' else '1',
                                     BallPossession={'H': '1', 'A': '2', 'D': 'bad'}[labels[i]])
                    ET.SubElement(fs, 'Frame', **attrs)
        ET.ElementTree(root).write(positions, encoding='utf-8', xml_declaration=True)
        return metadata, positions, expected
    return build


def load_pair(paths):
    metadata = load_sportec_metadata(paths[0])
    halves = load_sportec_tracking(paths[1], metadata)
    return halves, load_references(paths[1], halves)


def edit_xml(path, edit):
    tree = ET.parse(path)
    edit(tree.getroot())
    tree.write(path, encoding='utf-8')


@pytest.mark.parametrize('namespace', [False, True])
@pytest.mark.parametrize('pitch', [(100, 60), (110, 72)])
def test_native_to_metric_coordinates_and_explicit_team_mapping(sportec_factory, pitch, namespace):
    paths = sportec_factory(pitch=pitch, namespace=namespace)
    metadata = load_sportec_metadata(paths[0])
    halves, refs = load_pair(paths)
    assert metadata.tracking.team_ids == ('home-7', 'away-2')
    assert metadata.tracking.pitch_m == pitch
    for h, r, (players, ball, labels) in zip(halves, refs, paths[2]):
        np.testing.assert_allclose(h.tracking.player_xy, players + np.array(pitch)/2)
        np.testing.assert_allclose(h.tracking.ball_xy, ball + np.array(pitch)/2)
        np.testing.assert_array_equal(r.labels, labels)
        assert h.tracking.track_ids == ('person-9', 'person-2')
        np.testing.assert_array_equal(h.tracking.teams, ['H', 'A'])
        np.testing.assert_array_equal(h.tracking.frame, np.arange(120))
        assert not h.tracking.metadata.normalized
        assert h.tracking.metadata.fps == 25
        assert r.coverage['valid_fraction'] == 1


@pytest.mark.parametrize('native', [(1, 1), (0, 999), (10001, 20001)])
def test_native_frame_origins_and_half_boundaries(sportec_factory, native):
    halves, refs = load_pair(sportec_factory(native=native))
    for index, half in enumerate(halves):
        np.testing.assert_array_equal(half.native_frame, np.arange(native[index], native[index] + 120))
        assert half.tracking.metadata.periods[0].id == index + 1
        result = predict_methods(half)
        assert result['footballhack_causal'][0] == 'X'
        assert result['nearest_temporal'][0] == 'X'
        assert result['footballhack_causal'][10] == ('H' if index == 0 else 'A')
    assert audit_match(halves, refs)['coverage_gate_passed']


def test_native_gaps_are_nan_not_interpolated_or_shortened(sportec_factory):
    halves, refs = load_pair(sportec_factory(gaps=(12, 13, 14), ball_missing=(20, 21)))
    for h, r in zip(halves, refs):
        assert len(h.native_frame) == 120
        assert np.isnan(h.tracking.player_xy[12:15]).all()
        assert np.isnan(h.tracking.ball_xy[12:15]).all()
        assert np.isnan(h.tracking.ball_xy[20:22]).all()
        np.testing.assert_array_equal(r.labels[12:15], 'X')
        assert r.labels[20] in 'HA'  # reference valid even with no ball coordinate
        assert r.coverage['missing_ball_records'] == 3
        pred = predict_methods(h)['footballhack_causal']
        assert 'D' not in pred[6:30]


@pytest.mark.parametrize('fps', [20, 24, 30, 50])
def test_non25hz_timestamps_rejected(sportec_factory, fps):
    paths = sportec_factory(fps=fps)
    with pytest.raises(ValueError, match='25 Hz'):
        load_pair(paths)


@pytest.mark.parametrize('change,message', [
    (lambda r: r.find('Positions/FrameSet').set('MatchId', 'wrong'), 'MatchId'),
    (lambda r: r.find('Positions/FrameSet').set('GameSection', 'unknown'), 'firstHalf'),
    (lambda r: r.find('Positions/FrameSet').set('PersonId', 'unknown'), 'unknown participating'),
    (lambda r: r.find('Positions/FrameSet/Frame').set('N', '1.5'), 'N'),
    (lambda r: r.find('Positions/FrameSet/Frame').set('T', '2024-01-01T00:00:00'), 'timezone'),
    (lambda r: r.find('Positions/FrameSet/Frame').set('X', 'inf'), 'infinite'),
    (lambda r: r.find('Positions/FrameSet/Frame').set('X', 'wrong'), 'coordinate'),
    (lambda r: r.find('Positions/FrameSet/Frame').set('T', '2024-01-01T00:00:00.001+00:00'), 'timestamp'),
])
def test_invalid_native_inputs_fail_closed(sportec_factory, change, message):
    paths = sportec_factory()
    edit_xml(paths[1], change)
    with pytest.raises(ValueError, match=message):
        load_pair(paths)


def test_missing_axis_and_missing_attributes_stay_nan(sportec_factory):
    paths = sportec_factory()
    def edit(root):
        for fs in root.findall('Positions/FrameSet'):
            frame = fs.find('Frame')
            frame.attrib.pop('X')
            frame.set('Y', '')
    edit_xml(paths[1], edit)
    halves, _ = load_pair(paths)
    assert np.isnan(halves[0].tracking.player_xy[0]).all()
    assert np.isnan(halves[0].tracking.ball_xy[0]).all()


def test_duplicate_object_frames_rejected(sportec_factory):
    paths = sportec_factory()
    edit_xml(paths[1], lambda r: r.find('Positions/FrameSet').append(ET.fromstring(ET.tostring(r.find('Positions/FrameSet/Frame')))))
    with pytest.raises(ValueError, match='duplicate'):
        load_pair(paths)


def test_incomplete_halves_and_huge_frame_span_rejected(sportec_factory):
    paths = sportec_factory()
    with pytest.raises(ValueError, match='frame limit'):
        load_sportec_tracking(paths[1], load_sportec_metadata(paths[0]), max_half_frames=50)
    def edit(root):
        parent = root.find('Positions')
        for fs in list(parent):
            if fs.get('GameSection') == 'secondHalf':
                parent.remove(fs)
    edit_xml(paths[1], edit)
    with pytest.raises(ValueError, match='both halves'):
        load_pair(paths)


def test_third_party_officials_do_not_become_players(sportec_factory):
    paths = sportec_factory()
    def edit(root):
        frame_set = ET.SubElement(root.find('Positions'), 'FrameSet', TeamId='REFEREES', PersonId='referee')
        ET.SubElement(frame_set, 'Frame', N='bad', X='0', Y='0')
    edit_xml(paths[1], edit)
    halves, _ = load_pair(paths)
    assert halves[0].tracking.player_xy.shape == (120, 2, 2)


@pytest.mark.parametrize('target', ['metadata', 'positions'])
def test_xml_entities_and_malformed_xml_rejected(sportec_factory, target):
    paths = sportec_factory()
    path = paths[target == 'positions']
    path.write_text('<!DOCTYPE x [<!ENTITY e "unsafe">]><x>&e;</x>')
    with pytest.raises(ValueError, match='DTD/entity'):
        load_pair(paths)
    path.write_text('<broken>')
    with pytest.raises(ValueError, match='invalid XML|invalid Sportec XML'):
        load_pair(paths)


@pytest.mark.parametrize('edit,message', [
    (lambda r: r.find('MatchInformation/Environment').set('PitchX', '0'), 'PitchX'),
    (lambda r: r.find('MatchInformation/General').set('GuestTeamId', 'home-7'), 'distinct'),
    (lambda r: r.find('MatchInformation/Teams/Team').set('Role', 'home'), 'Role'),
    (lambda r: r.find('MatchInformation/Teams/Team/Players/Player').set('PersonId', 'person-9'), 'duplicate'),
])
def test_invalid_metadata(sportec_factory, edit, message):
    paths = sportec_factory()
    edit_xml(paths[0], edit)
    with pytest.raises(ValueError, match=message):
        load_pair(paths)


@pytest.mark.parametrize('status,possession,expected', [
    ('0', '1', 'D'), ('0', '2', 'D'), ('0', None, 'D'), ('0', 'bad', 'D'),
    ('1', '1', 'H'), ('1', '2', 'A'), ('1', '0', 'X'), ('1', None, 'X'),
    (None, '1', 'X'), ('bad', '1', 'X'), ('', '2', 'X'), ('1.0', '1', 'X'),
])
def test_reference_codes_conservative(status, possession, expected):
    assert reference_label(status, possession) == expected


@pytest.mark.parametrize('mode', ['flip', 'remove', 'invalid'])
def test_reference_mutation_cannot_change_any_prediction(sportec_factory, mode):
    paths = sportec_factory(ball_missing=(16, 17))
    halves, refs = load_pair(paths)
    original = [predict_methods(h) for h in halves]
    def edit(root):
        for fs in root.findall('Positions/FrameSet'):
            for f in fs:
                if fs.get('TeamId') == 'BALL':
                    if mode == 'remove':
                        f.attrib.pop('BallStatus', None)
                        f.attrib.pop('BallPossession', None)
                    else:
                        f.set('BallStatus', 'bad' if mode == 'invalid' else '0')
                        f.set('BallPossession', 'bad' if mode == 'invalid' else '2')
    edit_xml(paths[1], edit)
    new_halves, new_refs = load_pair(paths)
    assert not np.array_equal(refs[0].labels, new_refs[0].labels)
    for h, new_h, before in zip(halves, new_halves, original):
        np.testing.assert_array_equal(h.tracking.player_xy, new_h.tracking.player_xy)
        np.testing.assert_array_equal(h.tracking.ball_xy, new_h.tracking.ball_xy)
        np.testing.assert_array_equal(h.native_frame, new_h.native_frame)
        for method, labels in predict_methods(new_h).items():
            np.testing.assert_array_equal(labels, before[method])


@pytest.mark.parametrize('chunk_size', [1, 2, 7, 31, 4096])
def test_file_to_predictions_chunk_invariance_and_determinism(sportec_factory, chunk_size):
    halves, refs = load_pair(sportec_factory(ball_missing=(12, 13)))
    for half in halves:
        first = predict_methods(half)
        second = predict_methods(half, chunk_size=chunk_size)
        assert tuple(first) == METHODS
        for method in METHODS:
            np.testing.assert_array_equal(first[method], second[method])
            assert set(second[method]) <= set('HADX')
        assert set(first['footballhack_causal']) == set('HADX')
        assert first['footballhack_causal'][10] == ('H' if half.section == 'firstHalf' else 'A')


def test_source_speed_and_distance_are_not_features(sportec_factory):
    paths = sportec_factory()
    halves, _ = load_pair(paths)
    before = predict_methods(halves[0])
    def edit(root):
        for f in root.iter('Frame'):
            f.set('S', 'invalid'); f.set('D', 'NaN'); f.set('A', '-inf')
    edit_xml(paths[1], edit)
    after = predict_methods(load_pair(paths)[0][0])
    for name in METHODS:
        np.testing.assert_array_equal(before[name], after[name])


def test_published_smoothing_votes_and_legacy_missing_semantics():
    codes = np.r_[np.zeros(20), np.full(20, -1), np.ones(20)]
    actual = majority_vote(codes)
    # All 60 frames visible in the central +/-50 window: 20 vs 20 => home tie.
    assert actual[30] == 'H'
    np.testing.assert_array_equal(majority_vote(np.full(120, -1)), 'D')
    np.testing.assert_array_equal(majority_vote(np.ones(120)), 'A')


def test_no_ball_baseline_D_is_not_model_dead_evidence(sportec_factory):
    halves, _ = load_pair(sportec_factory(ball_missing=range(120)))
    predictions = predict_methods(halves[0])
    np.testing.assert_array_equal(predictions['nearest_raw'], 'D')
    np.testing.assert_array_equal(predictions['nearest_smooth50'], 'D')
    np.testing.assert_array_equal(predictions['nearest_temporal'], 'X')
    np.testing.assert_array_equal(predictions['footballhack_causal'], 'X')


def test_official_scorer_bytes_and_frozen_model_integrity():
    p = verify_frozen_sources()
    path = Path(__file__).resolve().parents[1] / 'src/footballhack/external/_official/possession_metrics.py'
    assert sha256(path.read_bytes()).hexdigest() == p['official_scorer_sha256']
    assert p['pipeline_defaults']['temporal']['mode'] == 'causal'
    assert p['pipeline_defaults']['control']['use_player_motion'] is False


def test_scoring_matches_unchanged_official_implementation(sportec_factory):
    from footballhack.external._official import possession_metrics as pm
    halves, refs = load_pair(sportec_factory(ball_missing=(16, 17)))
    pred = predict_methods(halves[0])['footballhack_causal']
    expected = pm.evaluate(pm.label_codes(refs[0].labels), pm.label_codes(pred))
    actual = score_half(halves[0], refs[0], pred)
    for key in ('macro_f1', 'spell_f1_mean', 'per_class', 'n_graded', 'n_pred_spells', 'confusion', 'spell_f1'):
        assert actual[key] == expected[key]
    assert actual['n_frames'] == 120
    assert actual['diagnostics']['missing_ball']['n_valid_reference'] == 2


def test_X_mask_and_spell_fill_are_official_not_frame_exclusion(sportec_factory):
    halves, refs = load_pair(sportec_factory())
    reference = replace(refs[0], labels=np.array(['H'] * 120))
    pred = np.array(['H'] * 120)
    pred[30:40] = 'X'
    scored = score_half(halves[0], reference, pred)
    assert scored['spell_f1_mean'] == 1
    assert scored['macro_f1'] < 1
    assert scored['n_abstain_on_graded'] == 10
    assert scored['n_pred_spells'] == 1
    gt = reference.labels.copy(); gt[60:65] = 'X'
    masked = score_half(halves[0], replace(reference, labels=gt), pred)
    assert masked['n_graded'] == 115
    assert masked['n_gt_spells'] == 2


def test_score_alignment_and_fps_failures(sportec_factory):
    halves, refs = load_pair(sportec_factory())
    with pytest.raises(ValueError, match='alignment'):
        score_half(halves[0], refs[1], refs[0].labels)
    with pytest.raises(ValueError, match='alignment'):
        score_half(halves[0], replace(refs[0], native_frame=refs[0].native_frame+1), refs[0].labels)
    with pytest.raises(ValueError, match='frame count'):
        score_half(halves[0], refs[0], refs[0].labels[:-1])
    bad_half = replace(halves[0], tracking=replace(halves[0].tracking,
        metadata=replace(halves[0].tracking.metadata, fps=50)))
    with pytest.raises(ValueError, match='25 Hz'):
        score_half(bad_half, refs[0], refs[0].labels)


def test_references_cannot_shift_native_frames_or_clocks(sportec_factory):
    paths = sportec_factory()
    halves, _ = load_pair(paths)
    def edit(root):
        ball = next(f for f in root.findall('Positions/FrameSet') if f.get('TeamId') == 'BALL')
        ball.find('Frame').set('T', '2024-01-01T00:00:00.001+00:00')
    edit_xml(paths[1], edit)
    with pytest.raises(ValueError, match='misalignment'):
        load_references(paths[1], halves)


def test_coverage_gate_blocks_invalid_reference_before_prediction(sportec_factory, monkeypatch):
    paths = sportec_factory()
    def edit(root):
        for f in root.iter('Frame'):
            f.attrib.pop('BallStatus', None)
    edit_xml(paths[1], edit)
    halves, refs = load_pair(paths)
    audit = audit_match(halves, refs)
    assert not audit['coverage_gate_passed']
    assert refs[0].coverage['missing_status'] == 120
    monkeypatch.setattr('footballhack.external.methods.predict_methods', lambda *a, **kw: pytest.fail('prediction must not start'))
    with pytest.raises(ValueError, match='coverage audit failed'):
        evaluate_match(halves, refs)


def test_evaluation_reports_all_methods_each_half_and_aggregates(sportec_factory):
    halves, refs = load_pair(sportec_factory())
    report = evaluate_match(halves, refs, chunk_size=7)
    assert set(report['methods']) == set(METHODS)
    for method, values in report['methods'].items():
        assert len(values['halves']) == 2
        assert values['summary']['n_frames'] == 240
        assert values['summary']['n_graded'] == 240
        assert values['summary']['spell_f1_mean'] == pytest.approx(np.mean([h['spell_f1_mean'] for h in values['halves']]))
        assert 'turnover_timing_25_frames' in values['halves'][0]['diagnostics']
    # Reports must encode as strict JSON and be byte reproducible.
    assert json.dumps(report, sort_keys=True, allow_nan=False) == json.dumps(evaluate_match(halves, refs, chunk_size=31), sort_keys=True, allow_nan=False)


def test_halves_are_scored_independently_even_when_labels_agree(sportec_factory):
    halves, refs = load_pair(sportec_factory())
    results = []
    for half, ref in zip(halves, refs):
        labels = np.full(120, 'H')
        results.append(score_half(half, replace(ref, labels=labels), labels))
    aggregate = aggregate_halves(results)
    assert aggregate['n_gt_spells'] == aggregate['n_pred_spells'] == 2
    assert aggregate['spell_f1_mean'] == 1
    assert aggregate['reference_mean_spell_duration_s'] == pytest.approx(4.8)


def test_frozen_manifest_metadata_only_seed_and_overlap():
    m = load_manifest()
    assert m['seed'] == 2027
    assert assign_splits(m['matches']) == m['assignments']
    assert assign_splits(list(reversed(m['matches']))) == m['assignments']
    assert club_overlap(m['matches'], m['assignments']) == m['club_overlap']
    assert [len(m['assignments'][p]) for p in ('development', 'validation', 'holdout')] == [3, 2, 2]
    assert len(set(sum(m['assignments'].values(), []))) == 7
    assert all(teams == ['DFL-CLU-00000P'] for teams in m['club_overlap'].values())
    assert m['assignments']['holdout'] == ['J03WMX', 'J03WOH']


def test_holdout_rejected_before_reading_data_or_references(sportec_factory, monkeypatch):
    with pytest.raises(ValueError, match='sealed'):
        partition_matches('holdout')
    from footballhack.external.__main__ import run_partition
    with pytest.raises(ValueError, match='sealed'):
        run_partition(Path('/does-not-exist'), 'holdout', evaluate=True)
    halves, refs = load_pair(sportec_factory())
    sealed_half = replace(halves[0], match_id='DFL-MAT-J03WMX')
    monkeypatch.setattr('footballhack.external.methods.predict_methods', lambda *a, **kw: pytest.fail('holdout prediction attempted'))
    with pytest.raises(ValueError, match='sealed'):
        evaluate_match((sealed_half, halves[1]), refs)


def fixture_record(paths):
    record = dict(match_id='SYNTH', team_ids=['home-7', 'away-2'], pitch_m=[100., 60.])
    for kind, path in zip(('metadata', 'positions'), paths[:2]):
        content = path.read_bytes()
        record[kind] = dict(name=path.name, size=len(content), computed_md5=md5(content).hexdigest())
    return record


def test_cli_files_through_full_evaluation_and_json(sportec_factory, tmp_path, monkeypatch, capsys):
    import footballhack.external.__main__ as cli
    paths = sportec_factory()
    record = fixture_record(paths)
    monkeypatch.setattr(cli, 'partition_matches', lambda partition: (record,))
    output = tmp_path / 'invented-report.json'
    assert cli.main(['evaluate', '--partition', 'development', '--data-dir', str(tmp_path), '--output', str(output), '--chunk-size', '7']) == 0
    report = json.loads(output.read_text())
    assert report['evaluated'] is True
    assert report['comparison']['footballhack_causal']['n_frames'] == 240
    assert set(report['delta_vs_smooth50']) == set(METHODS)
    assert 'spell_f1_mean' not in capsys.readouterr().out
    assert cli.main(['evaluate', '--partition', 'development', '--data-dir', str(tmp_path), '--output', str(output)]) == 2


def test_audit_does_not_predict_or_score(sportec_factory, tmp_path, monkeypatch):
    import footballhack.external.__main__ as cli
    paths = sportec_factory()
    monkeypatch.setattr(cli, 'partition_matches', lambda partition: (fixture_record(paths),))
    monkeypatch.setattr(cli, 'evaluate_match', lambda *a, **k: pytest.fail('audit must not evaluate'))
    monkeypatch.setattr('footballhack.external.methods.predict_methods', lambda *a, **k: pytest.fail('audit must not predict'))
    report = cli.run_partition(tmp_path, 'development')
    assert not report['evaluated']
    assert report['matches'][0]['coverage_gate_passed']
    assert 'comparison' not in report


def test_cli_checksum_and_path_guards(sportec_factory, tmp_path, monkeypatch):
    import footballhack.external.__main__ as cli
    paths = sportec_factory()
    record = fixture_record(paths)
    monkeypatch.setattr(cli, 'partition_matches', lambda partition: (record,))
    paths[1].write_bytes(paths[1].read_bytes() + b' ')
    with pytest.raises(ValueError, match='checksum'):
        cli.run_partition(tmp_path, 'development')
    with pytest.raises(ValueError, match='outside any Git'):
        cli.outside_git(Path(__file__).parent)
    monkeypatch.delenv('SPORTEC_DATA_DIR', raising=False)
    monkeypatch.setenv('MTAH_DATA_DIR', '/do-not-access-competition')
    assert cli.main(['audit', '--partition', 'development', '--output', str(tmp_path/'r.json')]) == 2


def test_turnover_latency_false_switch_and_stoppage_diagnostics(sportec_factory):
    halves, refs = load_pair(sportec_factory())
    truth = np.array(['H'] * 40 + ['A'] * 40 + ['D'] * 20 + ['H'] * 20)
    pred = truth.copy()
    pred[40:45] = 'H'  # known five-frame late turnover
    scored = score_half(halves[0], replace(refs[0], labels=truth), pred)
    diagnostic = scored['diagnostics']
    turn = diagnostic['turnover_timing_25_frames']
    assert turn['n_gt'] == turn['n_pred'] == turn['n_matched'] == 1
    assert turn['mean_latency'] == turn['mean_abs_latency'] == 5
    assert diagnostic['false_possession_switches'] == 0
    assert diagnostic['stoppage_entry_timing_25_frames']['n_matched'] == 1
    assert diagnostic['restart_timing_25_frames']['n_matched'] == 1
    assert diagnostic['duration_s']['reference_dead'] == .8
    assert diagnostic['dead_ball_precision_recall']['precision'] == 1
    pred[10:15] = 'A'  # two extra direct switches
    false = score_half(halves[0], replace(refs[0], labels=truth), pred)['diagnostics']
    assert false['false_possession_switches'] == 2


def test_missing_ball_metrics_keep_invalid_reference_out_without_new_spells(sportec_factory):
    halves, refs = load_pair(sportec_factory(ball_missing=(0, 1, 2, 3)))
    truth = refs[0].labels.copy(); truth[0] = 'X'
    pred = truth.copy(); pred[1:4] = ['H', 'A', 'X']
    scored = score_half(halves[0], replace(refs[0], labels=truth), pred)
    missing = scored['diagnostics']['missing_ball']
    assert missing['n_frames'] == 4
    assert missing['n_valid_reference'] == 3
    assert missing['accuracy'] == pytest.approx(1/3)
    assert missing['per_class']['H']['recall'] == pytest.approx(1/3)
    assert 'spell_f1_mean' not in missing


@pytest.mark.parametrize('n', [1, 49, 51, 101, 205])
def test_smooth50_matches_direct_published_window_counting(n):
    codes = np.random.default_rng(1234).integers(-1, 2, size=n)
    expected = []
    for i in range(n):
        window = codes[max(0, i-50):min(n, i+51)]
        h, a = np.count_nonzero(window == 0), np.count_nonzero(window == 1)
        expected.append('D' if h+a == 0 else ('H' if h >= a else 'A'))
    np.testing.assert_array_equal(majority_vote(codes), expected)


def test_half_clock_overlap_rejected(sportec_factory):
    paths = sportec_factory()
    def edit(root):
        for fs in root.findall('Positions/FrameSet'):
            if fs.get('GameSection') == 'secondHalf':
                for f in fs:
                    f.set('T', f.get('T').replace('T01:', 'T00:'))
    edit_xml(paths[1], edit)
    with pytest.raises(ValueError, match='overlap'):
        load_pair(paths)


def test_timestamp_utc_offsets_do_not_shift_alignment(sportec_factory):
    paths = sportec_factory()
    before, _ = load_pair(paths)
    def edit(root):
        for f in root.iter('Frame'):
            local = datetime.fromisoformat(f.get('T')).astimezone(timezone(timedelta(hours=2)))
            f.set('T', local.isoformat())
    edit_xml(paths[1], edit)
    after, _ = load_pair(paths)
    for first, second in zip(before, after):
        np.testing.assert_array_equal(first.timestamp_us, second.timestamp_us)


def test_unknown_reference_codes_reported_without_defaults(sportec_factory):
    paths = sportec_factory()
    def edit(root):
        for fs in root.findall('Positions/FrameSet'):
            if fs.get('TeamId') == 'BALL':
                frames = fs.findall('Frame')
                frames[1].set('BallStatus', 'invalid')
                frames[2].set('BallPossession', '0')
                frames[3].attrib.pop('BallStatus')
    edit_xml(paths[1], edit)
    halves, refs = load_pair(paths)
    for ref in refs:
        np.testing.assert_array_equal(ref.labels[1:4], 'X')
        assert ref.coverage['invalid_status'] == 1
        assert ref.coverage['missing_status'] == 1
        assert ref.coverage['unknown_live_possession'] == 1
        assert ref.coverage['valid_fraction'] == pytest.approx(117/120)


def test_raw_tie_and_shared_decoder_uncertainty_are_declared(sportec_factory):
    paths = sportec_factory()
    def edit(root):
        for fs in root.findall('Positions/FrameSet'):
            for f in fs:
                f.set('X', '0'); f.set('Y', '0')
    edit_xml(paths[1], edit)
    halves, _ = load_pair(paths)
    result = predict_methods(halves[0])
    np.testing.assert_array_equal(result['nearest_raw'], 'H')
    np.testing.assert_array_equal(result['nearest_temporal'], 'X')
    np.testing.assert_array_equal(result['footballhack_causal'], 'X')


def test_shared_temporal_and_frozen_model_are_H_A_symmetric(sportec_factory):
    halves, _ = load_pair(sportec_factory())
    half = halves[0]
    # Swap physical team positions while preserving valid explicit team mapping.
    swapped = replace(half, tracking=replace(half.tracking, player_xy=half.tracking.player_xy[:, ::-1].copy()))
    original, opposite = predict_methods(half), predict_methods(swapped)
    for name in ('nearest_raw', 'nearest_temporal', 'footballhack_causal'):
        expected = np.where(original[name] == 'H', 'A', np.where(original[name] == 'A', 'H', original[name]))
        np.testing.assert_array_equal(opposite[name], expected)


def test_missing_or_wrong_reference_label_arrays_fail(sportec_factory):
    halves, refs = load_pair(sportec_factory())
    with pytest.raises(ValueError, match='no valid reference'):
        score_half(halves[0], replace(refs[0], labels=np.full(120, 'X')), np.full(120, 'H'))
    with pytest.raises(ValueError, match='labels outside'):
        score_half(halves[0], refs[0], np.full(120, '?'))


def test_tampered_model_and_split_checksums_fail(monkeypatch):
    import footballhack.external.evaluation as ev
    changed = protocol()
    changed['model_sha256']['features.py'] = '0' * 64
    monkeypatch.setattr(ev, 'protocol', lambda: changed)
    with pytest.raises(ValueError, match='frozen model source changed'):
        ev.verify_frozen_sources()
    changed = protocol(); changed['split_manifest_sha256'] = '0' * 64
    monkeypatch.setattr(ev, 'protocol', lambda: changed)
    with pytest.raises(ValueError, match='frozen split manifest changed'):
        ev.verify_frozen_sources()


def test_dataset_and_report_symlinks_into_git_are_rejected(tmp_path):
    from footballhack.external.__main__ import outside_git
    link = tmp_path / 'link'
    link.symlink_to(Path(__file__).parent)
    with pytest.raises(ValueError, match='outside any Git'):
        outside_git(link/'synthetic.xml')


def test_existing_external_artifact_reused_without_network(sportec_factory, tmp_path, monkeypatch):
    import footballhack.external.__main__ as cli
    paths = sportec_factory()
    monkeypatch.setattr(cli, 'urlopen', lambda *a, **k: pytest.fail('must reuse verified local artifact'))
    cli.fetch_artifact(tmp_path, fixture_record(paths)['metadata'])


def test_fetch_verifies_download_and_removes_failed_partial(tmp_path, monkeypatch):
    import io
    import footballhack.external.__main__ as cli
    body = b'invented external fixture'
    artifact = dict(name='invented.xml', size=len(body), computed_md5=md5(body).hexdigest(), download_url='https://example.invalid/synthetic')
    monkeypatch.setattr(cli, 'urlopen', lambda *a, **k: io.BytesIO(body))
    cli.fetch_artifact(tmp_path, artifact)
    assert (tmp_path/'invented.xml').read_bytes() == body
    artifact['name'] = 'bad.xml'; artifact['computed_md5'] = '0' * 32
    with pytest.raises(ValueError, match='checksum'):
        cli.fetch_artifact(tmp_path, artifact)
    assert not (tmp_path/'bad.xml').exists()
    assert not (tmp_path/'bad.xml.partial').exists()


def test_single_match_selection_rejects_other_partitions():
    from footballhack.external.__main__ import select_matches
    assert [r['match_id'] for r in select_matches('development', 'J03WN1')] == ['J03WN1']
    for match in ('J03WPY', 'J03WMX', 'unknown'):
        with pytest.raises(ValueError, match='requested unsealed partition'):
            select_matches('development', match)
    with pytest.raises(ValueError, match='sealed'):
        select_matches('holdout', 'J03WMX')


def test_single_match_audit_reports_time_and_memory_without_scoring(sportec_factory, tmp_path, monkeypatch):
    import footballhack.external.__main__ as cli
    paths = sportec_factory()
    monkeypatch.setattr(cli, 'partition_matches', lambda partition: (fixture_record(paths),))
    monkeypatch.setattr(cli, 'evaluate_match', lambda *a, **k: pytest.fail('audit cannot score'))
    report = cli.run_partition(tmp_path, 'development', match_id='SYNTH')
    assert len(report['matches']) == 1
    resources = report['matches'][0]['resources']
    assert report['matches'][0]['reference_isolation']['passed']
    assert not report['matches'][0]['reference_isolation']['native_predictions_computed']
    assert resources['elapsed_s'] > 0 and resources['process_peak_rss_bytes'] > 0
    half = report['matches'][0]['halves'][0]
    assert half['duration_s'] == 4.8
    assert half['team_ids'] == ('home-7', 'away-2')
    assert half['player_xy_range_m']['y'] == [30, 30]
    assert half['timestamp_end_inclusive_us'] - half['timestamp_start_us'] == 119 * 40_000


@pytest.mark.parametrize('dimensions,match,ok', [((100, 60), 'DFL-MAT-SYNTH', True),
    ((105, 68), 'DFL-MAT-SYNTH', False), ((100, 60), 'wrong', False)])
def test_native_tracking_pitch_header_matches_metadata(sportec_factory, dimensions, match, ok):
    paths = sportec_factory()
    def edit(root):
        header = ET.Element('MetaData', MatchId=match, Type='pitch-size')
        ET.SubElement(header, 'PitchSize', X=str(dimensions[0]), Y=str(dimensions[1]))
        root.find('Positions').insert(0, header)
    edit_xml(paths[1], edit)
    if ok:
        halves, _ = load_pair(paths)
        assert halves[0].tracking.metadata.pitch_m == (100, 60)
    else:
        with pytest.raises(ValueError, match='disagrees'):
            load_pair(paths)


def test_benchmark_observation_preserves_all_frozen_predictions(sportec_factory):
    from footballhack.external.benchmark import observe_methods
    halves, _ = load_pair(sportec_factory(ball_missing=(10, 11)))
    original = predict_methods(halves[0])
    observed, profile = observe_methods(halves[0])
    for name in METHODS:
        np.testing.assert_array_equal(original[name], observed[name])
    assert profile['total_bundled_prediction_s'] > 0
    assert profile['stages']['nearest_temporal_decoder']['calls'] == 1


def test_benchmark_diagnostics_perfect_and_missed_dead_intervals():
    from footballhack.external.benchmark import diagnostics
    reference = np.array(['H']*50+['D']*50+['A']*50)
    exact = diagnostics([(reference, reference)])
    assert exact['dead_intervals']['matched_iou50'] == 1
    assert exact['team_ownership_errors_raw'] == 0
    assert exact['transitions']['all_25']['n_matched'] == 2
    predicted = np.array(['H']*100+['A']*50)
    missed = diagnostics([(reference, predicted)])
    assert missed['dead_intervals']['missed_iou50'] == 1
    assert missed['dead_intervals']['no_overlap_reference_intervals'] == 1
    assert missed['missed_dead_frames_assigned_to_team'] == 50


def test_benchmark_X_and_halftime_aggregation_never_merge_runs():
    from footballhack.external.benchmark import diagnostics
    reference = np.full(100, 'H')
    prediction = np.full(100, 'X')
    result = diagnostics([(reference, prediction), (reference, prediction)])
    assert result['predicted_X']['n_frames'] == 200
    assert result['predicted_X']['n_runs'] == 2
    assert result['predicted_X']['longest_s'] == 4
    assert result['predicted_X']['n_runs_at_least_cut'] == 2


def test_benchmark_capture_uses_original_runner_once_per_half(sportec_factory, tmp_path, monkeypatch):
    import footballhack.external.benchmark as bench
    paths = sportec_factory()
    halves, refs = load_pair(paths)
    import footballhack.external.__main__ as cli
    monkeypatch.setattr(cli, 'partition_matches', lambda p: (fixture_record(paths),))
    # Exercise callback capture without reading any native audit or data.
    observed = []
    original = bench._FROZEN_PREDICT_METHODS
    def count(half, **kwargs):
        observed.append(half.section)
        return original(half, **kwargs)
    monkeypatch.setattr(bench, '_FROZEN_PREDICT_METHODS', count)
    from unittest.mock import patch
    def wrapped(half, **kwargs):
        return bench.observe_methods(half, **kwargs)[0]
    with patch('footballhack.external.methods.predict_methods', wrapped):
        report = evaluate_match(halves, refs)
    assert observed == ['firstHalf', 'secondHalf']
    assert set(report['methods']) == set(METHODS)


def test_benchmark_report_equal_match_means_and_paired_differences(sportec_factory, tmp_path):
    from copy import deepcopy
    from footballhack.external.benchmark import APPROVED, diagnostics
    from footballhack.external.benchmark_report import build_report, render_report
    halves, references = load_pair(sportec_factory())
    match = evaluate_match(halves, references)
    profiles = []
    for half in halves:
        prediction = predict_methods(half)
        for name in METHODS:
            match['methods'][name].setdefault('test_labels', []).append(prediction[name])
        profiles.append(dict(stages={name:dict(wall_s=.01) for name in (
            'footballhack_pipeline','nearest_shared_features','nearest_array_preparation_and_runner_overhead',
            'nearest_temporal_decoder','shared_dead_evidence_nested_in_pipeline','majority_vote')}))
    for name in METHODS:
        labels = match['methods'][name].pop('test_labels')
        match['methods'][name]['aggregate_diagnostics'] = diagnostics([(r.labels, p) for r,p in zip(references,labels)])
        match['methods'][name]['official_scoring_wall_s'] = .01
    match['prediction_profiles'] = profiles
    match['resources'] = dict(elapsed_s=1., process_peak_rss_bytes=1024**2)
    for index, match_id in enumerate(APPROVED):
        data = deepcopy(match); data['match_id'] = 'DFL-MAT-'+match_id
        data['methods']['nearest_raw']['summary']['spell_f1_mean'] = (0., .3, .9)[index]
        directory = tmp_path/match_id; directory.mkdir()
        (directory/'report.json').write_text(json.dumps(dict(partition='development',evaluated=True,protocol=protocol(),matches=[data])))
    result = build_report(tmp_path)
    assert result['equal_match_average']['nearest_raw']['spell_f1_mean'] == pytest.approx(.4)
    difference = result['paired_differences']['footballhack_causal minus nearest_raw']['equal_match_mean']['spell_f1_mean']
    assert difference == pytest.approx(result['equal_match_average']['footballhack_causal']['spell_f1_mean']-.4)
    rendered = render_report(result)
    assert all(match_id in rendered for match_id in APPROVED)
    assert 'not isolated per-method memory' in rendered
