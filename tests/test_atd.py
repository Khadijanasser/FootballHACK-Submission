"""Public-format parsing with invented XML/text files only."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from footballhack import load_atd, load_atd_metadata


@pytest.mark.parametrize("specs", [True, False])
@pytest.mark.parametrize("namespace,trailing", [(False, False), (True, False), (False, True)])
def test_file_parsing_frame_conversion_team_and_slot_order(atd_factory, specs, namespace, trailing):
    metadata, raw, players, ball = atd_factory(n=4, specs=specs, namespace=namespace, trailing=trailing)
    data = load_atd(metadata, raw)
    np.testing.assert_array_equal(data.frame, [0, 1, 2, 3])
    np.testing.assert_allclose(data.player_xy, players)
    np.testing.assert_allclose(data.ball_xy, ball)
    np.testing.assert_array_equal(data.teams, ["H", "A", "H"])
    assert data.track_ids == ("slot-9", "slot-2", "slot-1")
    assert data.metadata.team_names == ("Synthetic Home", "Synthetic Away")
    assert data.metadata.pitch_m == (100., 60.) and data.metadata.fps == 25
    assert data.metadata.normalized


def test_channel_layout_changes_resolve_back_to_metadata_order(atd_factory):
    metadata, raw, players, _ = atd_factory(n=8, orders=[(1, 4, (2, 0, 1)), (5, 8, (1, 2, 0))])
    data = load_atd(metadata, raw)
    np.testing.assert_allclose(data.player_xy, players)
    np.testing.assert_array_equal(data.teams, ["H", "A", "H"])


def test_coordinates_missing_gaps_and_off_pitch_are_preserved(atd_factory):
    ball = np.array([[.5, .5], [np.nan, .5], [-.05, 1.1], [.6, .5]])
    players = np.tile([[.4, .5], [.6, .5], [.2, .5]], (4, 1, 1))
    players[1, 0, 1] = np.nan
    metadata, raw, _, _ = atd_factory(n=4, players=players, ball=ball, record_frames=[0, 1, 3])
    data = load_atd(metadata, raw)
    np.testing.assert_array_equal(data.frame, [0, 1, 3])
    assert np.isnan(data.ball_xy[1, 0]) and data.ball_xy[1, 1] == .5
    assert np.isnan(data.player_xy[1, 0, 1])
    metadata, raw, _, _ = atd_factory(n=4, players=players, ball=ball, folder="outside")
    np.testing.assert_allclose(load_atd(metadata, raw).ball_xy[2], [-.05, 1.1])


def test_metadata_periods_use_raw_one_based_inclusive_counts(atd_factory):
    metadata, raw, _, _ = atd_factory(n=12, periods=[(1, 5), (8, 12)])
    periods = load_atd(metadata, raw).metadata.periods
    assert [(p.id, p.start_frame, p.end_frame) for p in periods] == [(1, 1, 5), (2, 8, 12)]


@pytest.mark.parametrize("empty_value", [None, "", "   "])
def test_optional_empty_provider_parameters_do_not_invalidate_tracking(atd_factory, empty_value):
    metadata, raw, players, ball = atd_factory(n=4, periods=[(0, 4)])
    tree = ET.parse(metadata)
    provider = tree.find("Metadata/GlobalConfig/ProviderGlobalParameters")
    parameter = ET.SubElement(provider, "ProviderParameter")
    ET.SubElement(parameter, "Name").text = "optional_export_note"
    ET.SubElement(parameter, "Value").text = empty_value
    tree.write(metadata)
    data = load_atd(metadata, raw)
    np.testing.assert_allclose(data.player_xy, players)
    np.testing.assert_allclose(data.ball_xy, ball)
    assert [(p.start_frame, p.end_frame) for p in data.metadata.periods] == [(0, 4)]


def test_empty_optional_period_boundaries_are_absent_not_fabricated(atd_factory):
    metadata, _, _, _ = atd_factory(n=4)
    tree = ET.parse(metadata)
    provider = tree.find("Metadata/GlobalConfig/ProviderGlobalParameters")
    for key in ("first_half_start", "first_half_end"):
        parameter = ET.SubElement(provider, "ProviderParameter")
        ET.SubElement(parameter, "Name").text = key
        ET.SubElement(parameter, "Value")
    tree.write(metadata)
    assert load_atd_metadata(metadata).periods == ()


def test_one_empty_period_boundary_still_rejects_incomplete_interval(atd_factory):
    metadata, _, _, _ = atd_factory(n=4, periods=[(0, 4)])
    tree = ET.parse(metadata)
    for parameter in tree.findall("Metadata/GlobalConfig/ProviderGlobalParameters/ProviderParameter"):
        if parameter.findtext("Name") == "first_half_end":
            parameter.find("Value").text = None
    tree.write(metadata)
    with pytest.raises(ValueError, match="incomplete first_half metadata boundaries"):
        load_atd_metadata(metadata)


@pytest.mark.parametrize("start,end,expected", [
    ("0", "4", [(0, 4)]),
    ("0", "0", []),
    ("1.0", "4.0", [(0, 4)]),
    ("1.5", "3.5", [(1, 3)]),
    ("1e0", "4e0", [(0, 4)]),
])
def test_numeric_provider_period_counters_intersect_observed_frame_ids(atd_factory, start, end, expected):
    metadata, _, _, _ = atd_factory(n=4, periods=[(0, 4)])
    tree = ET.parse(metadata)
    for parameter in tree.findall("Metadata/GlobalConfig/ProviderGlobalParameters/ProviderParameter"):
        name = parameter.findtext("Name")
        if name == "first_half_start":
            parameter.find("Value").text = start
        elif name == "first_half_end":
            parameter.find("Value").text = end
    tree.write(metadata)
    assert [(p.start_frame, p.end_frame) for p in load_atd_metadata(metadata).periods] == expected


@pytest.mark.parametrize("start,end", [("NaN", "4"), ("0", "Infinity"), ("-1", "4"), ("4", "2")])
def test_invalid_provider_period_values_are_still_rejected(atd_factory, start, end):
    metadata, _, _, _ = atd_factory(n=4, periods=[(0, 4)])
    tree = ET.parse(metadata)
    for parameter in tree.findall("Metadata/GlobalConfig/ProviderGlobalParameters/ProviderParameter"):
        if parameter.findtext("Name") == "first_half_start":
            parameter.find("Value").text = start
        elif parameter.findtext("Name") == "first_half_end":
            parameter.find("Value").text = end
    tree.write(metadata)
    with pytest.raises(ValueError, match="invalid first_half metadata boundaries"):
        load_atd_metadata(metadata)


def test_metric_metadata_uses_declared_units(atd_factory):
    metadata, raw, players, ball = atd_factory(n=4, normalized=False)
    data = load_atd(metadata, raw)
    assert not data.metadata.normalized
    np.testing.assert_allclose(data.player_xy, players)
    np.testing.assert_allclose(data.ball_xy, ball)


@pytest.mark.parametrize("replacement", ["", "0:0,0,0;0,0,0;0,0,0:0,0,0\n", "1.5:0:0\n",
    "2:0,0,0:0,0,0\n", "1:bad,0,0;0,0,0;0,0,0:0,0,0\n",
    "1:inf,0,0;0,0,0;0,0,0:0,0,0\n", "\n"])
def test_invalid_or_incomplete_records_fail(atd_factory, replacement):
    metadata, raw, _, _ = atd_factory(n=3)
    raw.write_text(replacement, encoding="utf-8")
    with pytest.raises(ValueError):
        load_atd(metadata, raw)


@pytest.mark.parametrize("order", [[0, 0], [1, 0]])
def test_duplicate_and_unordered_frame_ids_fail(atd_factory, order):
    metadata, raw, _, _ = atd_factory(n=3, record_frames=order)
    with pytest.raises(ValueError, match="strictly increasing"):
        load_atd(metadata, raw)


@pytest.mark.parametrize("xpath,attribute,value", [
    ("Metadata/Teams/Team", "id", "away-id"),
    ("Metadata/Players/Player", "teamId", "unrecognized-team"),
    ("Metadata/Players/Player", "id", "slot-2"),
    ("Metadata/PlayerChannels/PlayerChannel", "playerId", "not-a-slot"),
    ("DataFormatSpecifications/DataFormatSpecification", "startFrame", "0"),
])
def test_invalid_metadata_identifiers(atd_factory, xpath, attribute, value):
    metadata, _, _, _ = atd_factory(n=3)
    tree = ET.parse(metadata)
    tree.find(xpath).set(attribute, value)
    tree.write(metadata)
    with pytest.raises(ValueError):
        load_atd_metadata(metadata)


@pytest.mark.parametrize("xpath,value", [
    ("Metadata/GlobalConfig/FrameRate", "0"),
    ("Metadata/Sessions/Session/FieldSize/Width", "NaN"),
    ("Metadata/Sessions/Session/FieldSize/Height", "-1"),
    ("Metadata/Devices/Device/Sensors/Sensor/Channels/Channel/Unit", "centimetres"),
])
def test_invalid_units_dimensions_or_fps(atd_factory, xpath, value):
    metadata, _, _, _ = atd_factory(n=3)
    tree = ET.parse(metadata)
    tree.find(xpath).text = value
    tree.write(metadata)
    with pytest.raises(ValueError):
        load_atd_metadata(metadata)


def test_layout_does_not_silently_skip_uncovered_frames(atd_factory):
    metadata, raw, _, _ = atd_factory(n=3)
    tree = ET.parse(metadata)
    tree.find("DataFormatSpecifications/DataFormatSpecification").set("endFrame", "2")
    tree.write(metadata)
    with pytest.raises(ValueError, match="outside declared"):
        load_atd(metadata, raw)


def test_explicit_score_mapping_cannot_silently_flip_home_and_away(atd_factory):
    metadata, _, _, _ = atd_factory(n=3)
    tree = ET.parse(metadata)
    parameters = ET.SubElement(tree.find("Metadata/Sessions/Session"), "MatchParameters")
    ET.SubElement(parameters, "Score", idLocalTeam="away-id", idVisitingTeam="home-id")
    tree.write(metadata)
    with pytest.raises(ValueError, match="contradicts"):
        load_atd_metadata(metadata)


def test_large_synthetic_input_crosses_loader_buffer_boundary(atd_factory):
    metadata, raw, _, ball = atd_factory(n=4100)
    data = load_atd(metadata, raw)
    assert len(data.frame) == 4100 and data.frame[-1] == 4099
    np.testing.assert_allclose(data.ball_xy, ball)


@pytest.mark.parametrize("text", ["<broken>", '<!DOCTYPE a [<!ENTITY b "bad">]><a>&b;</a>'])
def test_bad_xml_is_rejected(tmp_path, text):
    path = tmp_path / "invented.xml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_atd_metadata(path)
