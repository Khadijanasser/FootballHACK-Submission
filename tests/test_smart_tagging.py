"""Invented SportsCode phases, with no match-specific tagging or outputs."""
import numpy as np
import pytest

from footballhack.io import TaggingConfig, load_smart_tagging


@pytest.mark.parametrize("source,expected", TaggingConfig().set_piece_types)
def test_recognized_set_piece_is_context_not_dead(tagging_factory, source, expected):
    path = tagging_factory([dict(start=5, end=17, code=" SeT   PiEcEs ",
                                 labels=[("Type", source.swapcase()), ("Team", "N/A")])])
    data = load_smart_tagging(path)
    assert [(t.kind, t.start_frame, t.end_frame) for t in data.tags] == [(expected, 5, 17), ("restart", 5, 17)]
    assert data.instances[0].team is None and data.instances[0].issue is None


@pytest.mark.parametrize("source", TaggingConfig().live_codes)
def test_known_active_phases_support_live_only(tagging_factory, source):
    data = load_smart_tagging(tagging_factory([dict(start=0, end=10, code=source.lower())]))
    assert len(data.tags) == 1 and data.tags[0].kind == "live_phase"


@pytest.mark.parametrize("source", ["GOALS", "BALL IN THE BOX", "PLAYERS IN THE BOX", "LONG BALL", "mystery"])
def test_spatial_goal_and_unknown_codes_remain_unknown(tagging_factory, source):
    data = load_smart_tagging(tagging_factory([dict(start=0, end=10, code=source)]))
    assert data.tags[0].kind == "unknown" and data.instances[0].issue == "unknown_code"


def test_ambiguous_type_labels_never_use_last_value_as_truth(tagging_factory):
    labels = [("Type", "Throw-in"), ("Type", "Free Kick"), ("Team", "Synthetic Home"), ("Team", "Synthetic Away")]
    path = tagging_factory([dict(start=0, end=10, code="SET PIECES", labels=labels)])
    data = load_smart_tagging(path, team_map={"Synthetic Home": "H", "Synthetic Away": "A"})
    assert len(data.tags) == 1 and data.tags[0].kind == "unknown"
    assert data.instances[0].issue == "ambiguous_type" and data.instances[0].team is None
    assert data.instances[0].labels == tuple(labels)


def test_duplicate_agreeing_labels_namespaces_file_order_and_explicit_team_map(tagging_factory):
    path = tagging_factory([
        dict(start=20, end=25, code="BUILD UP", labels=[("Team", "Synthetic Home"), ("Half", "2nd Half")]),
        dict(start=0, end=10, code="SET PIECES", labels=[("Type", "Throw-in"), ("type", "throw-IN")]),
    ], namespace=True)
    data = load_smart_tagging(path, team_map={"Synthetic Home": "H"})
    assert data.instances[0].start_frame == 20 and data.instances[1].kind == "throw_in"
    assert data.instances[0].team == "H" and data.instances[0].half == 2
    assert load_smart_tagging(path).instances[0].team is None


def test_seconds_rounding_matches_public_loader_half_open_contract(tagging_factory):
    path = tagging_factory([dict(start_s=.125, end_s=.375, code="BUILD UP")])
    tag = load_smart_tagging(path, fps=20).tags[0]
    assert (tag.start_frame, tag.end_frame) == (2, 8)  # Python nearest-even rounding


def test_unknown_set_piece_type_does_not_generate_restart(tagging_factory):
    data = load_smart_tagging(tagging_factory([dict(start=0, end=10, code="SET PIECES", labels=[("Type", "Unknown")])]))
    assert len(data.tags) == 1 and data.tags[0].kind == "unknown"


def test_configurable_explicit_vocab_and_confidence(tagging_factory):
    cfg = TaggingConfig(confidence=.8, live_codes=("CUSTOM ACTIVE",), set_piece_types=(("Penalty", "penalty"),))
    path = tagging_factory([dict(start=0, end=10, code="CUSTOM ACTIVE"),
                            dict(start=10, end=20, code="SET PIECES", labels=[("Type", "Penalty")])])
    data = load_smart_tagging(path, config=cfg)
    assert [t.kind for t in data.tags] == ["live_phase", "penalty", "restart"]
    assert all(t.confidence == .8 for t in data.tags)


@pytest.mark.parametrize("start,end", [(-1, 1), (1, 1), (2, 1), (np.nan, 1), (0, np.inf), (0, .00001)])
def test_invalid_or_collapsed_intervals(tagging_factory, start, end):
    path = tagging_factory([dict(start_s=start, end_s=end, code="BUILD UP")])
    with pytest.raises(ValueError, match="instance 1"):
        load_smart_tagging(path)


@pytest.mark.parametrize("fps", [0, -1, np.nan, np.inf, True])
def test_invalid_fps(tagging_factory, fps):
    with pytest.raises(ValueError):
        load_smart_tagging(tagging_factory([dict(start=0, end=10)]), fps=fps)


@pytest.mark.parametrize("kwargs", [
    {"confidence": 1.1}, {"confidence": np.nan}, {"confidence": True},
    {"live_codes": ("SET PIECES",)}, {"live_codes": ("",)},
    {"set_piece_types": (("Thing", "DEAD"),)},
    {"set_piece_types": (("Throw-in", "throw_in"), ("throw-IN", "corner"))},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        TaggingConfig(**kwargs)


def test_empty_tagging_requires_explicit_omission(tagging_factory):
    with pytest.raises(ValueError, match="no instances"):
        load_smart_tagging(tagging_factory([]))
