"""Sportec/DFL XML to prediction-only arrays. No possession/status inputs.

Coordinates are provider centre-origin metres, converted to corner-origin
metres without rotation or clipping. Native N and UTC T define a separate dense
grid for each half. No interpolation, alive filtering, speed fields or events.
"""
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ._xml import integer, positive_number, read_xml
from .atd import ATDMetadata, Period, TrackSlot, TrackingData

SECTIONS = {"firstHalf": 1, "secondHalf": 2}
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class SportecMetadata:
    match_id: str
    competition_id: str
    season: str
    tracking: ATDMetadata


@dataclass(frozen=True)
class SportecHalf:
    match_id: str
    section: str
    native_frame: np.ndarray
    timestamp_us: np.ndarray
    tracking: TrackingData


def _required(node, name):
    value = node.get(name) if node is not None else None
    if not value or not value.strip():
        raise ValueError(f"missing Sportec {name}")
    return value.strip()


def load_sportec_metadata(path) -> SportecMetadata:
    root = read_xml(path)
    match = root.find('MatchInformation')
    if match is None:
        raise ValueError("missing MatchInformation")
    general, environment = match.find('General'), match.find('Environment')
    home = _required(general, 'HomeTeamId')
    away = general.get('AwayTeamId') or general.get('GuestTeamId')
    if not away or away == home:
        raise ValueError("two distinct team IDs required")
    if general.get('AwayTeamId') and general.get('GuestTeamId') not in (None, away):
        raise ValueError("conflicting guest/away team IDs")
    pitch = tuple(positive_number(_required(environment, k), k) for k in ('PitchX', 'PitchY'))
    teams = match.findall('Teams/Team')
    if len(teams) != 2 or {t.get('TeamId') for t in teams} != {home, away}:
        raise ValueError("teams do not match home/away metadata")
    slots, names, people = [], [], set()
    for side, team_id in zip(('H', 'A'), (home, away)):
        team = next(t for t in teams if t.get('TeamId') == team_id)
        allowed_roles = (None, 'home') if side == 'H' else (None, 'guest', 'away')
        if team.get('Role') not in allowed_roles:
            raise ValueError("team Role contradicts home/away IDs")
        names.append(_required(team, 'TeamName'))
        players = team.findall('Players/Player')
        if not players:
            raise ValueError("empty team roster")
        for player in players:
            person = _required(player, 'PersonId')
            if person in people:
                raise ValueError("duplicate PersonId in roster")
            people.add(person)
            slots.append(TrackSlot(person, team_id, side, player.get('Shortname', person)))
    return SportecMetadata(_required(general, 'MatchId'), _required(general, 'CompetitionId'),
                           general.get('Season', ''), ATDMetadata(
                               (home, away), tuple(names), tuple(slots), 25.0, pitch, False))


def _timestamp(value):
    try:
        stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if stamp.utcoffset() is None:
            raise ValueError("timezone required")
        delta = stamp.astimezone(timezone.utc) - EPOCH
        return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
    except (ValueError, AttributeError, OverflowError) as error:
        raise ValueError("invalid timezone-aware Sportec T timestamp") from error


def iter_sportec_frames(path, *, header_check=None):
    """Bounded XML tree memory; reject declarations, even across read boundaries.

    Internal format reader shared with evaluation-only references. Consumers
    select their own fields; prediction never interprets label attributes.
    """
    with Path(path).open('rb') as source:
        tail = b''
        while block := source.read(65536):
            scan = tail + block.replace(b'\x00', b'').upper()
            if b'<!DOCTYPE' in scan or b'<!ENTITY' in scan:
                raise ValueError("DTD/entity declarations are not supported")
            tail = scan[-16:]
    stack, context = [], None
    try:
        for event, element in ET.iterparse(path, events=('start', 'end')):
            name = element.tag.rsplit('}', 1)[-1]
            if event == 'start':
                stack.append(element)
                if header_check is not None and name in ('MetaData', 'PitchSize'):
                    header_check(name, element.attrib)
                if name == 'FrameSet':
                    if context is not None:
                        raise ValueError("nested FrameSet")
                    context = dict(element.attrib)
                continue
            if name == 'Frame':
                if context is None:
                    raise ValueError("Frame outside FrameSet")
                yield context, element.attrib
            if name == 'FrameSet':
                context = None
            stack.pop()
            if stack:
                stack[-1].remove(element)
            element.clear()
    except ET.ParseError as error:
        raise ValueError(f"invalid Sportec XML: {error}") from error


def frame_key(context, attrs, match_id):
    if context.get('MatchId') != match_id:
        raise ValueError("tracking MatchId disagrees with metadata")
    section = context.get('GameSection')
    if section not in SECTIONS:
        raise ValueError("only explicit firstHalf/secondHalf supported")
    return section, integer(attrs.get('N'), 'N'), _timestamp(attrs.get('T'))


def _coordinate(value):
    if value is None or not value.strip():
        return np.nan
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError("invalid Sportec coordinate") from error
    if np.isinf(number):
        raise ValueError("infinite Sportec coordinate")
    return number


def load_sportec_tracking(path, metadata: SportecMetadata, *, max_half_frames=250_000):
    """Read positions only; validate all clocks against the documented 25 Hz.

    Unknown third-party objects (e.g. referees) are ignored. A participating
    team's unknown player is an error. Metadata roster order is retained within
    explicit H then A teams; shirts are never used as identities.
    """
    if not isinstance(metadata, SportecMetadata):
        raise TypeError("SportecMetadata required")
    integer(max_half_frames, 'max_half_frames', 1)
    meta = metadata.tracking
    def check_header(name, attrs):
        if name == 'MetaData' and attrs.get('MatchId') not in (None, metadata.match_id):
            raise ValueError('tracking header MatchId disagrees with metadata')
        if name == 'PitchSize':
            dimensions = tuple(positive_number(attrs.get(k), f'PitchSize.{k}') for k in ('X', 'Y'))
            if dimensions != meta.pitch_m:
                raise ValueError('tracking PitchSize disagrees with match metadata')
    return _tracking_from_frames(iter_sportec_frames(path, header_check=check_header),
                                 metadata, max_half_frames)


def _tracking_from_frames(frames, metadata, max_half_frames):
    """Position-only construction shared with the evaluation leakage audit."""
    meta = metadata.tracking
    slots = {(s.team_id, s.id): i for i, s in enumerate(meta.slots)}
    records, clocks, ball_seen, bounds = {}, {}, set(), {}
    for context, attrs in frames:
        team = context.get('TeamId', '')
        ball = team.casefold() == 'ball'
        if not ball and team not in meta.team_ids:
            continue
        slot = -1 if ball else slots.get((team, context.get('PersonId')))
        if slot is None:
            raise ValueError("unknown participating player or wrong team mapping")
        section, native, stamp = frame_key(context, attrs, metadata.match_id)
        clock = clocks.setdefault(section, {})
        if native in clock and clock[native] != stamp:
            raise ValueError("objects disagree on frame timestamp")
        clock[native] = stamp
        lower, upper = bounds.get(section, (native, native))
        bounds[section] = min(lower, native), max(upper, native)
        if bounds[section][1] - bounds[section][0] + 1 > max_half_frames:
            raise ValueError("half exceeds configured frame limit")
        object_records = records.setdefault((section, slot), {})
        if native in object_records:
            raise ValueError("duplicate object/frame observation")
        object_records[native] = (_coordinate(attrs.get('X')), _coordinate(attrs.get('Y')))
        if ball:
            ball_seen.add(section)
    if set(clocks) != set(SECTIONS) or ball_seen != set(SECTIONS):
        raise ValueError("both halves and ball FrameSets are required")
    halves = []
    previous_end = None
    offset = np.array(meta.pitch_m) / 2
    for section, period_id in SECTIONS.items():
        clock = clocks[section]
        first, last = min(clock), max(clock)
        if len(clock) < 2:
            raise ValueError("at least two frame timestamps per half required to verify FPS")
        anchor = clock[first]
        if any(t - anchor != (n - first) * 40_000 for n, t in clock.items()):
            raise ValueError("native frame/timestamp clock is not exactly 25 Hz")
        if previous_end is not None and anchor <= previous_end:
            raise ValueError("halves overlap or reverse in absolute time")
        previous_end = clock[last]
        n_frames = last - first + 1
        native = np.arange(first, last + 1, dtype=np.int64)
        timestamps = anchor + np.arange(n_frames, dtype=np.int64) * 40_000
        players = np.full((n_frames, len(meta.slots), 2), np.nan)
        ball_xy = np.full((n_frames, 2), np.nan)
        for slot in range(-1, len(meta.slots)):
            observations = records.get((section, slot), {})
            if observations:
                indices = np.fromiter(observations, dtype=np.int64) - first
                positions = np.array(list(observations.values())) + offset
                if slot == -1:
                    ball_xy[indices] = positions
                else:
                    players[indices, slot] = positions
        tracking = TrackingData(np.arange(n_frames, dtype=np.int64), players, ball_xy,
                                np.array([s.side for s in meta.slots]), replace(
                                    meta, periods=(Period(period_id, 0, n_frames),)))
        halves.append(SportecHalf(metadata.match_id, section, native, timestamps, tracking))
    return tuple(halves)
