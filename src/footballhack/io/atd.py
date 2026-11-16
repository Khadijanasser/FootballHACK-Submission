"""Metrica tactical FIFA/EPTS XML + text, from public format/source contracts.

Raw records: frame_id:player_x,player_y,speed;...:ball_x,ball_y,speed.
Player arrays always follow metadata Player order, never numeric-ID sorting.
Optional EPTS channel references are honored, including changed layouts.
"""
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._xml import integer, positive_number, read_xml, required_text


@dataclass(frozen=True)
class Period:
    """Explicit tactical VIDEO-frame interval [start_frame, end_frame)."""
    id: int
    start_frame: int
    end_frame: int

    def __post_init__(self):
        for name, minimum in (("id", 1), ("start_frame", 0), ("end_frame", 1)):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
                raise ValueError(f"period {name} must be an integer")
            integer(value, name, minimum)
        if self.end_frame <= self.start_frame:
            raise ValueError("period must be nonempty")


@dataclass(frozen=True)
class TrackSlot:
    """An export slot, not a player identity or real shirt number."""
    id: str
    team_id: str
    side: str
    name: str


@dataclass(frozen=True)
class _Register:
    separator: str | None = None
    children: tuple = ()
    target: tuple | str | None = None


@dataclass(frozen=True)
class _Layout:
    start: int
    end: int
    register: _Register


@dataclass(frozen=True)
class ATDMetadata:
    team_ids: tuple[str, str]
    team_names: tuple[str, str]
    slots: tuple[TrackSlot, ...]
    fps: float
    pitch_m: tuple[float, float]
    normalized: bool
    periods: tuple[Period, ...] = ()
    layouts: tuple[_Layout, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class TrackingData:
    frame: NDArray
    player_xy: NDArray
    ball_xy: NDArray
    teams: NDArray
    metadata: ATDMetadata

    @property
    def track_ids(self):
        return tuple(slot.id for slot in self.metadata.slots)


def _identifier(element, name):
    value = element.get(name)
    if value is None or not value.strip():
        raise ValueError(f"missing {element.tag}.{name}")
    return value.strip()


def _metadata_periods(meta):
    period_names = ("first_half", "second_half", "first_extra_half", "second_extra_half")
    boundary_keys = {f"{name}_{suffix}" for name in period_names for suffix in ("start", "end")}
    values = {}
    for node in meta.findall("GlobalConfig/ProviderGlobalParameters/ProviderParameter"):
        key = node.findtext("Name", default="").strip()
        if key not in boundary_keys:
            continue
        # Provider parameters are optional. The public Kloppy EPTS reader
        # ignores empty Values; unrelated or unset export fields are not errors.
        value = node.findtext("Value", default="").strip()
        if not value:
            continue
        if key in values and values[key] != value:
            raise ValueError(f"contradictory provider parameter {key}")
        values[key] = value
    periods = []
    for i, name in enumerate(period_names, 1):
        start, end = values.get(f"{name}_start"), values.get(f"{name}_end")
        if start is None and end is None:
            continue
        if start is None or end is None:
            raise ValueError(f"incomplete {name} metadata boundaries")
        # Public EPTS readers accept numeric provider counters, including zero
        # and decimal notation. Intersect inclusive bounds with observable raw
        # frame IDs 1,2,... before converting to half-open video intervals.
        try:
            lower, upper = Decimal(start), Decimal(end)
        except InvalidOperation as error:
            raise ValueError(f"invalid {name} metadata boundaries") from error
        if not (lower.is_finite() and upper.is_finite()) or lower < 0 or upper < lower:
            raise ValueError(f"invalid {name} metadata boundaries")
        first_raw = integer(str(max(1, int(lower.to_integral_value(rounding=ROUND_CEILING)))), name, 1)
        last_raw = integer(str(int(upper.to_integral_value(rounding=ROUND_FLOOR))), name, 0)
        if last_raw < first_raw:
            continue  # An empty/zero placeholder covers no tracking observations.
        periods.append(Period(i, first_raw - 1, last_raw))
    return tuple(periods)


def _layouts(root, meta, slot_indices):
    channels = {}
    for ref in meta.findall("PlayerChannels/PlayerChannel"):
        ref_id = _identifier(ref, "id")
        player_id, channel = _identifier(ref, "playerId"), _identifier(ref, "channelId")
        if ref_id in channels or player_id not in slot_indices:
            raise ValueError("duplicate channel reference or unknown playerId")
        channels[ref_id] = (slot_indices[player_id], channel.casefold())

    def compile_register(node):
        if node.tag in ("DataFormatSpecification", "SplitRegister"):
            separator = _identifier(node, "separator")
            if len(separator) != 1 or separator not in ":;,":
                raise ValueError("unsupported EPTS register separator")
            children = tuple(compile_register(child) for child in node)
            if not children:
                raise ValueError("empty EPTS split register")
            return _Register(separator, children)
        if node.tag == "StringRegister":
            name = _identifier(node, "name").casefold()
            return _Register(target="frame" if name in ("framecount", "frame_id", "frameid") else None)
        if node.tag == "PlayerChannelRef":
            key = _identifier(node, "playerChannelId")
            if key not in channels:
                raise ValueError(f"unknown playerChannelId {key}")
            slot, channel = channels[key]
            return _Register(target=(slot, channel))
        if node.tag == "BallChannelRef":
            return _Register(target=(-1, _identifier(node, "channelId").casefold()))
        raise ValueError(f"unsupported EPTS register {node.tag}")

    result = []
    for spec in root.findall("DataFormatSpecifications/DataFormatSpecification"):
        start = integer(spec.get("startFrame"), "layout.startFrame", 1)
        end = integer(spec.get("endFrame"), "layout.endFrame", 1)
        if end < start or (result and start <= result[-1].end):
            raise ValueError("invalid, overlapping or unordered EPTS layouts")
        register = compile_register(spec)
        targets = []

        def visit(node):
            if node.children:
                for child in node.children:
                    visit(child)
            elif node.target is not None:
                targets.append(node.target)

        visit(register)
        required = {"frame"} | {(slot, axis) for slot in range(-1, len(slot_indices)) for axis in ("x", "y")}
        if not required.issubset(set(targets)) or len(set(targets)) != len(targets):
            raise ValueError("EPTS layout needs unique frame/player/ball coordinate references")
        # Layout selection must be possible before recursive field parsing.
        if register.separator != ":" or register.children[0].target != "frame":
            raise ValueError("expected tactical ATD frame prefix followed by ':'")
        result.append(_Layout(start, end, register))
    return tuple(result)


def load_atd_metadata(path) -> ATDMetadata:
    root = read_xml(path)
    meta = root if root.tag == "Metadata" else root.find("Metadata")
    if meta is None:
        raise ValueError("missing EPTS Metadata")
    teams = meta.findall("Teams/Team")
    if len(teams) != 2:
        raise ValueError("ATD metadata must contain exactly two teams")
    team_ids = tuple(_identifier(team, "id") for team in teams)
    if len(set(team_ids)) != 2:
        raise ValueError("duplicate team IDs")
    names = tuple(required_text(team, "Name") for team in teams)
    score = meta.find("Sessions/Session/MatchParameters/Score")
    if score is not None and (score.get("idLocalTeam"), score.get("idVisitingTeam")) != team_ids:
        raise ValueError("explicit home/away mapping contradicts metadata team order")
    slots, seen = [], set()
    for player in meta.findall("Players/Player"):
        slot_id, team_id = _identifier(player, "id"), _identifier(player, "teamId")
        if slot_id in seen or team_id not in team_ids:
            raise ValueError("duplicate slot ID or unknown teamId")
        seen.add(slot_id)
        slots.append(TrackSlot(slot_id, team_id, "H" if team_id == team_ids[0] else "A",
                               player.findtext("Name", default="").strip()))
    if not slots:
        raise ValueError("ATD metadata contains no track slots")
    fps = positive_number(required_text(meta, "GlobalConfig/FrameRate"), "FrameRate")
    length = positive_number(required_text(meta, "Sessions/Session/FieldSize/Width"), "FieldSize.Width")
    width = positive_number(required_text(meta, "Sessions/Session/FieldSize/Height"), "FieldSize.Height")
    units = []
    for sensor in meta.findall("Devices/Device/Sensors/Sensor"):
        if sensor.get("id", "").casefold() == "position":
            for channel in sensor.findall("Channels/Channel"):
                if channel.get("id", "").casefold() in ("x", "y"):
                    units.append(required_text(channel, "Unit").casefold())
    # The organizers' documented raw format is normalized. If sensors declare
    # units, honor them and reject mixed/unknown coordinate units explicitly.
    if not units or all(unit == "normalized" for unit in units):
        normalized = True
    elif all(unit in ("m", "meter", "metre", "meters", "metres") for unit in units):
        normalized = False
    else:
        raise ValueError("unsupported or mixed position units")
    layouts = _layouts(root, meta, {slot.id: i for i, slot in enumerate(slots)})
    return ATDMetadata(team_ids, names, tuple(slots), fps, (length, width), normalized,
                       _metadata_periods(meta), layouts)


def _number(text):
    text = text.strip()
    if not text or text.casefold() == "nan":
        return np.nan
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError("invalid numeric ATD observation") from error
    if not np.isfinite(value):
        raise ValueError("ATD observations must be finite or NaN")
    return value


def _parse_register(register, text, players, ball, raw_frame):
    if register.separator is not None:
        parts = text.split(register.separator)
        if len(parts) == len(register.children) + 1 and parts[-1] == "":
            parts.pop()  # EPTS permits a trailing separator at each nesting level.
        if len(parts) != len(register.children):
            raise ValueError("ATD field count does not match EPTS layout")
        for child, part in zip(register.children, parts):
            _parse_register(child, part, players, ball, raw_frame)
    elif register.target == "frame":
        if integer(text, "frame_id", 1) != raw_frame:
            raise ValueError("inconsistent EPTS frame register")
    elif isinstance(register.target, tuple):
        slot, channel = register.target
        value = _number(text)
        if channel in ("x", "y"):
            (ball if slot == -1 else players[slot])[0 if channel == "x" else 1] = value
    # Other StringRegisters are descriptive and are not interpreted as coordinates.


def _default_register(n_slots):
    def triplet(slot):
        return _Register(",", tuple(_Register(target=(slot, axis)) for axis in ("x", "y", "s")))
    return _Register(":", (_Register(target="frame"),
                           _Register(";", tuple(triplet(i) for i in range(n_slots))), triplet(-1)))


def load_atd(metadata_path, raw_path) -> TrackingData:
    """Read only these explicit paths; retain gaps, partial NaNs and slot order.

    Chunked buffers bound Python-object overhead. Supplied speed fields are
    validated but not used; the existing feature module derives causal speeds.
    """
    meta = load_atd_metadata(metadata_path)
    n_slots, capacity = len(meta.slots), 4096
    frame_chunks, player_chunks, ball_chunks = [], [], []
    frames = np.empty(capacity, dtype=np.int64)
    players = np.empty((capacity, n_slots, 2))
    ball = np.empty((capacity, 2))
    used, last, layout_index = 0, 0, 0
    fallback = _default_register(n_slots)
    with Path(raw_path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            try:
                if not line or ":" not in line:
                    raise ValueError("expected frame:players:ball record")
                frame = integer(line.split(":", 1)[0], "frame_id", 1)
                if frame <= last:
                    raise ValueError("frame IDs must be strictly increasing and unique")
                register = fallback
                if meta.layouts:
                    while layout_index < len(meta.layouts) and frame > meta.layouts[layout_index].end:
                        layout_index += 1
                    if layout_index == len(meta.layouts) or frame < meta.layouts[layout_index].start:
                        raise ValueError("frame is outside declared EPTS layouts")
                    register = meta.layouts[layout_index].register
                players[used].fill(np.nan)
                ball[used].fill(np.nan)
                _parse_register(register, line, players[used], ball[used], frame)
                frames[used], last = frame - 1, frame
                used += 1
            except ValueError as error:
                # Do not reproduce tracking lines or coordinates in error messages.
                raise ValueError(f"ATD line {line_number}: {error}") from error
            if used == capacity:
                frame_chunks.append(frames)
                player_chunks.append(players)
                ball_chunks.append(ball)
                frames = np.empty(capacity, dtype=np.int64)
                players = np.empty((capacity, n_slots, 2))
                ball = np.empty((capacity, 2))
                used = 0
    if used:
        frame_chunks.append(frames[:used].copy())
        player_chunks.append(players[:used].copy())
        ball_chunks.append(ball[:used].copy())
    if not frame_chunks:
        raise ValueError("ATD raw file contains no records")
    return TrackingData(np.concatenate(frame_chunks), np.concatenate(player_chunks),
                         np.concatenate(ball_chunks), np.array([slot.side for slot in meta.slots]), meta)
