"""SportsCode intervals on the tactical video clock; no possession labels."""
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from ..dead_ball import SET_PIECES, SmartTag
from ._xml import positive_number, read_xml, required_text


def _key(value):
    return " ".join(value.strip().casefold().split())


@dataclass(frozen=True)
class TaggingConfig:
    """Explicit vocabulary from public DATA.md; spatial/goal codes stay unknown."""
    confidence: float = 1.0
    live_codes: tuple[str, ...] = (
        "BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
        "DEFENSIVE TRANSITION", "DEFENDING IN DEFENSIVE THIRD",
        "DEFENDING IN MIDDLE THIRD", "DEFENDING IN ATTACKING THIRD",
    )
    set_piece_types: tuple[tuple[str, str], ...] = (
        ("Throw-in", "throw_in"), ("Corner Kick", "corner"),
        ("Goal Kick", "goal_kick"), ("Free Kick", "free_kick"),
        ("Kick Off (After Goal)", "kick_off"), ("Kick Off (Start)", "kick_off"),
    )

    def __post_init__(self):
        if (isinstance(self.confidence, (bool, np.bool_))
                or not isinstance(self.confidence, (int, float))
                or not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1):
            raise ValueError("tag confidence must be finite in [0, 1]")
        if (not isinstance(self.live_codes, tuple)
                or not all(isinstance(code, str) and _key(code) for code in self.live_codes)):
            raise ValueError("live_codes must be a tuple of nonempty strings")
        if "set pieces" in {_key(code) for code in self.live_codes}:
            raise ValueError("SET PIECES cannot be a live phase code")
        if not isinstance(self.set_piece_types, tuple):
            raise ValueError("set_piece_types must be a tuple of source/kind pairs")
        seen = set()
        for pair in self.set_piece_types:
            if (not isinstance(pair, tuple) or len(pair) != 2
                    or not isinstance(pair[0], str) or not _key(pair[0])
                    or pair[1] not in SET_PIECES or _key(pair[0]) in seen):
                raise ValueError("invalid or duplicate set_piece_types mapping")
            seen.add(_key(pair[0]))


@dataclass(frozen=True)
class TaggingInstance:
    start_frame: int
    end_frame: int
    code: str
    labels: tuple[tuple[str, str], ...]
    kind: str
    team: str | None
    half: int | None
    issue: str | None


@dataclass(frozen=True)
class TaggingData:
    tags: tuple[SmartTag, ...]
    instances: tuple[TaggingInstance, ...]


def load_smart_tagging(
    path: str | Path, *, fps: float = 25.0, config: TaggingConfig | None = None,
    team_map: Mapping[str, str] | None = None,
) -> TaggingData:
    """Read file-order instances, retaining unknown and ambiguous source labels.

    A recognized set piece emits its canonical tag AND a restart-context tag on
    the same interval. Neither asserts DEAD or the precise restart instant.
    Team names require an explicit mapping and never select temporal ownership.
    """
    fps = positive_number(fps, "fps")
    cfg = config if config is not None else TaggingConfig()
    if not isinstance(cfg, TaggingConfig):
        raise TypeError("config must be TaggingConfig")
    mapping = {} if team_map is None else dict(team_map)
    if any(not isinstance(k, str) or v not in ("H", "A") for k, v in mapping.items()):
        raise ValueError("team_map must explicitly map source names to H/A")
    mapping = {_key(k): v for k, v in mapping.items()}
    types = {_key(source): kind for source, kind in cfg.set_piece_types}
    live_codes = {_key(code) for code in cfg.live_codes}
    root = read_xml(path)
    tags, instances = [], []
    for index, node in enumerate(root.iter("instance"), 1):
        try:
            start_s, end_s = float(required_text(node, "start")), float(required_text(node, "end"))
            if not (np.isfinite(start_s) and np.isfinite(end_s) and 0 <= start_s < end_s):
                raise ValueError("expected finite nonnegative start < end")
            start, end = int(round(start_s * fps)), int(round(end_s * fps))
            if start >= end:
                raise ValueError("interval collapses after frame rounding")
            code = node.findtext("code", default="").strip()
            labels = tuple((required_text(label, "group"), required_text(label, "text"))
                           for label in node.findall("label"))
            groups = {}
            for group, text in labels:
                groups.setdefault(_key(group), set()).add(_key(text))

            def unique(group):
                values = groups.get(group, set())
                return next(iter(values)) if len(values) == 1 else None

            kind, issue = "unknown", "unknown_code"
            if _key(code) == "set pieces":
                kind = types.get(unique("type"), "unknown")
                issue = (None if kind != "unknown" else "ambiguous_type"
                         if len(groups.get("type", ())) > 1 else "unknown_set_piece_type")
            elif _key(code) in live_codes:
                kind, issue = "live_phase", None
            tag = SmartTag(start, end, kind, cfg.confidence)
            tags.append(tag)
            if kind in SET_PIECES:
                tags.append(SmartTag(start, end, "restart", cfg.confidence))
            instances.append(TaggingInstance(start, end, code, labels, kind,
                                             mapping.get(unique("team")),
                                             {"1st half": 1, "2nd half": 2}.get(unique("half")), issue))
        except (ValueError, OverflowError) as error:
            raise ValueError(f"Smart Tagging instance {index}: {error}") from error
    if not instances:
        raise ValueError("Smart Tagging XML contains no instances; omit tagging explicitly if unused")
    return TaggingData(tuple(tags), tuple(instances))
