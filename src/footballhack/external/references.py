"""Evaluation-only provider labels. Prediction modules must not import this."""
from dataclasses import dataclass

import numpy as np

from footballhack.io.sportec import frame_key, iter_sportec_frames


@dataclass(frozen=True)
class ReferenceHalf:
    match_id: str
    section: str
    native_frame: np.ndarray
    labels: np.ndarray
    coverage: dict


def reference_label(status, possession):
    """Strict published categorical fields; invalid/missing codes abstain.

    Dead takes precedence over ownership. No absent-value defaults or imputation.
    """
    if status == '0':
        return 'D'
    if status == '1':
        return {'1': 'H', '2': 'A'}.get(possession, 'X')
    return 'X'


def load_references(path, halves):
    """Align ball reference fields to verified prediction grids, never vice versa."""
    if len(halves) != 2 or len({h.match_id for h in halves}) != 1:
        raise ValueError("two aligned match halves required")
    lookup = {h.section: h for h in halves}
    labels = {s: np.full(len(h.native_frame), 'X', dtype='<U1') for s, h in lookup.items()}
    seen = {s: set() for s in lookup}
    counts = {s: {'missing_status': 0, 'invalid_status': 0,
                  'unknown_live_possession': 0} for s in lookup}
    for context, attrs in iter_sportec_frames(path):
        if context.get('TeamId', '').casefold() != 'ball':
            continue
        section, native, stamp = frame_key(context, attrs, halves[0].match_id)
        if section not in lookup:
            raise ValueError("reference section outside tracking coverage")
        half = lookup[section]
        index = native - int(half.native_frame[0])
        if not 0 <= index < len(half.native_frame) or stamp != half.timestamp_us[index]:
            raise ValueError("reference frame/timestamp misalignment")
        if native in seen[section]:
            raise ValueError("duplicate ball reference frame")
        seen[section].add(native)
        status, possession = attrs.get('BallStatus'), attrs.get('BallPossession')
        labels[section][index] = reference_label(status, possession)
        counts[section]['missing_status'] += status in (None, '')
        counts[section]['invalid_status'] += status not in (None, '', '0', '1')
        counts[section]['unknown_live_possession'] += status == '1' and possession not in ('1', '2')
    result = []
    for half in halves:
        values = labels[half.section]
        coverage = dict(counts[half.section])
        coverage.update(n_frames=len(values), n_ball_records=len(seen[half.section]),
                        missing_ball_records=len(values) - len(seen[half.section]),
                        n_valid=int(np.count_nonzero(values != 'X')),
                        valid_fraction=float(np.mean(values != 'X')),
                        label_counts={k: int(np.count_nonzero(values == k)) for k in 'HADX'})
        result.append(ReferenceHalf(half.match_id, half.section, half.native_frame.copy(), values, coverage))
    return tuple(result)
