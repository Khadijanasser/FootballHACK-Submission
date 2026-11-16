"""Metadata-only split assignment, seed 2027, with no reference/metric inputs."""
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

SEED = 2027
PARTITIONS = ('development', 'validation', 'holdout')
CANONICAL_IDS = ('J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WPY', 'J03WQQ', 'J03WR9')


def assign_splits(records):
    """Minimise club/partition-pair overlap, then seeded hash tie-break.

    Fixed league allocation: dev 1+2, validation 0+2, holdout 1+1 (BL1+BL2).
    Seven matches cannot represent two BL1 matches in three partitions. Hashing
    avoids version-dependent PRNG shuffles. No match results/labels considered.
    """
    records = sorted(records, key=lambda r: r['match_id'])
    if len(records) != 7 or len({r['match_id'] for r in records}) != 7:
        raise ValueError("seven unique matches required")
    ids = tuple(r['match_id'] for r in records)
    by_id = {r['match_id']: r for r in records}
    candidates = []
    for dev in combinations(ids, 3):
        remaining = sorted(set(ids) - set(dev))
        for val in combinations(remaining, 2):
            holdout = tuple(sorted(set(remaining) - set(val)))
            groups = (dev, val, holdout)
            if tuple(sum(by_id[m]['competition_id'] == 'DFL-COM-000001' for m in g) for g in groups) != (1, 0, 1):
                continue
            clubs = [set(t for m in g for t in by_id[m]['team_ids']) for g in groups]
            overlap = sum(len(clubs[i] & clubs[j]) for i in range(3) for j in range(i + 1, 3))
            tie = sha256(f'{SEED}:{json.dumps(groups, separators=(",", ":"))}'.encode()).hexdigest()
            candidates.append((overlap, tie, groups))
    if not candidates:
        raise ValueError("metadata cannot satisfy predefined league allocation")
    _, _, groups = min(candidates)
    return {name: list(group) for name, group in zip(PARTITIONS, groups)}


def club_overlap(records, assignments):
    clubs = {p: {team for r in records if r['match_id'] in ids for team in r['team_ids']}
             for p, ids in assignments.items()}
    return {f'{a}/{b}': sorted(clubs[a] & clubs[b])
            for a, b in combinations(PARTITIONS, 2)}


def load_manifest():
    return json.loads(Path(__file__).with_name('sportec_split.json').read_text())


def partition_matches(partition):
    # Deliberately no holdout release switch in this milestone.
    if partition not in ('development', 'validation'):
        raise ValueError("holdout is sealed; only development/validation may be opened")
    manifest = load_manifest()
    ids = set(manifest['assignments'][partition])
    return tuple(r for r in manifest['matches'] if r['match_id'] in ids)
