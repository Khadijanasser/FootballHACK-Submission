"""Explicit local-only external audit/evaluation commands; holdout stays sealed."""
import argparse
from hashlib import md5, sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import resource
import sys
import time
from urllib.request import urlopen

from footballhack.io.sportec import load_sportec_metadata, load_sportec_tracking
from .evaluation import (aggregate_halves, audit_match, check_native_reference_isolation,
                         evaluate_match, verify_frozen_sources)
from .references import load_references
from .splits import load_manifest, partition_matches


def outside_git(path):
    path = Path(path).expanduser().resolve()
    if any((parent / '.git').exists() for parent in (path, *path.parents)):
        raise ValueError("external datasets and reports must be outside any Git worktree")
    return path


def _installed_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def verify_artifact(path, artifact):
    digest = md5()
    size = 0
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
    if size != artifact['size'] or digest.hexdigest() != artifact['computed_md5']:
        raise ValueError(f"provider artifact checksum/size mismatch: {artifact['name']}")


def fetch_artifact(root, artifact):
    target = root / artifact['name']
    if target.exists():
        verify_artifact(target, artifact)
        return
    temporary = target.with_suffix(target.suffix + '.partial')
    try:
        with urlopen(artifact['download_url'], timeout=60) as source, temporary.open('wb') as dest:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                dest.write(block)
        verify_artifact(temporary, artifact)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def select_matches(partition, match_id=None):
    records = partition_matches(partition)  # rejects holdout before touching paths
    if match_id is not None:
        records = tuple(r for r in records if r['match_id'] == match_id)
        if not records:
            raise ValueError('--match must belong to the requested unsealed partition')
    return records


def run_partition(root, partition, *, evaluate=False, chunk_size=4096, match_id=None):
    records = select_matches(partition, match_id)
    verify_frozen_sources()
    reports = []
    for record in records:
        started = time.perf_counter()
        metadata_path, positions_path = (root / record[k]['name'] for k in ('metadata', 'positions'))
        for key, path in (('metadata', metadata_path), ('positions', positions_path)):
            outside_git(path)
            verify_artifact(path, record[key])
        metadata = load_sportec_metadata(metadata_path)
        if (metadata.match_id != 'DFL-MAT-' + record['match_id']
                or list(metadata.tracking.team_ids) != record['team_ids']
                or list(metadata.tracking.pitch_m) != record['pitch_m']):
            raise ValueError("metadata disagrees with frozen split manifest")
        halves = load_sportec_tracking(positions_path, metadata)
        references = load_references(positions_path, halves)
        reports.append(evaluate_match(halves, references, chunk_size=chunk_size) if evaluate
                       else audit_match(halves, references))
        if not evaluate:
            reports[-1]['reference_isolation'] = check_native_reference_isolation(positions_path, metadata, halves)
        reports[-1]['resources'] = dict(
            elapsed_s=time.perf_counter() - started,
            process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                       * (1 if sys.platform == 'darwin' else 1024)),
            scope='wall time includes checksum verification, loading, references and isolation audit; RSS is process lifetime peak')
    report = dict(partition=partition, evaluated=evaluate, matches=reports,
                  protocol=verify_frozen_sources(),
                  runtime=dict(python=sys.version.split()[0], numpy=_installed_version('numpy'), pandas=_installed_version('pandas')),
                  harness_sha256={str(path.relative_to(Path(__file__).parents[1])): sha256(path.read_bytes()).hexdigest()
                                  for path in sorted(Path(__file__).parent.glob('*.py'))} |
                                  {'io/sportec.py': sha256((Path(__file__).parents[1] / 'io/sportec.py').read_bytes()).hexdigest()},
                  split_manifest_sha256=sha256(Path(__file__).with_name('sportec_split.json').read_bytes()).hexdigest())
    if evaluate:
        from .methods import METHODS
        report['comparison'] = {m: aggregate_halves([h for r in reports for h in r['methods'][m]['halves']]) for m in METHODS}
        floor = report['comparison']['nearest_smooth50']
        report['delta_vs_smooth50'] = {m: {k: values[k] - floor[k] for k in ('spell_f1_mean', 'macro_f1')}
                                       for m, values in report['comparison'].items()}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('manifest', 'fetch-metadata', 'fetch', 'audit', 'evaluate'))
    parser.add_argument('--partition', choices=('development', 'validation'))
    parser.add_argument('--match', help='one match within the requested unsealed partition')
    parser.add_argument('--data-dir', help='external directory, or SPORTEC_DATA_DIR')
    parser.add_argument('--output', help='JSON report outside Git; required for audit/evaluate')
    parser.add_argument('--chunk-size', type=int, default=4096, help='resource setting only; predictions invariant')
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest()
        if args.command == 'manifest':
            print(json.dumps({k: manifest[k] for k in ('dataset_doi', 'assignments', 'club_overlap', 'holdout_status')}, indent=2))
            return 0
        raw_root = args.data_dir or os.environ.get('SPORTEC_DATA_DIR')
        if not raw_root:
            raise ValueError('--data-dir or SPORTEC_DATA_DIR required; no competition directory fallback')
        root = outside_git(raw_root)
        if args.command != 'fetch-metadata' and not args.partition:
            raise ValueError('--partition development|validation required')
        if args.command == 'fetch-metadata' and args.match:
            raise ValueError('use fetch --partition development --match ID for a single match')
        records = manifest['matches'] if args.command == 'fetch-metadata' else select_matches(args.partition, args.match)
        if args.command in ('fetch-metadata', 'fetch'):
            root.mkdir(parents=True, exist_ok=True)
            for record in records:
                for kind in (('metadata',) if args.command == 'fetch-metadata' else ('metadata', 'positions')):
                    outside_git(root / record[kind]['name'])
                    fetch_artifact(root, record[kind])
            print('Verified external files ready.')
            return 0
        if not args.output:
            raise ValueError('--output required; detailed reports are never printed')
        if args.chunk_size <= 0:
            raise ValueError('--chunk-size must be positive')
        output = outside_git(args.output)
        if output.exists():
            raise ValueError('report already exists; use a new output path to preserve provenance')
        report = run_partition(root, args.partition, evaluate=args.command == 'evaluate',
                               chunk_size=args.chunk_size, match_id=args.match)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + '.partial')
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
        temporary.replace(output)
        if args.command == 'audit' and not all(r['coverage_gate_passed'] for r in report['matches']):
            print('Audit saved: reference coverage gate failed. Review local report before scoring.', file=sys.stderr)
            return 2
        print(f"{args.command.capitalize()} complete: {len(report['matches'])} matches. Report saved locally.")
        return 0
    except (ValueError, OSError) as error:
        print(f'External evaluation stopped: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
