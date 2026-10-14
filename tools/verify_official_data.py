"""Verify the publication bundle and, optionally, all local official data.

This command never invokes a solver, performs learning, or writes measurements.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics

from official_data import ROOT, EVIDENCE, registry, module_root, resolve_recorded_path

METHODS = {'default_setup_default_solve', 'default_setup_lstdq', 'bandit_default', 'bandit_lstdq'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8), (a, b)


def verify(full=False):
    catalog = registry()
    inventory = read(EVIDENCE / 'files.json')['files']
    actual = {str(p.relative_to(EVIDENCE)) for p in EVIDENCE.rglob('*')
              if p.is_file() and p.name != 'files.json' and p.name != '.DS_Store'}
    assert actual == set(inventory), 'Publication file inventory differs'
    for name, expected in inventory.items():
        path = EVIDENCE / name
        assert not path.is_symlink() and path.resolve().is_relative_to(EVIDENCE.resolve())
        assert sha(path) == expected, name
    checks = {}
    for module, count in [('module04', 720000), ('module06', 120000)]:
        summary = read(EVIDENCE / module / 'summary.json')
        assert summary['passed']
        full_rows = [r for r in summary['rows'] if r['window'] == 'full']
        assert sum(r['cases'] for r in full_rows) == count
        expected_experiments = {f'{family}_{grid}' for family in ['diffusion', 'advection']
                                for grid in [40, 60, 80]} if module == 'module04' else {None}
        seen = set()
        for r in summary['rows']:
            key = (r.get('experiment'), str(r['seed']), r['method'], r['window'])
            assert key not in seen
            seen.add(key)
            assert r['cases'] == (5000 if r['window'] in ['full', 'joint'] else 1000)
        expected = {(experiment, str(seed), method, window)
                    for experiment in expected_experiments for seed in range(1, 7)
                    for method in METHODS for window in ['full', 'first_1000', 'joint', 'last_1000']}
        assert seen == expected, f'{module}: unexpected seed, method, window, or experiment'
        for r in summary['rows']:
            default = next(d for d in summary['rows'] if d.get('experiment') == r.get('experiment')
                           and d['seed'] == r['seed'] and d['window'] == r['window']
                           and d['method'] == 'default_setup_default_solve')
            close(r['native_reduction_pct'], 100 * (1-r['native_sec']/default['native_sec']))
            close(r['inclusive_reduction_pct'], 100 * (1-r['total_sec']/default['total_sec']))
        for a in summary['aggregates']:
            selected = [r for r in summary['rows'] if r.get('experiment') == a.get('experiment')
                        and r['window'] == a['window'] and r['method'] == a['method']]
            assert len(selected) == 6
            for metric in ['native_reduction_pct', 'inclusive_reduction_pct']:
                close(a[metric+'_mean'], statistics.mean(r[metric] for r in selected))
                close(a[metric+'_sd'], statistics.stdev(r[metric] for r in selected))
        checks[module] = {'method_cases': count, 'seeds': 6, 'summary_rows': len(seen)}
    summary5 = read(EVIDENCE / 'module05/analysis/summary.json')
    assert summary5['audit']['fresh_scan_trials'] == 29520
    assert summary5['audit']['evaluation_trials'] == 8535
    assert summary5['audit']['exact_coverage'] and summary5['audit']['evaluation_failures'] == 0
    assert read(EVIDENCE / 'module05/protocol.json')['training_seeds'] == list(range(1, 7))
    checks['module05'] = summary5['audit']
    p6 = read(EVIDENCE / 'module06/protocol.json')
    assert p6['beta'] == 2 and p6['activation_case'] == 0 and not p6['shared_prefix']
    for batch in ['batch_01', 'batch_02']:
        for seed in (range(1, 4) if batch == 'batch_01' else range(4, 7)):
            cfg = read(EVIDENCE / f'module06/batches/{batch}/seed_{seed}/experiment_config.json')
            space = cfg['setup']['configuration_spaces']['expanded_no_endpoints']
            assert space['numeric_grids']['agg_num_levels'] == [0, 1, 2]
            assert space['agg_interp_types'] == [4, 6, 7]
    for name, digest in read(EVIDENCE / 'module04/source_manifest.json').items():
        assert sha(ROOT / name) == digest, f'Official execution source differs: {name}'
    local_files = 0
    if full:
        files = read(EVIDENCE / 'raw_manifest.json')['files']
        for module in catalog['modules']:
            module_root(module)
        protocol5 = read(module_root('module05') / 'protocol.json')
        for seed, source in protocol5['sources'].items():
            for record in source['original_files'].values():
                selected = resolve_recorded_path(record['path'])
                expected_stage = module_root('module04') / 'batches' / ('batch_01' if int(seed) <= 3 else 'batch_02') / 'diffusion_60' / f'seed_{seed}'
                assert selected.resolve().is_relative_to(expected_stage.resolve())
                assert sha(selected) == record['sha256']
        for name, expected in files.items():
            path = ROOT / name
            assert name.startswith('data/official/') and not path.is_symlink()
            assert path.resolve().is_relative_to((ROOT/'data/official').resolve())
            assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], name
            local_files += 1
        actual_local = {str(p.relative_to(ROOT)) for p in (ROOT/'data/official').rglob('*')
                        if p.is_file() and p.name != '.DS_Store'}
        allowed = set(files) | {'data/official/source_paths.json'}
        assert actual_local <= allowed, f'Unregistered official files: {actual_local-allowed}'
    return {'passed': True, 'modules': checks, 'publication_files': len(inventory),
            'local_files_verified': local_files, 'archive_selected': False, 'numerical_runs': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Hash all locally stored official raw data')
    args = parser.parse_args()
    print(json.dumps(verify(args.full), indent=2))
