"""Verify the published study data without running a solver."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
METHODS = ('default_setup_default_solve', 'default_setup_lstdq', 'bandit_default', 'bandit_lstdq')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8), (a, b)


def verify(raw=False):
    inventory = read(DATA/'manifest.json')
    assert set(inventory['modules']) == {'module1', 'module2', 'module3'}
    for name, digest in inventory['files'].items():
        path = ROOT/name
        assert path.resolve().is_relative_to(ROOT) and not path.is_symlink(), name
        assert sha(path) == digest, name
    for name, digest in read(DATA/'source_manifest.json').items():
        assert sha(ROOT/name) == digest, name
    result = {}
    for module, count in [(1,720000),(3,120000)]:
        summary=read(DATA/f'module{module}/summary.json')
        rows=summary['rows'];full=[r for r in rows if r['window']=='full']
        assert sum(r['cases'] for r in full)==count
        experiments={f'{family}_{grid}' for family in ['diffusion','advection'] for grid in [40,60,80]} if module==1 else {None}
        actual={(r.get('experiment'),str(r['seed']),r['method'],r['window']) for r in rows}
        expected={(experiment,str(seed),method,window) for experiment in experiments for seed in range(1,7) for method in METHODS for window in ['full','first_1000','joint','last_1000']}
        assert actual==expected and len(rows)==len(expected)
        for row in rows:
            assert row['cases']==(5000 if row['window'] in ['full','joint'] else 1000)
            baseline=next(r for r in rows if r.get('experiment')==row.get('experiment') and r['seed']==row['seed'] and r['window']==row['window'] and r['method']==METHODS[0])
            close(row['native_reduction_pct'],100*(1-row['native_sec']/baseline['native_sec']))
            close(row['inclusive_reduction_pct'],100*(1-row['total_sec']/baseline['total_sec']))
        for a in summary['aggregates']:
            selected=[r for r in rows if r.get('experiment')==a.get('experiment') and r['window']==a['window'] and r['method']==a['method']]
            assert len(selected)==6
            for metric in ['native_reduction_pct','inclusive_reduction_pct']:
                close(a[metric+'_mean'],statistics.mean(r[metric] for r in selected))
                close(a[metric+'_sd'],statistics.stdev(r[metric] for r in selected))
        for config in (DATA/f'module{module}/configurations').rglob('*.json'):
            cfg=read(config)
            assert cfg['stream']['online_cases']==5000 and cfg['stream']['warmup_cases']==0
            assert cfg['setup']['candidate_schedule']['pool_size']==1024
            assert all(m.get('solve_activation_case',0)==0 for m in cfg['methods'])
            space=cfg['setup']['configuration_spaces']['expanded_no_endpoints']
            assert space['numeric_grids']['agg_num_levels']==[0,1,2] and space['agg_interp_types']==[4,6,7]
        result[f'module{module}']={'method_cases':count,'seeds':6}
    rows=read(DATA/'module1/summary.json')['rows']
    for path in (DATA/'module1/arrays').glob('*.npz'):
        with np.load(path,allow_pickle=False) as a:
            assert a['seeds'].tolist()==['1','2','3','4','5','6']
            assert tuple(a['methods'])==METHODS
            t=a['timings_sec'];assert t.shape==(6,4,5000,6)
            np.testing.assert_allclose(t[...,0]+t[...,1],t[...,2],atol=1e-8)
            np.testing.assert_allclose(t[...,2]+t[...,3]+t[...,4],t[...,5],atol=1e-8)
            for row in [r for r in rows if r['experiment']==path.stem]:
                start,stop={'full':(0,5000),'joint':(0,5000),'first_1000':(0,1000),'last_1000':(4000,5000)}[row['window']]
                sums=t[int(row['seed'])-1,METHODS.index(row['method']),start:stop].sum(axis=0)
                close(sums[2],row['native_sec']);close(sums[5],row['total_sec'])
    summary=read(DATA/'module2/summary.json')
    assert summary['audit']['fresh_scan_trials']==29520 and summary['audit']['evaluation_trials']==8535
    assert summary['audit']['exact_coverage'] and summary['audit']['evaluation_failures']==0
    with np.load(DATA/'module2/arrays.npz',allow_pickle=False) as a:
        assert a['costs_sec'].shape==(6,5,100,4) and a['all_repetitions_success'].all()
        totals=a['costs_sec'].sum(axis=2)
        for field,name in [(0,'native'),(1,'inclusive')]:
            reductions=100*(1-totals[:,:,field]/totals[:,:1,field])
            np.testing.assert_allclose(reductions,a[name+'_reductions_pct'])
            for row in summary['rl_comparisons']:
                if row['metric']!=name+'_continuation_sec':continue
                i=a['methods'].tolist().index(row['comparator'])
                values=100*(1-totals[:,4,field]/totals[:,i,field])
                close(values.mean(),row['mean_pct']);close(values.std(ddof=1),row['sample_sd_pct'])
    result['module2']=summary['audit']
    checked=0
    if raw:
        for name,record in read(DATA/'raw_files.json')['files'].items():
            path=ROOT/name
            assert path.resolve().is_relative_to((DATA/'official').resolve()) and not path.is_symlink()
            assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],name
            checked+=1
    return {'passed':True,'modules':result,'raw_files_verified':checked}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',action='store_true',help='Also verify the full local measurement files')
    print(json.dumps(verify(parser.parse_args().raw),indent=2))
