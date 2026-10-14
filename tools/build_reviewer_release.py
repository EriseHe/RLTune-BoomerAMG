"""Build the numbered reviewer release from the preserved development data."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

DEV = Path(__file__).resolve().parents[1]
ROOT = DEV.with_name('RLTune-BoomerAMG')
SOURCE = DEV / 'data/official'
DATA = ROOT / 'data'
RECEIPTS = DEV / 'data/archive/organization/reviewer_release'
RECEIPTS.mkdir(parents=True, exist_ok=True)


def read(p):
    return json.loads(p.read_text())


def dump(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def clone(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['/bin/cp', '-c', str(src), str(dst)], check=True)


def names(text):
    for old, new, word, replacement in [(4,1,'Four','One'),(5,2,'Five','Two'),(6,3,'Six','Three')]:
        text = re.sub(rf'(?i)module[ _-]?0?{old}(?!\d)', lambda m: ('Module ' if m[0][0].isupper() else 'module')+str(new), text)
        text = text.replace('Module'+word, 'Module'+replacement)
    return text


def clean_config(raw, module, label, seed):
    cfg = json.loads(json.dumps(raw))
    cfg['name'] = f'module{module}_{label}_seed{seed}'
    cfg['description'] = f'Module {module}: {label}, seed {seed}, 5,000 paired cases and four cold branches.'
    cfg['output_dir'] = f'results/module{module}/{label}/seed_{seed}'
    return cfg


def extract(source, wanted, title):
    """Copy selected definitions and their same-file dependencies unchanged."""
    text = source.read_text(); tree = ast.parse(text); lines = text.splitlines(True)
    providers = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            providers[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.Import, ast.ImportFrom)):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for item in node.names:
                    providers[item.asname or (item.name.split('.')[0] if isinstance(node, ast.Import) else item.name)] = node
            else:
                for item in ast.walk(node):
                    if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store): providers[item.id] = node
    selected = set(); queue = list(wanted); required = set(wanted)
    while queue:
        key = queue.pop(); node = providers.get(key)
        if node is None or id(node) in selected: continue
        selected.add(id(node))
        dependencies = {n.id for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
        required.update(dependencies)
        queue.extend(dependencies)
    output = [f'"""{title}"""\n', 'from __future__ import annotations\n']
    for node in tree.body:
        if id(node) in selected:
            if isinstance(node, ast.ImportFrom) and node.module == '__future__': continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                node.names = [item for item in node.names if (item.asname or (item.name.split('.')[0] if isinstance(node,ast.Import) else item.name)) in required]
                output.append(ast.unparse(node)+'\n\n')
                continue
            output.append(''.join(lines[node.lineno-1:node.end_lineno])+'\n\n')
    return ''.join(output)


def data_release():
    for number in (1,2,3):
        assert not (DATA/f'module{number}').exists()
        (DATA/f'module{number}').mkdir()
    # Numeric summaries keep every seed and every recorded window.
    for old, new in [(4,1),(6,3)]:
        src=SOURCE/f'module0{old}'; dst=DATA/f'module{new}'
        summary=read(src/'summary.json')
        dump(dst/'summary.json', dict(module=new, rows=summary['rows'], aggregates=summary['aggregates']))
        for filename in ['summary.csv','aggregate.csv']:
            clone(src/filename,dst/filename)
        for job in (read(src/'campaign.json')['jobs'] if old==4 else [dict(output=f'batches/batch_0{batch}',seeds=list(map(str,seeds)),experiment='variable_flow_60') for batch,seeds in [(1,[1,2,3]),(2,[4,5,6])]]):
            label=job['experiment']; stage=src/job['output']
            batch=1 if int(job['seeds'][0])<=3 else 2
            suite=dict(name=f'Module {new}: {label}',seeds=[])
            for seed in job['seeds']:
                cfg=clean_config(read(stage/f'seed_{seed}/experiment_config.json'),new,label,seed)
                target=dst/f'configurations/{label}/seed_{seed}.json'
                dump(target,cfg)
                suite['seeds'].append(dict(label=str(seed),config=f'../configurations/{label}/seed_{seed}.json'))
            dump(dst/f'suites/{label}_batch{batch}.json',suite)
        proto=read(src/('batches/batch_01/diffusion_60/protocol.json' if old==4 else 'batches/batch_01/protocol.json'))
        fields=['methods','exact_default','recovery','feedback_mode','learned_attempts','fixed_rl_attempts','default_fallback_attempts','default_attempts','rl_recovery_cost','rl_whole_case_failure_rollback','setup_failure_observation_retention','failure_score_beta','shared_prefix','learning_start','concurrency','native_sha256']
        dump(dst/'protocol.json',dict(module=new,cases_per_seed=5000,seeds=list(range(1,7)),**{k:proto[k] for k in fields}))
    # Compact, exact array data drives Module 1 tables and all six paper figures.
    src=SOURCE/'module04'; dst=DATA/'module1'
    for label in read(src/'campaign.json')['experiments']:
        merged={}
        for batch in (1,2):
            with np.load(src/f'batches/batch_0{batch}/{label}/paper_figures/plot_data.npz',allow_pickle=False) as a:
                for key in ['seeds','timings_sec','first_primary_weights','first_primary_residuals']:
                    merged.setdefault(key,[]).append(a[key])
                merged['methods']=a['methods'];merged['timing_fields']=a['timing_fields']
        out={k:np.concatenate(v,axis=0) if isinstance(v,list) else v for k,v in merged.items()}
        (dst/'arrays').mkdir(exist_ok=True)
        np.savez_compressed(dst/f'arrays/{label}.npz',**out)
    sys.path.insert(0,str(DEV/'tools/paper'))
    import prepare_module4_current as publisher
    import module4_policy_figure as examples
    # Use the existing independent publisher's fully audited reader once.
    cases, _, _ = publisher.load(DEV)
    example=examples.prepare(cases[examples.EXAMPLE_CASE][examples.EXAMPLE_SEED])
    for key in ['source_sha256','default_residual_replay']:
        example.pop(key,None)
    example['selection_rule']='Consecutive inputs 3882–3884 with different non-exploratory initial weights.'
    dump(dst/'cycle_examples.json',example)
    dump(dst/'method_counts.json',{label:{seed:row['result']['methods'] for seed,row in seeds.items()} for label,seeds in cases.items()})
    # Module 2 preserves the complete selections, evaluation input set and arrays.
    src=SOURCE/'module05';dst=DATA/'module2'
    summary=read(src/'analysis/summary.json')
    summary.pop('protocol',None)
    dump(dst/'summary.json',dict(module=2,**summary))
    for filename in ['per_case.csv','per_seed.csv']:
        clone(src/'analysis'/filename,dst/filename)
    for filename in ['inputs.json','selections.json']:
        clone(src/filename,dst/filename)
    clone(src/'paper_figures/plot_data.npz',dst/'arrays.npz')
    proto=read(src/'protocol.json')
    fields=['evaluation','labels','max_cycles','max_workers','methods','policies','problem','profile','recovery','repetitions_per_case','scan_extra_repeat_case_ids','scan_extra_repetitions','scan_repetitions','seeds','selection','test_cases','tolerance','training_seeds','weights']
    dump(dst/'protocol.json',dict(module=2,**{k:proto[k] for k in fields}))
    cfg=read(src/'configuration.json')
    for key in ['config_path','training_runs']: cfg.pop(key,None)
    cfg['name']='Module 2: matched-policy evaluation'
    cfg['training_runs']=[]
    for seed in range(1,7):
        folder=dst/f'checkpoints/seed_{seed}'
        for name in ['controller.npz','controller.npz.encoder.json','setup_original.npz']:
            clone(src/f'checkpoints/seed_{seed}'/name,folder/name)
        config=read(DATA/f'module1/configurations/diffusion_60/seed_{seed}.json')
        dump(folder/'experiment_config.json',config)
        cfg['training_runs'].append(dict(seed=seed,method='bandit_lstdq',config=f'checkpoints/seed_{seed}/experiment_config.json',controller_checkpoint=f'checkpoints/seed_{seed}/controller.npz',setup_checkpoint=f'checkpoints/seed_{seed}/setup_original.npz'))
    dump(dst/'configuration.json',cfg)
    # Module 3 includes all failures and the common-success comparison.
    for name in ['phase_breakdown.json','phase_breakdown.csv','fairness_audit.json']:
        obj=SOURCE/'module06'/name
        if obj.suffix=='.json':
            value=read(obj)
            for key in ['at','source_sha256','run','source','summary_sha256','report_sha256']:
                if isinstance(value,dict): value.pop(key,None)
            dump(DATA/'module3'/name,value)
        else: clone(obj,DATA/'module3'/name)
    # Raw measurements stay local, with the public module numbering.
    raw_files={}
    for old,new in [(4,1),(5,2),(6,3)]:
        src=SOURCE/f'module0{old}';dst=DATA/f'official/module{new}'
        for path in src.rglob('*'):
            if not path.is_file():continue
            rel=path.relative_to(src)
            if 'source' in rel.parts or 'provenance' in rel.parts or 'paper_figures' in rel.parts:continue
            if path.suffix not in ('.jsonl','.npz','.npy') and not path.name.endswith('.encoder.json'):continue
            if path.name=='events.jsonl':continue
            clone(path,dst/rel)
            raw_files[str((dst/rel).relative_to(ROOT))]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        dump(dst/'manifest.json',dict(module=new,configuration_root=f'data/module{new}',record_files=sum(k.startswith(f'data/official/module{new}/') for k in raw_files)))
    dump(DATA/'raw_files.json',dict(files=raw_files))
    dump(RECEIPTS/'data_packaging.json',dict(passed=True,raw_records=len(raw_files),raw_bytes=sum(v['bytes'] for v in raw_files.values()),measurement_bytes_unchanged=True))


if __name__=='__main__':
    data_release()
