"""Reproduce the paper tables and figures from the released measurements."""
from pathlib import Path
import argparse
import csv
import json
import os

os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import module1_figures as online
import module2_figures as policy
from verify_data import DATA, read, verify


def reproduce(output):
    verify()
    output.mkdir(parents=True,exist_ok=False)
    cases={};counts=read(DATA/'module1/method_counts.json')
    for path in sorted((DATA/'module1/arrays').glob('*.npz')):
        with np.load(path,allow_pickle=False) as a:
            cases[path.stem]={seed:dict(a=a['timings_sec'][i],result={'methods':counts[path.stem][seed]})
                for i,seed in enumerate(a['seeds'].tolist())}
    aggregates,records,selected,_=online.statistics(cases)
    with np.load(DATA/'module1/weights.npz',allow_pickle=False) as a:
        label,seed=str(a['case']),str(a['seed'])
        assert (label,seed)==(selected['case'],selected['seed'])
        cases[label][seed]['w']=a['first_primary_weights']
    examples=read(DATA/'module1/cycle_examples.json')
    destination=output/'module1';(destination/'sections').mkdir(parents=True)
    online.plots(cases,selected,destination/'figures',examples)
    online.write_rows(destination,aggregates,records,selected)
    online.write_values(destination,cases,aggregates,records,selected,examples)
    policy.publish(DATA/'module2',output/'module2')
    for module in [1,3]:
        folder=output/f'module{module}';folder.mkdir(exist_ok=True)
        summary=read(DATA/f'module{module}/summary.json')
        for key in ['rows','aggregates']:
            values=summary[key]
            with (folder/f'{key}.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(values[0]))
                writer.writeheader();writer.writerows(values)
    print(json.dumps({'passed':True,'output':str(output),'modules':[1,2,3],'native_solves':0},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('output/paper'))
    reproduce(parser.parse_args().output.resolve())
