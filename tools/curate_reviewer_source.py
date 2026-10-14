"""Select only the execution dependencies and publishers for the paper release."""
from pathlib import Path
import ast
import json
import shutil
import subprocess
from build_reviewer_release import DEV, ROOT, RECEIPTS, extract, names, dump

assert not (ROOT/'data/manifest.json').exists(), 'One-shot migration: the reviewer release already exists.'

BASE='experiments/paper_final/'
selected={
 'run_06_beta_sweep.py':['native_hashes','TRACE_FIELDS','VARIANT','execute_default_case','metrics','audit_case'],
 'run_06_retry_screen.py':['audit_final_model','request_performance_qos'],
 'run_06_retry_repeatability.py':['observed_model'],
 'run_06_final_validation.py':['audit_controller_counts'],
 'run_06_recovery.py':['request_performance_qos'],
 'analyze_06_recovery.py':['audit'],
 'common/frozen_inputs.py':['execution_source_paths'],
}
renames={
 BASE+'run_04_factorial.py':BASE+'run_online.py',
 BASE+'run_05_matched_policy.py':BASE+'run_policy.py',
 BASE+'plot_04_factorial.py':BASE+'plot_online.py',
 BASE+'plot_05_matched_policy.py':BASE+'plot_policy.py',
 BASE+'run_06_beta_sweep.py':BASE+'recovery/metrics.py',
 BASE+'run_06_retry_screen.py':BASE+'recovery/model_audit.py',
 BASE+'run_06_retry_repeatability.py':BASE+'recovery/observations.py',
 BASE+'run_06_final_validation.py':BASE+'recovery/controller_audit.py',
 BASE+'run_06_recovery.py':BASE+'recovery/system.py',
 BASE+'analyze_06_recovery.py':BASE+'recovery/audit.py',
}
sources={str(p.relative_to(DEV)):p.read_text() for p in (DEV/'experiments').rglob('*.py')
         if 'reproduction' not in p.parts}
for filename,wanted in selected.items():
 sources[BASE+filename]=extract(DEV/BASE/filename,wanted,'Execution and verification helpers for the official studies.')

def module_path(module):
 path=module.replace('.','/')
 for name in [path+'.py',path+'/__init__.py']:
  if name in sources:return name

pending=[BASE+'run_04_factorial.py',BASE+'run_05_matched_policy.py'];keep=set()
while pending:
 path=pending.pop()
 if path in keep:continue
 keep.add(path)
 tree=ast.parse(sources[path]);pkg=path.removesuffix('.py').replace('/','.').rsplit('.',1)[0]
 for node in ast.walk(tree):
  targets=[]
  if isinstance(node,ast.Import):targets=[a.name for a in node.names]
  if isinstance(node,ast.ImportFrom):
   parts=pkg.split('.')
   base='.'.join(parts[:len(parts)-node.level+1]) if node.level else ''
   full='.'.join(p for p in [base,node.module] if p)
   targets=[full]+[full+'.'+a.name for a in node.names]
  for target in targets:
   found=module_path(target)
   if found:pending.append(found)

# The original studies remain on development; only their execution closure is released.
tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
for name in tracked:
 if not name:continue
 p=ROOT/name
 if name.startswith(('experiments/','tools/','docs/')) or '/tests/' in name:
  if p.is_file():p.unlink()

for old in sorted(keep):
 text=sources[old]
 for source,target in renames.items():
  a=source.removesuffix('.py').replace('/','.');b=target.removesuffix('.py').replace('/','.')
  text=text.replace(a,b)
  old_parent,old_stem=a.rsplit('.',1);new_parent,new_stem=b.rsplit('.',1)
  text=text.replace(f'from {old_parent} import {old_stem}',f'from {new_parent} import {new_stem}')
  text=text.replace(Path(source).name,Path(target).name)
 text=names(text)
 text=text.replace('Use run_05_policy_minimax for the accepted paper bundle\'s frozen retiming.', 'Use the released Module 2 configuration and checkpoints.')
 path=ROOT/renames.get(old,old);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)

# Source capture also includes the public command entry points.
p=ROOT/BASE/'common/frozen_inputs.py'
text=p.read_text().replace('"experiments/paper_final",','"experiments/paper_final",')
p.write_text(text)

# Reuse the reviewed plotting functions; only their data reader is replaced.
p=ROOT/'tools/module1_figures.py';p.parent.mkdir(parents=True,exist_ok=True)
text=extract(DEV/'tools/paper/prepare_module4_current.py',
 ['statistics','plots','write_rows','write_values','style','COLORS','WEIGHT_CMAP'],
 'Tables and figures for Module 1.')
text=text.replace('import module4_policy_figure','import cycle_examples').replace('module4_policy_figure.', 'cycle_examples.')
text=names(text).replace('; unreviewed R99.','.')
text=text.replace('Module 1_', 'module1_')
p.write_text(text)
p=ROOT/'tools/cycle_examples.py'
p.write_text(extract(DEV/'tools/paper/module4_policy_figure.py',['draw'],'Illustrative executed relaxation schedules.'))
shutil.copy2(DEV/'tools/paper/paper_figure_io.py',ROOT/'tools/paper_figure_io.py')

# The Module 2 drawing body is unchanged; arrays are supplied by the public reader.
text=(DEV/'tools/paper/prepare_module5_current.py').read_text()
body=text[text.index('    comparisons, rows = [], []'):text.index('    recovery = [r for r in trials')]
body=body.replace('review / "displayed_weights.npz"','output / "displayed_weights.npz"')
body=body.replace('paper / "sections/current_matched_rows.tex"','paper / "sections/module2_rows.tex"')
prefix='''"""Tables and figures for Module 2."""
import json
import numpy as np
import matplotlib.pyplot as plt
from module1_figures import COLORS, WEIGHT_CMAP, style
from paper_figure_io import save

METHODS = ("reference", "fixed", "oracle", "periodic", "rl")

def publish(data, output):
    paper = output
    (paper / "sections").mkdir(parents=True, exist_ok=True)
    (paper / "figures").mkdir(parents=True, exist_ok=True)
    summary = json.loads((data / "summary.json").read_text())
    protocol = json.loads((data / "protocol.json").read_text())
    summary["protocol"] = protocol
    with np.load(data / "arrays.npz", allow_pickle=False) as a:
        seeds = a["seeds"].tolist()
        cases = a["case_ids"]
        costs = a["costs_sec"]
        weights = a["first_repetition_primary_weights"]
        success = a["all_repetitions_success"]
        totals = costs.sum(axis=2)
        native = 100 * (1 - totals[:, :, 0] / totals[:, :1, 0])
        inclusive = 100 * (1 - totals[:, :, 1] / totals[:, :1, 1])
        np.testing.assert_allclose(native, a["native_reductions_pct"])
        np.testing.assert_allclose(inclusive, a["inclusive_reductions_pct"])
    assert success.all() and seeds == [1, 2, 3, 4, 5, 6]
    manifest = {"illustrative_seed": seeds[int(np.argmax(native[:, 4]))]}
'''
(ROOT/'tools/module2_figures.py').write_text(prefix+body+'    return comparisons\n')
dump(RECEIPTS/'source_selection.json',dict(execution_modules=len(keep),renames=renames,selected_functions=selected,retained=sorted(keep)))
print('EXECUTION_MODULES',len(keep))
