"""Serial three-method validation of one frozen setup-recovery protocol.

Compute one fresh setup-only prefix and copy its complete learning state into
two fresh continuations. Default is timed on every input. After the prefix,
setup-only, setup+RL and Default execute serially in balanced input-wise order.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import itertools
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from experiments.runtime import configure_single_thread, prevent_sleep, stop_sleep_prevention, THREAD_KEYS
configure_single_thread()
import numpy as np

from experiments.paper_final import run_06_retry_screen as screen
from experiments.paper_final import run_06_retry_repeatability as serial
from experiments.paper_final.common.artifacts import (
    ROOT, checkpoint_payload_hash, digest, dump, emit, file_hash, now, read, ready, write_csv,
)
from experiments.paper_final.recovery.retry_comparison import use_retry_rule
from experiments.paper_final.recovery.joint_validation import copy_complete_prefix, execute_learned
from setup.space import DEFAULT_SETUP_PARAMS

METHODS=("setup_only","setup_rl","default")
ORDERS=tuple(itertools.permutations(METHODS))


def order_for(index, prefix_cases):
    if index<prefix_cases:
        return ("prefix","default") if index%2==0 else ("default","prefix")
    return ORDERS[(index-prefix_cases)%len(ORDERS)]


def audit_controller_counts(rows, summary):
    retained=[attempt for row in rows if row['outcome']['controller_update_committed']
              for attempt in row['outcome']['primary_attempts'] if attempt.get('rl_episode_recorded',False)]
    steps=sum(len(attempt['cycle_actions']) for attempt in retained)
    if summary['steps']!=steps or summary['episodes']!=len(retained):
        raise ValueError('RL counters differ from retained attempt observations')


def prepare_stage(output, *, seed, feedback_mode, beta=5.0, cases=5000, prefix_cases=1000):
    if feedback_mode not in ("suffix_delayed","adjacent_delayed"):
        raise ValueError("Choose one delayed feedback rule")
    if not 0<prefix_cases<cases<=5000:
        raise ValueError("Invalid full-stream or smoke boundary")
    if cases==5000 and prefix_cases!=1000:
        raise ValueError("The full experiment requires a 1000-input prefix")
    if beta!=5.0:
        raise ValueError("This validation holds beta fixed at 5")
    output.mkdir(parents=True,exist_ok=False)
    raw=read(screen.sweep.CONFIG)
    seeds=read(screen.SEED_CONFIGS[seed])
    raw['seeds']=copy.deepcopy(seeds['seeds'])
    for key in ('seed_groups','shuffle_seeds'):raw['stream'][key]=list(seeds['stream'][key])
    raw['stream']['expected_sha256']=''
    raw['solve']['lstdq']['beta']=beta
    screen.sweep.validate_configuration(raw,beta)
    checks={}; stream_hash=None
    for name in ('prefix',*METHODS):
        config=copy.deepcopy(raw)
        if name in ('prefix','setup_only'):config=screen.sweep.baseline_configuration(config,'linucb')
        elif name=='default':config=screen.sweep.baseline_configuration(config,'default')
        config['name']=f'final_recovery_{name}_seed{seed}_{feedback_mode}'
        config['description']=f'Serial {cases}-input validation, fresh {prefix_cases}-input common prefix; {feedback_mode}; beta 5'
        config['output_dir']=str(output/name)
        runtime=screen.sweep.decode(config,output/name)
        if stream_hash is None:stream_hash=runtime.manifest['sha256']
        if runtime.manifest['sha256']!=stream_hash:raise ValueError('Unpaired streams')
        if seed==5 and stream_hash!=screen.FULL_STREAM_HASH:raise ValueError('Seed-5 input stream changed')
        if (runtime.bundle is not None)!=(name=='setup_rl'):raise ValueError('RL attached to the wrong method')
        if runtime.bundle is not None and (runtime.bundle.controller.steps or runtime.bundle.controller.episodes):
            raise ValueError('Prepared controller is trained')
        config['stream']['expected_sha256']=stream_hash
        path=output/'configs'/f'{name}.json';dump(path,config)
        inputs=screen.input_rows(runtime,cases)
        checks[name]=dict(config_sha256=file_hash(path),inputs_sha256=digest(inputs))
        if name=='prefix':
            (output/'inputs.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in inputs))
    if len({c['inputs_sha256'] for c in checks.values()})!=1:raise ValueError('Input pairing failed')
    hashes=screen.source_hashes();dump(output/'source_manifest.json',hashes)
    for name in hashes:
        destination=output/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination)
    protocol=dict(seed=seed,cases=cases,prefix_cases=prefix_cases,joint_cases=cases-prefix_cases,
        smoke=cases!=5000,beta=beta,feedback_mode=feedback_mode,methods=METHODS,
        failure_score_beta=serial.FailureAwareRetryLinUCB.failure_beta,
        primary_selector='Ordinary runtime LinUCB lower confidence score',
        retry_selector='Minimize failure mean plus failure_score_beta times shared uncertainty; runtime LCB breaks ties',
        feedback_timing='Binary failure target and precision update immediately; mean-preserving provisional runtime labels resolve once at case completion',
        learned_attempts=3,default_fallback_attempts=1,total_attempt_budget=4,
        rl_recovery_cost=False,rl_whole_case_failure_rollback=True,setup_failure_observation_retention=True,
        full_stream_sha256=stream_hash,preparation_checks=checks,native_sha256=screen.sweep.native_hashes(),
        exact_default=DEFAULT_SETUP_PARAMS,prepared_at=now(),execution='one complete method case at a time',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        shared_prefix='Freshly measured once; included identically in both learned method totals; complete setup state copied into fresh continuations',
        continuation_history='Restore previous-arm/history anchors in addition to regression, failure vector, candidate statistics, cursor and RNG',
        setup_label='Own attempt T on success; failed attempt receives adjacent or full measured suffix, including actual fallback; T retains the prior-update-time estimate',
        metric='All-input end_to_end_runtime; each executed attempt, fallback and measured learner cost counted once per reported method')
    dump(output/'protocol.json',protocol)
    dump(output/'status.json',dict(status='prepared',native_solves_executed=0))


def initialize(output,name):
    protocol=read(output/'protocol.json');directory=output/name
    directory.mkdir(exist_ok=False)
    config_path=output/'configs'/f'{name}.json'
    if file_hash(config_path)!=protocol['preparation_checks'][name]['config_sha256']:
        raise ValueError('Prepared configuration changed')
    runtime=screen.sweep.decode(read(config_path),directory);runtime.name=name
    state=dict(pid=os.getpid(),cases=protocol['cases'],full_stream_sha256=runtime.manifest['sha256'],
        inputs_sha256=digest(screen.input_rows(runtime,protocol['cases'])),
        candidate_sha256=None,initial_bandit_sha256=None,initial_controller_sha256=None,
        native_sha256=screen.sweep.native_hashes(),controller_steps=0)
    model=None
    if name!='default':
        branches=screen.build_named_setup_branches(runtime.args,study=screen.resolve_composable_study(runtime.args),
            warmup_instances=(),stream_hash=runtime.manifest['sha256'])
        runtime.branch=branches.branches[runtime.method.name]
        rule='failure_model_suffix' if protocol['feedback_mode']=='suffix_delayed' else 'failure_model_adjacent'
        model=serial.observed_model(use_retry_rule(runtime.branch.policy.model,rule))
        runtime.branch.policy.model=model
        runtime.rows=screen.CandidateRows(model,cases=5000,attempts=3,directory=directory/'unused_extra_candidates')
        model.save_mutable_state(directory/'initial_bandit.npz')
        state.update(candidate_sha256=file_hash(Path(runtime.rows.base._ids.filename)),
            initial_bandit_sha256=checkpoint_payload_hash(directory/'initial_bandit.npz',exclude=('metadata',)))
        if not any(dict(model.actions[int(a)])==DEFAULT_SETUP_PARAMS for a in model._cand.always_include_arms):
            raise ValueError('Catalog reference differs from exact Default')
    if runtime.bundle is not None:
        runtime.bundle.save(directory/'initial_controller.npz')
        state['initial_controller_sha256']=checkpoint_payload_hash(directory/'initial_controller.npz')
        if runtime.bundle.controller.steps or runtime.bundle.controller.episodes:
            raise ValueError('RL trained before the common prefix boundary')
    for key in ('inputs_sha256',):
        if state[key]!=protocol['preparation_checks'][name][key]:raise ValueError('Input pairing changed')
    if state['native_sha256']!=protocol['native_sha256']:raise ValueError('Native library changed')
    dump(directory/'initial_state.json',state)
    return SimpleNamespace(runtime=runtime,model=model,state=state,directory=directory,rows=[],previous=0.0)


def transfer_prefix(prefix,continuations,protocol):
    path=prefix.directory/'final_bandit.npz'
    prefix.model.save_mutable_state(path)
    dump(prefix.directory/'history.json',dict(history=[asdict(s) for s in prefix.model.history],
        candidate_stats_history=prefix.model.candidate_stats_history))
    checks={}
    for name,worker in continuations.items():
        if worker.state['initial_bandit_sha256']!=prefix.state['initial_bandit_sha256']:
            raise ValueError('Continuation cold start differs from the prefix cold start')
        if worker.state['candidate_sha256']!=prefix.state['candidate_sha256']:
            raise ValueError('Continuation candidate schedule differs')
        worker.previous=copy_complete_prefix(prefix.model,worker.model,path,previous_update=prefix.previous)
        worker.model.save_mutable_state(worker.directory/'prefix_bandit.npz')
        if checkpoint_payload_hash(worker.directory/'prefix_bandit.npz',exclude=('metadata',))!=checkpoint_payload_hash(path,exclude=('metadata',)):
            raise ValueError('Continuation checkpoint differs from common state')
        checks[name]=dict(prefix_bandit_sha256=checkpoint_payload_hash(path,exclude=('metadata',)),
            history_sha256=digest([asdict(s) for s in worker.model.history]),
            candidate_cursor=worker.model._candidate_schedule.cursor,observations=worker.model.t,
            previous_update=worker.previous,controller_steps=0)
        if checks[name]['candidate_cursor']!=protocol['prefix_cases']*3:
            raise ValueError('Wrong setup/RL boundary')
        if worker.runtime.bundle is not None and (worker.runtime.bundle.controller.steps or worker.runtime.bundle.controller.episodes):
            raise ValueError('RL trained during setup-only prefix')
        worker.rows=[dict(row,method=name,shared_prefix=True,prefix_source='prefix/trajectory.jsonl') for row in prefix.rows]
    for key in ('A_inv','b','failure_b'):
        if np.shares_memory(getattr(continuations['setup_only'].model,key),getattr(continuations['setup_rl'].model,key)):
            raise ValueError('Continuations share mutable arrays')
    return checks


def run_stage(output):
    protocol=read(output/'protocol.json')
    if read(output/'status.json')['status']!='prepared':raise ValueError('Stage already started')
    if screen.source_hashes()!=read(output/'source_manifest.json'):raise ValueError('Frozen source changed')
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size()!=1:raise ValueError('Expected one MPI rank')
    qos=screen.request_performance_qos()
    dump(output/'environment.json',dict(pid=os.getpid(),python=sys.version,executable=sys.executable,
        qos=qos,mpi=MPI.Get_library_version(),threads={key:os.environ.get(key) for key in THREAD_KEYS}))
    workers={n:initialize(output,n) for n in ('prefix','default')}
    warmup=[screen.sweep.execute_default_case(workers['default'].runtime,i) for i in range(3)]
    dump(output/'native_warmup.json',dict(measured=False,learner_updates=0,records=warmup))
    handles={n:(w.directory/'trajectory.jsonl').open('w') for n,w in workers.items()}
    started=time.perf_counter();prefix_audit=None
    try:
        with (output/'execution.jsonl').open('w') as schedule:
            for index in range(protocol['cases']):
                if index==protocol['prefix_cases']:
                    prefix=workers['prefix']
                    prefix_audit=screen.audit_final_model(prefix.runtime,prefix.rows)
                    new={n:initialize(output,n) for n in METHODS[:2]}
                    checks=transfer_prefix(prefix,new,protocol)
                    dump(output/'prefix_pairing.json',dict(passed=True,continuations=checks,
                        prefix_model_audit=prefix_audit,rl_steps=0))
                    dump(output/'checkpoint_1000.json',dict(actual_prefix_cases=protocol['prefix_cases'],
                        common_prefix=True,passed=True,rl_steps=0,**screen.metrics(prefix.rows)))
                    for n,w in new.items():
                        handles[n]=(w.directory/'trajectory.jsonl').open('w')
                        for row in w.rows:handles[n].write(json.dumps(ready(row),allow_nan=False)+'\n')
                        workers[n]=w
                    handles['prefix'].flush();handles['prefix'].close()
                    dump(prefix.directory/'result.json',dict(complete=True,cases=len(prefix.rows),
                        trajectory_sha256=file_hash(prefix.directory/'trajectory.jsonl'),
                        controller=dict(steps=0,episodes=0),model_audit=prefix_audit,previous_update=prefix.previous))
                for position,name in enumerate(order_for(index,protocol['prefix_cases'])):
                    worker=workers[name];begin=time.monotonic_ns()
                    if name=='default':row=screen.sweep.execute_default_case(worker.runtime,index)
                    else:
                        row,worker.previous=execute_learned(worker.runtime,index,worker.previous,
                            feedback_mode=protocol['feedback_mode'],activation_case=protocol['prefix_cases'])
                    end=time.monotonic_ns()
                    row['execution']=dict(pid=os.getpid(),position=position)
                    worker.rows.append(row)
                    handles[name].write(json.dumps(ready(row),allow_nan=False)+'\n')
                    schedule.write(json.dumps(dict(input_index=index,method=name,pid=os.getpid(),
                        position=position,start_ns=begin,end_ns=end))+'\n')
                if (index+1)%100==0 or index+1==protocol['cases']:
                    for handle in handles.values():
                        if not handle.closed:handle.flush()
                    schedule.flush()
                    names=('prefix','default') if index<protocol['prefix_cases'] else METHODS
                    progress=dict(status='running',pid=os.getpid(),processed=index+1,cases=protocol['cases'],
                        elapsed_sec=time.perf_counter()-started,methods={n:screen.metrics(workers[n].rows) for n in names})
                    dump(output/'status.json',progress)
                    emit(output,'Serial validation block complete',processed=index+1,
                        totals={n:round(v['total_sec'],3) for n,v in progress['methods'].items()})
                if (index+1)%1000==0 and index>=protocol['prefix_cases']:
                    for name in METHODS[:2]:
                        workers[name].model.save_mutable_state(workers[name].directory/f'bandit_{index+1}.npz')
                    workers['setup_rl'].runtime.bundle.save(workers['setup_rl'].directory/f'controller_{index+1}.npz')
    finally:
        for handle in handles.values():
            if not handle.closed:handle.close()
    for name in METHODS:
        worker=workers[name];model_audit=None;controller=dict(steps=0,episodes=0)
        if worker.model is not None:
            model_audit=screen.audit_final_model(worker.runtime,worker.rows)
            worker.model.save_mutable_state(worker.directory/'final_bandit.npz')
            if worker.model._candidate_schedule.cursor!=protocol['cases']*3:raise ValueError('Wrong final candidate boundary')
        if worker.runtime.bundle is not None:
            worker.runtime.bundle.save(worker.directory/'final_controller.npz')
            controller=worker.runtime.bundle.summary()
        dump(worker.directory/'result.json',dict(complete=True,cases=len(worker.rows),
            trajectory_sha256=file_hash(worker.directory/'trajectory.jsonl'),controller=controller,
            bandit_observations=0 if worker.model is None else worker.model.t,model_audit=model_audit))
    if screen.source_hashes()!=read(output/'source_manifest.json') or screen.sweep.native_hashes()!=protocol['native_sha256']:
        raise ValueError('Source or native library changed during validation')
    analyze_stage(output)
    dump(output/'status.json',dict(status='complete',audits_passed=True,completed_at=now()))


def analyze_stage(output):
    protocol=read(output/'protocol.json');cases=protocol['cases'];boundary=protocol['prefix_cases']
    if not read(output/'prefix_pairing.json')['passed']:raise ValueError('Prefix copying failed')
    prefix=[json.loads(s) for s in (output/'prefix'/'trajectory.jsonl').read_text().splitlines()]
    expected=[json.loads(s) for s in (output/'inputs.jsonl').read_text().splitlines()]
    table=[];records={}
    for name in METHODS:
        result=read(output/name/'result.json');path=output/name/'trajectory.jsonl'
        if not result['complete'] or file_hash(path)!=result['trajectory_sha256']:raise ValueError('Incomplete method')
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        if len(rows)!=cases:raise ValueError('Incomplete method stream')
        for index,row in enumerate(rows):
            if {k:row[k] for k in ('problem_index','mkw','context')}!=expected[index]:raise ValueError('Input pairing changed')
            active=name=='setup_rl' and index>=boundary
            if row['rl_active']!=active:raise ValueError('Wrong RL activation')
            if name=='default':screen.sweep.audit_default_case(row)
            else:
                screen.sweep.audit_case(row,protocol['beta'] if active else None,feedback_mode=protocol['feedback_mode'])
                if index<boundary:
                    for key in ('mkw','context','params','outcome','rl_active'):
                        if row[key]!=prefix[index][key]:raise ValueError('Copied prefix differs from measured common prefix')
        if name!='default':
            if not result['model_audit']['passed'] or result['bandit_observations']!=sum(r['outcome']['primary_attempt_count'] for r in rows):
                raise ValueError('Setup observation audit failed')
        if name!='setup_rl' and result['controller']!=dict(steps=0,episodes=0):raise ValueError('Control trained RL')
        if name=='setup_rl' and result['controller']['steps']<=0:raise ValueError('RL did not train in active phase')
        audit_controller_counts(rows,result['controller'])
        records[name]=rows
        windows=[(f'all_{cases}',rows),(f'prefix_{boundary}',rows[:boundary]),
                 (f'joint_{cases-boundary}',rows[boundary:]),(f'last_{min(1000,cases-boundary)}',rows[-min(1000,cases-boundary):])]
        windows.extend((f'inputs_{start+1}_{min(start+250,cases)}',rows[start:start+250])
                       for start in range(boundary,cases,250))
        for window,subset in windows:
            table.append(dict(branch=name,window=window,**screen.metrics(subset)))
    executed=[json.loads(s) for s in (output/'execution.jsonl').read_text().splitlines()]
    expected_count=2*boundary+3*(cases-boundary)
    if len(executed)!=expected_count:raise ValueError('Wrong executed case count')
    cursor=0;previous_end=-1
    for index in range(cases):
        order=order_for(index,boundary);block=executed[cursor:cursor+len(order)];cursor+=len(order)
        if tuple(r['method'] for r in block)!=order:raise ValueError('Execution order changed')
        for row in block:
            if row['input_index']!=index or row['start_ns']<previous_end or row['end_ns']<row['start_ns']:
                raise ValueError('Overlapping or reordered timing')
            previous_end=row['end_ns']
    if len({r['pid'] for r in executed})!=1:raise ValueError('Expected one numerical process')
    dump(output/'execution_audit.json',dict(passed=True,actual_method_cases=expected_count,
        reported_method_records=cases*3,prefix_measured_once=True,overlap_count=0))
    dump(output/'comparison.json',dict(complete=True,audits_passed=True,protocol=protocol,rows=table))
    write_csv(output/'comparison.csv',table)
    lines=[f"Serial {cases}-input final-protocol validation, seed {protocol['seed']}",
        f"Feedback: {protocol['feedback_mode']}; failure-aware retries; three learned attempts plus exact Default.",
        f"Fresh common {boundary}-input setup prefix, copied with complete selection state; {cases-boundary} continuation inputs.",
        f"Solve RL: beta={protocol['beta']}, no recovery augmentation, rollback only on whole-case failure.",
        'All-input times include every attempt, fallback and learner overhead. Common prefix cost is included in both learned totals.',
        'One numerical process, serial complete cases, balanced method order; all audits passed.','']
    if protocol['smoke']:lines.insert(0,'SMOKE CHECK ONLY: shortened stream and early activation.')
    for row in table:
        lines.append(f"{row['branch']} / {row['window']}: total={row['total_sec']:.6f}s; failures={row['failures']}; "
            f"final fallbacks={row['fallbacks']}; second={row['second_attempts']}; third={row['third_attempts']}; "
            f"Default-setup learned retries={row['learned_default_retries']}")
    (output/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    for name,rows in records.items():
        for ax,key in zip(axes,('end_to_end_runtime','failed','fallback_used')):
            ax.plot(np.arange(1,cases+1),np.cumsum([r['outcome'][key] for r in rows]),label=name)
    for ax,label in zip(axes,('Cumulative time (s)','Cumulative final failures','Cumulative final Default fallbacks')):
        ax.set(xlabel='Input index',ylabel=label);ax.axvline(boundary,color='.5',ls=':');ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle(f"Seed {protocol['seed']}: {protocol['feedback_mode']}; serial setup-only vs setup+RL")
    fig.savefig(output/'comparison.png',dpi=180);fig.savefig(output/'comparison.pdf');plt.close(fig)


def prepare(output, *, seeds=(5,6), feedback_mode='suffix_delayed', cases=5000, prefix_cases=1000):
    output.mkdir(parents=True,exist_ok=False);stages=[]
    for seed in seeds:
        name=f'seed_{seed}'
        prepare_stage(output/name,seed=seed,feedback_mode=feedback_mode,cases=cases,prefix_cases=prefix_cases)
        stages.append(dict(name=name,seed=seed,protocol_sha256=file_hash(output/name/'protocol.json')))
    dump(output/'sequence_protocol.json',dict(stages=stages,feedback_mode=feedback_mode,cases=cases,prefix_cases=prefix_cases))
    dump(output/'status.json',dict(status='prepared',native_solves_executed=0))


def run(output):
    if read(output/'status.json')['status']!='prepared':raise ValueError('Sequence already started')
    with (output/'manager.lock').open('x') as f:f.write(str(os.getpid()))
    dump(output/'manager.json',dict(pid=os.getpid(),started_at=now()))
    guard=prevent_sleep();worker=None
    try:
        plan=read(output/'sequence_protocol.json')
        for stage in plan['stages']:
            directory=output/stage['name']
            if file_hash(directory/'protocol.json')!=stage['protocol_sha256']:raise ValueError('Protocol changed')
            dump(output/'status.json',dict(status='running',pid=os.getpid(),active_stage=stage['name']))
            emit(output,'Starting full serial validation',stage=stage['name'])
            with (output/f"{stage['name']}.log").open('w') as log:
                worker=subprocess.Popen([sys.executable,'-u','-m',__package__+'.run_06_final_validation',
                    '--stage','--output',str(directory)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                dump(output/'active_worker.json',dict(pid=worker.pid,stage=stage['name'],started_at=now()))
                code=worker.wait()
            if code:raise RuntimeError(f"Stage {stage['name']} exited {code}; inspect its log")
        table=[];lines=['Full serial validation completed and audited.','']
        for stage in plan['stages']:
            comparison=read(output/stage['name']/'comparison.json')
            if not comparison['audits_passed']:raise ValueError('Audit failed')
            table.extend(dict(seed=stage['seed'],**row) for row in comparison['rows'])
            lines.append((output/stage['name']/'REPORT.txt').read_text())
        write_csv(output/'comparison.csv',table)
        dump(output/'comparison.json',dict(complete=True,audits_passed=True,rows=table,plan=plan))
        (output/'REPORT.txt').write_text('\n'.join(lines))
        dump(output/'status.json',dict(status='complete',completed_at=now(),audits_passed=True))
    except BaseException as error:
        dump(output/'status.json',dict(status='failed',failed_at=now(),error=str(error)));raise
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate();worker.wait(timeout=30)
        stop_sleep_prevention(guard)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[5,6],choices=[5,6])
    parser.add_argument('--feedback',choices=['suffix_delayed','adjacent_delayed'],default='suffix_delayed')
    parser.add_argument('--cases',type=int,default=5000)
    parser.add_argument('--prefix',type=int,default=1000)
    modes=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','run','stage','analyze'):modes.add_argument('--'+name,action='store_true')
    args=parser.parse_args();output=args.output.resolve()
    if args.prepare:prepare(output,seeds=tuple(args.seeds),feedback_mode=args.feedback,cases=args.cases,prefix_cases=args.prefix)
    elif args.run:run(output)
    elif args.stage:run_stage(output)
    else:analyze_stage(output)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda signum,frame:sys.exit(128+signum))
    main()
