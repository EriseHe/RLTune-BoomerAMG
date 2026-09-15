"""Audit and summarize the two-method composite activation pilot (no solves)."""
from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from joint_experiment_plotting import _read_json_lines
from joint_online_common import method_stream_summary, validate_recovery_stream
from joint_reporting import _action_summary
from joint_rl_activation import ReliabilityActivationGate, ReliabilityActivationSpec
from online_td_experiment_common import _write_json


METHODS = ('fixed_start', 'composite_start')
COMPONENTS = ('setup_runtime', 'solve_runtime', 'infer_runtime', 'bandit_overhead_runtime')


def analyze(path: Path) -> dict:
    config = json.loads((path/'experiment_config.json').read_text())
    result = json.loads((path/'result.json').read_text())
    manifest = json.loads((path/'stream_manifest.json').read_text())
    fork = json.loads((path/'shared_prefix.json').read_text())
    total = config['stream']['online_cases']
    assert config['setup']['shared_online_prefix'] and fork['valid']
    assert tuple(m['id'] for m in config['methods']) == METHODS
    assert manifest['sha256'] == config['stream']['expected_sha256'] == result['protocol']['stream']['sha256']
    fixed_boundary = config['methods'][0]['solve_activation_case']
    rule = ReliabilityActivationSpec.from_mapping(config['methods'][1]['solve_activation'])
    gate = ReliabilityActivationGate(rule)
    old = ReliabilityActivationGate(ReliabilityActivationSpec(rule.p_bad, .01, rule.delta))
    records = {m:_read_json_lines(path/'trajectories'/f'{m}.jsonl') for m in METHODS}
    recovery = {}
    for method, rows in records.items():
        assert len(rows) == total
        assert result['bandit_online_steps'][method] == total
        recovery[method] = validate_recovery_stream(rows, expect_bandit_transaction=True)
        for i, row in enumerate(rows):
            assert row['online_index'] == i and row['stream_index'] == i
            assert row['mkw'] == records['fixed_start'][i]['mkw']
            outcome = row['outcome']
            costs = [float(outcome[k]) for k in (*COMPONENTS, 'end_to_end_runtime')]
            assert all(math.isfinite(v) and v >= 0 for v in costs)
            assert math.isclose(sum(costs[:-1]), costs[-1], rel_tol=1e-8, abs_tol=1e-10)
            if method == 'composite_start':
                saved = row['rl_activation']
                assert saved['controller_enabled'] == gate.active
                assert saved['observed_this_problem'] == gate.can_observe
                if gate.can_observe:
                    failed = outcome['first_primary_status'] != 'success'
                    gate.observe(first_attempt_failed=failed)
                    if old.can_observe:
                        old.observe(first_attempt_failed=failed)
                assert saved['crossing_case'] == gate.crossing_case
                assert saved['observations'] == gate.observations
                assert math.isclose(saved['log_evalue'], gate.log_evalue, abs_tol=1e-10)
                active = saved['controller_enabled']
            else:
                active = i >= fixed_boundary
            if not active:
                assert not outcome.get('cycle_actions') and not outcome.get('controller_update_committed')
    prefix = min(gate.crossing_case or total, fixed_boundary)
    assert fork['completed_instances'] == prefix
    order = _read_json_lines(path/'trajectories/method_order.jsonl')
    assert len(order) == total
    for i, row in enumerate(order):
        assert row['online_index'] == i
        assert row['method_order'] == ['fixed_start'] if i < prefix else set(row['method_order']) == set(METHODS)
        for method in METHODS:
            assert (records[method][i].get('shared_prefix_source') == 'fixed_start') == (i < prefix)
        if i < prefix:
            fixed = records['fixed_start'][i]; dynamic = records['composite_start'][i]
            for key in ('mkw', 'params', 'arm_index', 'bandit_timing'):
                assert fixed[key] == dynamic[key]
            a,b = fixed['outcome'],dynamic['outcome']
            assert math.isclose(b['end_to_end_runtime']-a['end_to_end_runtime'], b['activation_runtime'], abs_tol=1e-10)
            assert a['primary_attempts'] == b['primary_attempts']
    windows = {}
    for name, start in ((f'all_{total}',0), (f'last_{min(1000,total)}', max(0,total-1000))):
        summaries = {}
        for method, rows in records.items():
            subset = rows[start:]
            summary = method_stream_summary(subset)
            summary['first_attempt_status_counts'] = dict(Counter(r['outcome']['first_primary_status'] for r in subset))
            summary['activation_runtime_sec'] = sum(r['outcome'].get('activation_runtime',0.) for r in subset)
            summary['monitored_activation_runtime_sec'] = sum(r['outcome'].get('activation_runtime',0.) for r in subset if r.get('rl_activation',{}).get('observed_this_problem'))
            summary['extra_recovery_native_runtime_sec'] = sum(
                sum(a['setup_runtime']+a['solve_runtime'] for a in r['outcome']['primary_attempts'][1:])
                + r['outcome'].get('fallback_setup_runtime',0.) + r['outcome'].get('fallback_solve_runtime',0.)
                for r in subset)
            summary['mean_total_attempt_cycles'] = float(np.mean([
                sum(a['cycles'] for a in r['outcome']['primary_attempts']) + r['outcome'].get('fallback_cycles',0)
                for r in subset]))
            summary['actions'] = _action_summary(subset)
            reported = result['windows'].get(name,{}).get('methods',{}).get(method)
            if reported is not None:
                for key, value in summary['totals_sec'].items():
                    assert math.isclose(value, reported['totals_sec'][key], rel_tol=1e-9, abs_tol=1e-8)
            summaries[method] = summary
        fixed_cost, dynamic_cost = [summaries[m]['totals_sec']['end_to_end_runtime'] for m in METHODS]
        windows[name] = {'methods':summaries, 'dynamic_minus_fixed_sec':dynamic_cost-fixed_cost,
                         'dynamic_reduction_percent':100*(1-dynamic_cost/fixed_cost)}
    report = dict(valid=True, stream_sha256=manifest['sha256'], problems_per_method=total,
                  physical_method_executions=2*total-prefix, shared_prefix=fork,
                  activation=gate.summary(), recovery=recovery, windows=windows,
                  legacy_point_gate_reference_replay={**old.summary(), 'reference_prefix_length':gate.observations,
                      'censored_without_crossing':not old.active,
                      'interpretation':'Reference-prefix replay only; not a closed-loop runtime comparison.'})
    output = path/'activation_analysis'; output.mkdir(exist_ok=True)
    _write_json(output/'report.json',report)
    lines = ['# Composite activation pilot', '',
             f"Audited {total} problems per method; first dynamic RL problem: {gate.summary()['first_rl_case']}; shared prefix: {prefix}.", '',
             'Positive reduction means dynamic activation used less recorded online end-to-end time.', '',
             '| Window | Fixed (s) | Dynamic (s) | Dynamic − fixed (s) | Reduction |',
             '|---|---:|---:|---:|---:|']
    for name,w in windows.items():
        a,b = [w['methods'][m]['totals_sec']['end_to_end_runtime'] for m in METHODS]
        lines.append(f"| {name} | {a:.3f} | {b:.3f} | {b-a:+.3f} | {w['dynamic_reduction_percent']:+.2f}% |")
    lines += ['', '## Full-stream components and reliability', '',
              '| Method | Setup (s) | Solve (s) | Controller incl. gate (s) | Bandit (s) | Gate subset (s) | First failures | Unrecovered |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        summary = windows[f'all_{total}']['methods'][method]; t = summary['totals_sec']
        lines.append(f"| {method} | {t['setup_runtime']:.3f} | {t['native_solve_runtime']:.3f} | {t['controller_runtime']:.3f} | {t['setup_bandit_overhead']:.3f} | {summary['activation_runtime_sec']:.3f} | {summary['primary_failure_count']} | {summary['unrecovered_failure_count']} |")
    lines += ['', f"Fork/checkpoint preparation: {fork['fork_preparation_runtime_sec']:.4f} s, outside online costs.",
              f"Legacy point-gate prefix replay: first RL problem {old.summary()['first_rl_case']}; reference observations available {gate.observations}.", '',
              'One prespecified replicate is exploratory. The gate controls false activation under persistent unacceptable reference risk; it does not certify current/future RL risk or imply runtime superiority.', '',
              'Recovery is included in setup/solve totals. Gate time is already included in controller time. Shared-prefix costs count once per logical method and correspond to one physical execution.', '',
              '![Cumulative cost and evidence](cost_and_evidence.png)', '']
    (output/'report.md').write_text('\n'.join(lines))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,1,figsize=(10,9),layout='constrained')
    x = np.arange(1,total+1)
    cumulative = {}
    for method, rows in records.items():
        cumulative[method] = np.cumsum([r['outcome']['end_to_end_runtime'] for r in rows])
        axes[0].plot(x,cumulative[method],label=method)
    axes[0].set_ylabel('Cumulative online cost (s)'); axes[0].legend()
    axes[1].plot(x,cumulative['composite_start']-cumulative['fixed_start'],color='#a44a3f')
    axes[1].axhline(0,color='gray',lw=.8); axes[1].set_ylabel('Dynamic − fixed (s)')
    monitored = [r for r in records['composite_start'] if r['rl_activation']['observed_this_problem']]
    axes[2].plot(np.arange(1,len(monitored)+1),[r['rl_activation']['log_evalue'] for r in monitored])
    axes[2].axhline(gate.log_threshold,color='#a44a3f',ls='--',label='Activation threshold')
    failures = [i+1 for i,r in enumerate(monitored) if r['outcome']['first_primary_status']!='success']
    axes[2].scatter(failures,[0]*len(failures),marker='|',color='gray',label='First-attempt failure')
    axes[2].set_ylabel('log M'); axes[2].set_xlabel('Completed problem'); axes[2].legend()
    for ax in axes:
        ax.axvline(fixed_boundary,color='gray',ls=':',lw=.8)
        if gate.active:
            ax.axvline(gate.crossing_case,color='#42858c',ls=':',lw=.8)
        ax.grid(alpha=.15)
    fig.savefig(output/'cost_and_evidence.png',dpi=180); plt.close(fig)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir',type=Path)
    args = parser.parse_args()
    report = analyze(args.result_dir.resolve())
    print(json.dumps({'valid':report['valid'], 'activation':report['activation'],
                      'report':str(args.result_dir/'activation_analysis/report.md')},indent=2))
