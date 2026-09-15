"""Continuous evidence, shared-prefix state, and activation boundary checks."""
from __future__ import annotations

import _project_paths  # noqa: F401

import copy
from dataclasses import replace
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import mpmath as mp
import numpy as np
from scipy.integrate import quad

import joint_4k_execution as execution
from composable_joint_4k import build_named_setup_branches, resolve_composable_study
from joint_experiment_config import parse_joint_experiment_config, runtime_config_from_spec
from joint_method_spec import ComposableMethodSpec
from joint_online_common import report_online_outcome
from joint_rl_activation import (
    ReliabilityActivationGate, ReliabilityActivationSpec,
    _log_uniform_bernoulli_evalues,
)

CONFIGS = Path(__file__).resolve().parents[2] / "joint/solve_control/configs"
PILOT = CONFIGS / "paper_test_n60_composite_activation_seed1.json"


def composite(*, p0=.05, delta=.05, horizon=4999):
    return ReliabilityActivationSpec(p0, None, delta, "composite_reliability_mixture", horizon)


class CompositeEvidenceTests(unittest.TestCase):
    def test_scaled_integral_against_high_precision_at_horizon(self):
        cases = [(0, 0), (0, 4999), (4999, 0), (249, 4750), (250, 4749),
                 (2500, 2499), (1, 4998), (50, 4949)]
        with mp.workdps(65):
            p0 = mp.mpf('.05')
            for f, s in cases:
                actual = _log_uniform_bernoulli_evalues(np.array([f]), np.array([s]), .05)[0]
                expected = (mp.log(mp.betainc(f+1, s+1, 0, p0))
                            - (f+1)*mp.log(p0) - s*mp.log1p(-p0))
                self.assertAlmostEqual(actual, float(expected), places=9, msg=str((f, s)))
        f = np.arange(1, 5000)
        np.testing.assert_allclose(
            _log_uniform_bernoulli_evalues(f, np.zeros_like(f), .05),
            -np.log(f+1), atol=1e-13,
        )

    def test_explicit_cumulative_mixture_and_dormant_mass(self):
        for sequence in itertools.product((0, 1), repeat=6):
            gate = ReliabilityActivationGate(composite(p0=.3, delta=1e-12, horizon=6))
            self.assertEqual(gate.log_evalue, 0.)
            for t, failed in enumerate(sequence, 1):
                gate.observe(first_attempt_failed=bool(failed))
                direct = (6-t)/6
                for k in range(t):
                    f = sum(sequence[k:t]); s = t-k-f
                    integral, _ = quad(lambda q: (q/.3)**f * ((1-q)/.7)**s / .3, 0, .3)
                    direct += integral / 6
                self.assertAlmostEqual(math.exp(gate.log_evalue), direct, places=12)
            self.assertFalse(gate.can_observe)
            with self.assertRaises(RuntimeError):
                gate.observe(first_attempt_failed=False)

    def test_finite_tree_false_activation_control(self):
        # Exact enumeration, including optional stopping, not Monte Carlo.
        horizon = 8
        for risk in (.4, .7):
            activation_probability = 0.
            for seq in itertools.product((0, 1), repeat=horizon):
                gate = ReliabilityActivationGate(composite(p0=.4, delta=.2, horizon=horizon))
                for failed in seq:
                    if gate.can_observe:
                        gate.observe(first_attempt_failed=bool(failed))
                if gate.active:
                    activation_probability += risk**sum(seq) * (1-risk)**(horizon-sum(seq))
            self.assertLessEqual(activation_probability, .2 + 1e-13)

    def test_configuration_contract_and_legacy_roundtrip(self):
        raw = json.loads(PILOT.read_text())
        source = json.loads((CONFIGS / 'PAPER_FINAL/PAPER_FINAL_diffusion_n60_seed1.json').read_text())
        for key in ('problem', 'stream', 'seeds', 'solve'):
            self.assertEqual(raw[key], source[key])
        self.assertEqual({k:v for k,v in raw['setup'].items() if k != 'shared_online_prefix'}, source['setup'])
        runtime = runtime_config_from_spec(parse_joint_experiment_config(raw))
        resolve_composable_study(runtime)
        for spec in runtime.methods:
            self.assertEqual(ComposableMethodSpec.from_runner_token(spec.to_runner_token()), spec)
        for mutation in ({'horizon':5000}, {'horizon':0}, {'horizon':2.5}, {'p_good':.01}):
            invalid = copy.deepcopy(raw)
            invalid['methods'][1]['solve_activation'].update(mutation)
            with self.assertRaises(ValueError):
                resolve_composable_study(runtime_config_from_spec(parse_joint_experiment_config(invalid)))
        invalid = copy.deepcopy(raw); invalid['methods'][1]['seed_offset'] = 1
        with self.assertRaisesRegex(ValueError, 'matching LinUCB'):
            resolve_composable_study(runtime_config_from_spec(parse_joint_experiment_config(invalid)))
        old = json.loads((CONFIGS / 'paper_test_n60_context_activation.json').read_text())
        for spec in runtime_config_from_spec(parse_joint_experiment_config(old)).methods:
            self.assertEqual(ComposableMethodSpec.from_runner_token(spec.to_runner_token()), spec)


class SharedPrefixTests(unittest.TestCase):
    def test_crossings_before_at_after_fixed_boundary_and_never(self):
        for crossing in (2, 4, 6, 11, None):
            with self.subTest(crossing=crossing), tempfile.TemporaryDirectory() as tmp:
                self.run_prefix_case(Path(tmp), crossing)

    def run_prefix_case(self, output, crossing):
        total = 12; fixed_boundary = 4
        if crossing is None:
            rule = composite(p0=.8, horizon=total-1)
        else:
            calibration = ReliabilityActivationGate(composite(p0=.8, delta=1e-30, horizon=total-1))
            log_values = [0.]
            for _ in range(crossing):
                calibration.observe(first_attempt_failed=False)
                log_values.append(calibration.log_evalue)
            rule = composite(p0=.8, delta=math.exp(-sum(log_values[-2:])/2), horizon=total-1)
        raw = json.loads(PILOT.read_text())
        runtime = runtime_config_from_spec(parse_joint_experiment_config(raw))
        specs = tuple(replace(s, solve_activation_case=fixed_boundary if s.solve_activation_case else 0,
                              solve_activation=rule if s.solve_activation else None) for s in runtime.methods)
        runtime = replace(runtime, output_dir=output, online_cases=total, methods=specs)
        study = resolve_composable_study(runtime)
        branches = build_named_setup_branches(runtime, study=study, warmup_instances=[], stream_hash='test-only').branches
        source, target = [branches[name].policy.model for name in ('fixed_start', 'composite_start')]
        self.assertEqual(source.rng.bit_generator.state, target.rng.bit_generator.state)
        fork_at = min(crossing or total, fixed_boundary)
        trajectories = output/'trajectories'; trajectories.mkdir()
        calls = []
        bundles = {s.name: Mock() for s in specs}
        for bundle in bundles.values():
            bundle.summary.return_value = {}
        plan = execution.OnlineComparisonPlan(
            output_dir=output, trajectories_dir=trajectories, checkpoints_dir=output/'checkpoints',
            methods=tuple(study.methods), family_by_method=study.family_by_method,
            bandit_methods=study.bandit_methods, composable_specs=study.specs_by_name,
            online_instances=[({'index':i}, np.array([1.,.2,.5,.8])) for i in range(total)],
            branches=branches, controller_bundles=bundles, ppo_runner=None, protocol={},
            warmup=execution.WarmupArtifacts({},'',{},0), solve=execution.SolveExecutionConfig(1e-6,50),
            warmup_cases=0, online_cases=total, method_order_seed=19, progress_every=total,
            aot_enabled=True, aot_max_selections_per_case=3, default_setup_method='unused',
            include_solve_screen_report=False, shared_online_prefix=True,
        )
        def native(method, *, mkw, case_index, controller_enabled=None, **kwargs):
            active = case_index >= fixed_boundary if method == 'fixed_start' else controller_enabled
            def solve(params):
                calls.append((method, case_index, active))
                return dict(runtime=.05, native_runtime=.05, setup_runtime=.03,
                            solve_runtime=.02, native_solve_runtime=.02, infer_runtime=.001 if active else 0.,
                            failed=False, iterations=2, primary_status='success',
                            first_primary_status='setup_failure' if crossing is None else 'success',
                            recovered=crossing is None, fallback_used=False,
                            bandit_update_committed=True, controller_update_committed=active)
            return solve
        def bandit(*, policy, problem_context, solver_fn, prev_update_est, **kwargs):
            if source.t == target.t == fork_at:
                self.assertEqual(source.history[-1].arm_index, target.history[-1].arm_index)
                self.assertEqual(source._candidate_schedule.cursor, 3*fork_at)
                # Identical candidates/scores before the first independent update.
                probe_source = copy.deepcopy(source.rng.bit_generator.state)
                a = source._structured_candidate_subset(512)[0]
                b = target._structured_candidate_subset(512)[0]
                np.testing.assert_array_equal(a, b)
                source._candidate_schedule.set_cursor(3*fork_at)
                target._candidate_schedule.set_cursor(3*fork_at)
                source.rng.bit_generator.state = probe_source
                target.rng.bit_generator.state = copy.deepcopy(probe_source)
            params = policy.model.predict(problem_context)
            outcome = solver_fn(params)
            self.assertEqual(prev_update_est, 0. if policy.model.t == 0 else .004)
            policy.model.update(.05 + prev_update_est + (.01 if outcome['controller_update_committed'] else 0.),
                                failure_label=float(crossing is None))
            return params, outcome, {'overhead_sec':.002}, 0, .004
        hooks = execution.OnlineComparisonHooks(method_solver=native, run_default_setup_method=Mock(),
                                                report_online_outcome=report_online_outcome)
        with patch.object(execution, 'run_bandit_step_test_final', side_effect=bandit):
            rows, steps = execution._execute_online_instances(plan, hooks)
        self.assertEqual(steps, {'fixed_start':total, 'composite_start':total})
        self.assertEqual(len(calls), 2*total-fork_at)
        self.assertEqual(len([c for c in calls if c[0]=='fixed_start' and c[2]]), total-fixed_boundary)
        self.assertEqual(len([c for c in calls if c[0]=='composite_start' and c[2]]), 0 if crossing is None else total-crossing)
        final = rows['composite_start'][-1]['rl_activation']
        self.assertEqual(final['crossing_case'], crossing)
        self.assertEqual(final['observations'], crossing or total-1)
        audit = json.loads((output/'shared_prefix.json').read_text())
        self.assertTrue(audit['valid']); self.assertEqual(audit['candidate_schedule_cursor'],3*fork_at)
        for i in range(fork_at):
            a,b = rows['fixed_start'][i], rows['composite_start'][i]
            self.assertEqual(a['params'],b['params']); self.assertEqual(a['bandit_timing'],b['bandit_timing'])
            self.assertAlmostEqual(b['outcome']['end_to_end_runtime']-a['outcome']['end_to_end_runtime'],
                                   b['outcome']['activation_runtime'])
        self.assertFalse(np.shares_memory(source.A_inv,target.A_inv))
        self.assertEqual(source._candidate_schedule.cursor,3*total)
        self.assertEqual(target._candidate_schedule.cursor,3*total)


if __name__ == '__main__':
    unittest.main()
