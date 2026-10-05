"""Paper context/activation regressions using features and mocked solves only."""
from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import experiments.joint.solve_control.joint_4k_execution as execution
from experiments.joint.solve_control.composable_joint_4k import (
    build_composable_solve_runtime,
    build_named_setup_branches,
    resolve_composable_study,
)
from experiments.joint.solve_control.joint_experiment_config import parse_joint_experiment_config, runtime_config_from_spec
from experiments.joint.solve_control.joint_experiment_plotting import _compact_method_labels
from experiments.joint.solve_control.joint_method_spec import ComposableMethodSpec
from experiments.joint.solve_control.joint_online_common import report_online_outcome, validate_recovery_stream
from experiments.joint.solve_control.joint_rl_activation import ReliabilityActivationGate, ReliabilityActivationSpec
from problems.amg import compact_diffusion_context
from problems.registry import context_for_setup_method, learning_context_for_setup
from setup.space import DEFAULT_SETUP_PARAMS
from setup.learners.linucb.SharedLinUCB_AMG_v4 import SharedLinUCBv4Step


CONFIGS = Path(__file__).resolve().parents[2] / "joint/solve_control/configs"
SOURCE = "paper_final_n60_canonical8d_lstdq_v3_seed_stability_5k_encoderfix.json"
PRESET = "paper_test_n60_context_activation.json"


class PaperContextActivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads((CONFIGS / SOURCE).read_text())
        cls.raw = json.loads((CONFIGS / PRESET).read_text())
        cls.runtime = runtime_config_from_spec(parse_joint_experiment_config(cls.raw))

    def test_seed_c_baseline_and_common_experiment_controls_are_preserved(self):
        for key in ("problem", "stream", "seeds", "setup", "solve"):
            self.assertEqual(self.raw[key], self.source[key], key)
        baseline = {**self.raw["methods"][1], "id": self.source["methods"][3]["id"]}
        self.assertEqual(baseline, self.source["methods"][3])
        self.assertEqual(len(self.runtime.methods), 6)
        for spec in self.runtime.methods[1:]:
            self.assertEqual(spec.seed_offset, 2000029)
            self.assertEqual(spec.solve_kind, "recursive_lstdq_v3")
        for spec in self.runtime.methods:
            self.assertEqual(ComposableMethodSpec.from_runner_token(spec.to_runner_token()), spec)

    def test_compact_contexts_keep_bias_and_directional_information(self):
        x = np.array([1., .2, .4, .9, .5, 0., 0., 0.])
        for mode, size in (("diffusion3d", 4), ("diffusion4d", 5)):
            context = context_for_setup_method(
                problem_kind="scalar_anisotropic_diffusion", setup_kind="linucb",
                setup_context=mode, matrix_kwargs={}, stream_context=x,
            )
            np.testing.assert_array_equal(context, x[:size])
            contract = learning_context_for_setup("scalar_anisotropic_diffusion", "linucb", mode)
            self.assertEqual(contract.dimension, size)
            self.assertEqual(contract.interaction_indices, tuple(range(1, size)))
        np.testing.assert_array_equal(context_for_setup_method(
            problem_kind="scalar_anisotropic_diffusion", setup_kind="linucb_v5",
            matrix_kwargs={}, stream_context=x,
        ), x)
        with self.assertRaisesRegex(ValueError, "frozen LinUCB v5"):
            learning_context_for_setup("scalar_anisotropic_diffusion", "linucb_v5", "diffusion3d")
        nonzero = x.copy(); nonzero[-1] = .1
        with self.assertRaisesRegex(ValueError, "zero advection"):
            compact_diffusion_context(nonzero, mode="diffusion3d")

    def test_solve_contexts_share_the_projection_and_preserve_other_state(self):
        bundles = build_composable_solve_runtime(self.runtime, specs=self.runtime.methods).controller_bundles
        inputs = dict(
            mkw={"nx": 60, "ny": 60, "nz": 60, "k": 100, "c": 10, "a0": 1},
            initial_residual=1., residual=.02, previous_residual=.1,
            cycle=4, last_weight=2.7, last_cycle_time=.004,
            setup_params={**DEFAULT_SETUP_PARAMS, "coarsen_type": 3, "interp_type": 17},
            problem_context=np.array([1., 2/3, 1/3, 0., 1/3, 0., 0., 0.]),
        )
        full = bundles["context8d_fixed"].encoder.encode(**inputs)
        self.assertEqual(full.size, 35)
        for mode, coefficients in (("3d", 3), ("4d", 4)):
            target = np.concatenate((full[:7 + coefficients], full[14:]))
            for start in ("fixed", "dynamic"):
                name = f"context{mode}_{start}"
                bundle = bundles[name]
                np.testing.assert_array_equal(bundle.encoder.encode(**inputs), target)
                self.assertEqual(bundle.controller.joint_dim, 9 * (28 + coefficients))
        # Identical seed streams and exploration schedules for every controller.
        reference_rng = bundles["context8d_fixed"].controller.rng.bit_generator.state
        for bundle in bundles.values():
            self.assertEqual(bundle.controller.rng.bit_generator.state, reference_rng)
        old_runtime = runtime_config_from_spec(parse_joint_experiment_config(self.source))
        old = build_composable_solve_runtime(old_runtime, specs=old_runtime.methods).controller_bundles[
            self.source["methods"][3]["id"]
        ]
        new = bundles["context8d_fixed"]
        np.testing.assert_array_equal(old.encoder.encode(**inputs), full)
        self.assertEqual(old.controller.config, new.controller.config)
        self.assertEqual(old.controller.spec, new.controller.spec)
        # A deterministic TD transition sequence must leave the reproduced 8D
        # branch's value estimates, covariance and RNG state exactly unchanged.
        for episode in range(3):
            for bundle in (old, new):
                bundle.controller.start_episode(initial_environment_weight=1.)
                bundle.controller.update(features=full, action_index=episode + 2,
                    cost=.003 * (episode + 1), next_features=full, terminal=True)
                bundle.controller.finish_episode(learned=True)
            np.testing.assert_array_equal(old.controller.theta, new.controller.theta)
            for a, b in zip(old.controller._values(full, cycle=4), new.controller._values(full, cycle=4)):
                np.testing.assert_array_equal(a, b)

    def test_setup_branches_build_with_independent_state_and_paired_rngs(self):
        study = resolve_composable_study(self.runtime)
        with tempfile.TemporaryDirectory() as tmp:
            runtime = replace(self.runtime, output_dir=Path(tmp), online_cases=2)
            artifacts = build_named_setup_branches(
                runtime, study=study, warmup_instances=[], stream_hash='test-only'
            )
            models = {name: branch.policy.model for name, branch in artifacts.branches.items()}
            sizes = (8, 4, 5, 4, 5)
            self.assertEqual([model.d_x for model in models.values()], list(sizes))
            self.assertEqual(len({id(model.A_inv) for model in models.values()}), 5)
            rng = models['context8d_fixed'].rng.bit_generator.state
            for model in models.values():
                self.assertEqual(model.rng.bit_generator.state, rng)
            # Removing zero coordinates from 8D must preserve ridge predictions
            # and confidence widths on the same feedback and action sequence.
            full = models['context8d_fixed']; compact = models['context4d_fixed']
            x = np.array([1., .2, .4, .9, .5, 0., 0., 0.])
            arms = np.array([0, 5, 17])
            for arm in arms:
                for model, context in ((full, x), (compact, x[:5])):
                    model._last_phi = model._phi(context, int(arm))
                    model._last_arm = int(arm)
                    model.history.append(SharedLinUCBv4Step(
                        t=model.t + 1, arm_index=int(arm), loss=float('nan'),
                        pred_mean=float('nan'), pred_uncert=float('nan'),
                    ))
                    model.update(.02 + .001*int(arm))
                for left, right in zip(
                    full._score_subset(x, arms=arms, alpha=1.),
                    compact._score_subset(x[:5], arms=arms, alpha=1.),
                ):
                    np.testing.assert_allclose(left, right, rtol=1e-11, atol=1e-12)

    def test_log_recursion_matches_the_explicit_onset_mixture(self):
        spec = ReliabilityActivationSpec(p_bad=.2, p_good=.05, delta=1e-12)
        gate = ReliabilityActivationGate(spec)
        factors = []
        for failed in np.random.default_rng(73).integers(0, 2, 60):
            factors.append(spec.p_good/spec.p_bad if failed else (1-spec.p_good)/(1-spec.p_bad))
            gate.observe(first_attempt_failed=bool(failed))
            t = len(factors)
            direct = sum(math.prod(factors[k-1:]) / (k*(k+1)) for k in range(1, t+1)) + 1/(t+1)
            self.assertAlmostEqual(math.exp(gate.log_evalue), direct, places=13)

    def test_no_deadline_and_no_post_activation_monitoring(self):
        spec = ReliabilityActivationSpec(p_bad=.8, p_good=.1, delta=.5)
        gate = ReliabilityActivationGate(spec)
        for _ in range(2000):
            gate.observe(first_attempt_failed=True)
        self.assertFalse(gate.active)
        self.assertTrue(math.isfinite(gate.log_evalue))
        success = ReliabilityActivationGate(spec)
        success.observe(first_attempt_failed=False)
        self.assertTrue(success.active)
        self.assertEqual(success.summary()["first_rl_case"], 2)
        with self.assertRaisesRegex(RuntimeError, "stop observing"):
            success.observe(first_attempt_failed=False)

    def test_invalid_activation_rules_and_mixed_protocols_fail_early(self):
        for bad, good, delta in ((.05,.05,.05),(.01,.05,.05),(1.,.01,.05),(.05,.01,0.),(.05,.01,float('nan'))):
            with self.subTest(values=(bad,good,delta)), self.assertRaises(ValueError):
                ReliabilityActivationSpec(p_bad=bad,p_good=good,delta=delta)
        dynamic = self.raw["methods"][-1]
        for changes in ({"solve_activation_case": 1000}, {"setup_warmup_cases": 10}, {"solve": "default"}):
            with self.assertRaises(ValueError):
                ComposableMethodSpec.from_mapping({**dynamic, **changes})

    def test_mixed_runner_counts_first_attempts_and_enables_only_next_problem(self):
        specs = tuple(replace(s, solve_activation_case=2 if s.solve_activation_case else 0,
            solve_activation=ReliabilityActivationSpec(.8,.1,.5) if s.solve_activation else None)
            for s in self.runtime.methods)
        specs_by_name = {s.name:s for s in specs}
        calls = {s.name:[] for s in specs}
        setup_views = {}
        x = np.array([1., .2, .4, .9, .5, 0., 0., 0.])
        def native(*, mkw, active=False, **kwargs):
            method = mkw["method"]; index=mkw["index"]
            calls[method].append(active)
            failed_first = method == "context4d_dynamic" or (method == "context3d_dynamic" and index == 0)
            return dict(runtime=.05, native_runtime=.05, setup_runtime=.03,
                solve_runtime=.02, native_solve_runtime=.02, infer_runtime=.001 if active else 0.,
                failed=False, iterations=2, native_status="converged", primary_status="success",
                first_primary_status="nonconvergence" if failed_first else "success",
                recovered=failed_first, fallback_status="not_run", fallback_used=False,
                bandit_update_committed=True, controller_update_committed=active)
        bundles = {s.name: Mock() for s in specs if s.solve_kind != "default"}
        for bundle in bundles.values():
            bundle.run_case.side_effect = lambda case: native(mkw=case.mkw, active=True)
            bundle.summary.return_value = {}
        def fake_bandit(*, policy, problem_context, solver_fn, **kwargs):
            setup_views[policy.name] = problem_context
            return dict(DEFAULT_SETUP_PARAMS), solver_fn(dict(DEFAULT_SETUP_PARAMS)), {"overhead_sec":.002}, 0, 0.
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp); trajectories=output/'trajectories';trajectories.mkdir()
            plan = execution.OnlineComparisonPlan(
                output_dir=output, trajectories_dir=trajectories, checkpoints_dir=output/'checkpoints',
                methods=tuple(specs_by_name), family_by_method={s.name:s.family for s in specs},
                bandit_methods=tuple(s.name for s in specs if s.setup_kind != 'default'),
                online_instances=[({'index':i},x) for i in range(8)], composable_specs=specs_by_name,
                branches={s.name:SimpleNamespace(policy=SimpleNamespace(name=s.name),parameter_space={}) for s in specs if s.setup_kind!='default'},
                controller_bundles=bundles, ppo_runner=None, protocol={},
                warmup=execution.WarmupArtifacts({},'',{},0),
                solve=execution.SolveExecutionConfig(1e-6,50), warmup_cases=0, online_cases=8,
                method_order_seed=19, progress_every=8, aot_enabled=False,
                aot_max_selections_per_case=3, default_setup_method=specs[0].name,
                include_solve_screen_report=False,
            )
            def method_solver(method, **kwargs):
                kwargs['mkw']={**kwargs['mkw'], 'method':method}
                return execution.make_method_solver(method, solve=plan.solve,
                    controller_bundles=bundles, ppo_runner=None, composable_specs=specs_by_name, **kwargs)
            hooks=execution.OnlineComparisonHooks(
                method_solver=method_solver,
                run_default_setup_method=lambda solver_fn, **kw: report_online_outcome(solver_fn(dict(DEFAULT_SETUP_PARAMS)),bandit_timing={}),
                report_online_outcome=report_online_outcome,
                setup_context=lambda method,mkw,context: context_for_setup_method(
                    problem_kind='scalar_anisotropic_diffusion', setup_kind=specs_by_name[method].setup_kind,
                    setup_context=specs_by_name[method].setup_context, matrix_kwargs=mkw, stream_context=context),
            )
            with patch.object(execution,'solve_no_rl_case',side_effect=native), patch.object(execution,'run_bandit_step_test_final',side_effect=fake_bandit):
                records, steps = execution._execute_online_instances(plan,hooks)
            for name in ('context8d_fixed','context3d_fixed','context4d_fixed'):
                self.assertEqual(calls[name], [False]*2+[True]*6)
                self.assertNotIn('rl_activation',records[name][0])
            self.assertEqual(calls['context3d_dynamic'],[False]*3+[True]*5)
            self.assertEqual(calls['context4d_dynamic'],[False]*8)
            gate=records['context3d_dynamic'][-1]['rl_activation']
            self.assertEqual(gate['crossing_case'],3)
            self.assertEqual(gate['observations'],3)
            self.assertEqual(gate['first_attempt_failures'],1)
            self.assertEqual(gate['first_rl_case'],4)
            self.assertTrue((output/'rl_activation.json').is_file())
            for name in steps:
                self.assertEqual(steps[name],8)
                self.assertTrue(validate_recovery_stream(records[name],expect_bandit_transaction=True)['valid'])
            np.testing.assert_array_equal(setup_views['context3d_dynamic'],x[:4])
            np.testing.assert_array_equal(setup_views['context4d_dynamic'],x[:5])
            result = execution._build_result(
                plan, records=records, bandit_online_steps=steps,
                recovery_audit={}, final_bandit_dir=output/'final_bandit_states',
            )
            self.assertEqual(result['rl_activation']['context3d_dynamic']['rl_problem_count'],5)
            self.assertEqual(result['rl_activation']['context4d_dynamic']['rl_problem_count'],0)
            self.assertIn('vs_context8d_fixed',result['windows']['all_8']['comparisons'])

    def test_new_plot_labels_distinguish_context_and_activation(self):
        protocol={'method_specs':[asdict(s) for s in self.runtime.methods],
                  'methods':[s.name for s in self.runtime.methods]}
        labels=_compact_method_labels(protocol)
        self.assertEqual(labels['context8d_fixed'],'LinUCB v5 (8D) / start 1001')
        self.assertEqual(labels['context3d_dynamic'],'LinUCB (4D) / dynamic start')
        self.assertEqual(labels['context4d_dynamic'],'LinUCB (5D) / dynamic start')
        self.assertEqual(len(set(labels.values())),6)


if __name__ == '__main__':
    unittest.main()
