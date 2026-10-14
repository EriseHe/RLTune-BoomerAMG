"""Conditional encodings, branch isolation and single fixed-RL recovery budget."""
from __future__ import annotations
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from experiments.paper_final.common.artifacts import ROOT, read
from experiments.paper_final.online import factorial
from experiments.paper_final.online.configuration import (
    CandidateScheduleSpec, parse_joint_experiment_config, runtime_config_from_spec,
)
from experiments.paper_final.online.controllers import make_setup_obs_encoder
from experiments.paper_final.online.methods import (
    build_composable_solve_runtime, build_named_setup_branches, resolve_composable_study,
)
from experiments.paper_final.online.setup_branches import build_online_linucb_branch
from experiments.paper_final.recovery.execution import run_fixed_case, EpisodeRecorder
from experiments.paper_final.recovery.retry_comparison import use_retry_rule
from experiments.paper_final.test_06_recovery import outcome
from solve.tests.test_recursive_lstdq import RecursiveLstdqEpisodeTests
from setup.space import SetupConfigurationSpace, build_setup_param_space, DEFAULT_SETUP_PARAMS
from setup.learners.common import GenericActionFeatureEncoder

CONFIG = ROOT/"experiments/paper_final/04_online/20261009_zero_prefix_diffusion60/diffusion60_seed1.json"


class FactorialTests(unittest.TestCase):
    def test_candidate_pool_configuration_rejects_inconsistent_quotas(self):
        quotas = dict(anchor=128, balanced_global=384, coordinate_local=384, sobol=128)
        parsed = CandidateScheduleSpec.from_mapping(dict(mode="aot", pool_size=1024, structured_quotas=quotas))
        self.assertEqual(parsed.pool_size, 1024)
        self.assertEqual(parsed.structured_quotas, quotas)
        self.assertIsNone(CandidateScheduleSpec.from_mapping({}).pool_size)
        for invalid in (
            dict(pool_size=1024),
            dict(pool_size=0),
            dict(pool_size=True),
            dict(pool_size=1024.5),
            dict(structured_quotas=quotas),
            dict(pool_size=1024, structured_quotas=dict(quotas, sobol=127)),
            dict(pool_size=1024, structured_quotas=dict(quotas, sobol=-128)),
            dict(pool_size=1024, structured_quotas=dict(quotas, extra=0)),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                CandidateScheduleSpec.from_mapping(invalid)

    def test_explicit_1024_pool_reaches_both_learners_for_both_families(self):
        quotas = dict(anchor=128, balanced_global=384, coordinate_local=384, sobol=128)
        for family, expected_dim in (("diffusion", 420), ("advection", 735)):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temp:
                raw = read(ROOT / f"experiments/paper_final/04_online/20261009_full_zero_prefix/{family}_60/seed_1.json")
                raw["stream"].update(cases=8, online_cases=8)
                raw["setup"]["candidate_schedule"].update(pool_size=1024, structured_quotas=quotas)
                spec = parse_joint_experiment_config(raw, output_dir_override=Path(temp))
                args = runtime_config_from_spec(spec)
                self.assertEqual(args.setup_candidate_pool_size, 1024)
                with patch.dict("os.environ", {"CANDIDATE_POOL_SIZE": "256"}):
                    setup = build_named_setup_branches(args, study=resolve_composable_study(args),
                        warmup_instances=(), stream_hash=spec.stream.expected_sha256)
                candidates = []
                for name in factorial.METHODS[2:]:
                    branch = setup.branches[name]
                    model = branch.policy.model
                    audit = factorial.audit_features(SimpleNamespace(branch=branch, spec=spec))
                    self.assertEqual(audit["regression_features"], expected_dim)
                    self.assertEqual(audit["learner"]["candidate_pool_size"], 1024)
                    self.assertEqual(audit["learner"]["structured_quotas"], quotas)
                    arms, features, counts = model._structured_candidate_subset(1024)
                    self.assertEqual(len(np.unique(arms)), 1024)
                    self.assertEqual(features.shape, (1024, 104))
                    self.assertEqual(counts["candidate_count"], 1024)
                    self.assertEqual(model.t, 0)
                    self.assertEqual(model._cand.elite_cache_size, 64)
                    candidates.append(arms)
                np.testing.assert_array_equal(*candidates)

    def space(self):
        raw = read(CONFIG)["setup"]["configuration_spaces"]["expanded_no_endpoints"]
        return SetupConfigurationSpace.from_mapping("expanded_no_endpoints", raw)

    def test_explicit_grid_roundtrip_and_conditional_catalog(self):
        space = self.space()
        self.assertEqual(SetupConfigurationSpace.from_mapping(space.name, space.as_dict()), space)
        with patch.dict("os.environ", dict(SETUP_GRID_MAX="not-a-grid", P_MAX_ELMTS_VALUES="invalid", AGG_NUM_LEVELS_VALUES="invalid")):
            built = build_setup_param_space(tune_dim=7, configuration_space=space, parameter_resolution=20, materialize=False)
        self.assertEqual(len(built.actions), 16128000)
        self.assertFalse(built.actions.has_appended_default)
        self.assertEqual(built.actions[built.default_arm_index], DEFAULT_SETUP_PARAMS)
        encoder = GenericActionFeatureEncoder(built.parameter_spec)
        self.assertEqual(encoder.feature_dim, 104)
        families = []
        for depth in (0, 1, 2):
            for interp in (4, 6, 7):
                params = dict(DEFAULT_SETUP_PARAMS, agg_num_levels=depth, agg_interp_type=interp)
                arm = built.actions.index_of(params)
                action = built.actions[arm]
                self.assertEqual(action["agg_interp_type"], interp if depth else 4)
                families.append(arm)
        self.assertEqual(len(set(families)), 7)
        self.assertNotIn(0, space.numeric_grids["P_max_elmts"])
        self.assertNotIn(1, space.numeric_grids["max_row_sum"])

    def test_solve_and_setup_encoders_mask_inactive_but_distinguish_active_modes(self):
        space = self.space()
        encoder = make_setup_obs_encoder(configuration_space=space, parameter_resolution=20, strict_categories=True)
        baseline = encoder.encode(DEFAULT_SETUP_PARAMS)
        self.assertEqual(len(baseline), 8)
        for interp in (4, 6, 7):
            np.testing.assert_array_equal(encoder.encode(dict(DEFAULT_SETUP_PARAMS, agg_interp_type=interp)), baseline)
        for depth in (1, 2):
            active = [encoder.encode(dict(DEFAULT_SETUP_PARAMS, agg_num_levels=depth, agg_interp_type=interp)).tobytes() for interp in (4, 6, 7)]
            self.assertEqual(len(set(active)), 3)
        ordinary = [encoder.encode(dict(DEFAULT_SETUP_PARAMS, interp_type=interp)).tobytes() for interp in space.interp_types]
        self.assertEqual(len(set(ordinary)), 8)
        with self.assertRaises(ValueError):
            encoder.encode(dict(DEFAULT_SETUP_PARAMS, agg_num_levels=1, agg_interp_type=99))

    def test_optimized_features_and_uncertainty_match_dense_formulas(self):
        with tempfile.TemporaryDirectory() as temp:
            branch, _ = build_online_linucb_branch(seed=123, configuration_space=self.space(),
                action_space_mode="full_cartesian", parameter_resolution=20,
                candidate_schedule_dir=Path(temp), candidate_schedule_rounds=24,
                candidate_sampling="structured512", context_dim=4, context_interaction_indices=(1,2,3))
            runtime = SimpleNamespace(branch=branch)
            result = factorial.audit_features(runtime)
            self.assertEqual(result["regression_features"], 420)
            self.assertEqual(result["categorical_candidate_families"], 336)
            model = use_retry_rule(branch.policy.model, "failure_model_adjacent")
            x = np.array([1., .2, .3, .4])
            model.predict(x)
            observation = model.defer_cost(False)
            model.resolve_cost(observation, .3)
            arms = np.arange(100)
            mean, width = model._mean_uncertainty_subset(x, arms=arms)
            phi = np.asarray([model._phi(x, int(arm)) for arm in arms])
            np.testing.assert_allclose(mean, phi @ model.A_inv @ model.b, rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(width, np.sqrt(np.einsum('ij,jk,ik->i', phi, model.A_inv, phi)), rtol=1e-11, atol=1e-12)

    def test_two_controller_instances_have_matched_encodings_and_distinct_state(self):
        spec = parse_joint_experiment_config(read(CONFIG))
        methods = factorial.validate_methods(spec)
        bundles = build_composable_solve_runtime(runtime_config_from_spec(spec), specs=spec.methods).controller_bundles
        self.assertEqual(set(bundles), {"default_setup_lstdq", "bandit_lstdq"})
        first, joint = (bundles[name] for name in ("default_setup_lstdq", "bandit_lstdq"))
        self.assertEqual(first.protocol_metadata()["state_encoder"], joint.protocol_metadata()["state_encoder"])
        self.assertFalse(np.shares_memory(first.controller.b, joint.controller.b))
        self.assertIsNot(first.controller.rng, joint.controller.rng)
        self.assertEqual(methods["default_setup_lstdq"].solve_activation_case, 0)
        self.assertEqual(methods["bandit_lstdq"].solve_activation_case, 0)
        self.assertEqual(first.controller.spec.uncertainty_beta, 2.)

    def test_delayed_activation_is_rejected(self):
        for name in ("default_setup_lstdq", "bandit_lstdq"):
            raw = read(CONFIG)
            next(m for m in raw["methods"] if m["id"] == name)["solve_activation_case"] = 1000
            with self.assertRaisesRegex(ValueError, "first input"):
                factorial.validate_methods(parse_joint_experiment_config(raw))

    def test_balanced_order_and_reproducibility(self):
        orders = factorial.balanced_orders(240, 123)
        self.assertEqual(orders, factorial.balanced_orders(240, 123))
        self.assertNotEqual(orders, factorial.balanced_orders(240, 124))
        for offset in range(0,240,24):
            block = orders[offset:offset+24]
            self.assertEqual(len(set(block)),24)
            for position in range(4):
                for name in factorial.METHODS:
                    self.assertEqual(sum(order[position] == name for order in block),6)

    def test_fixed_rl_one_primary_then_fallback_and_whole_failure_rollback(self):
        for status, fallback_status in (("success",None), ("nonconvergence","success"), ("nonconvergence","nonconvergence")):
            with self.subTest(status=status, fallback=fallback_status):
                controller = RecursiveLstdqEpisodeTests._controller()
                recorder = EpisodeRecorder(controller)
                initial = controller.b.copy()
                calls = []
                def solve():
                    calls.append("primary")
                    recorder.start_episode(initial_environment_weight=1.)
                    recorder.update(features=np.ones(3), action_index=0, cost=.02,
                        next_features=np.ones(3), terminal=True)
                    recorder.finish_episode(learned=True)
                    return outcome(status)
                def fallback():
                    calls.append("fallback")
                    self.assertIsNotNone(fallback_status)
                    return outcome(fallback_status)
                result = run_fixed_case(solve, recorder, 0., False, fallback_solver=fallback, max_attempts=1)
                self.assertEqual(calls,["primary"] if status=="success" else ["primary","fallback"])
                self.assertEqual(result["primary_attempt_count"],1)
                self.assertFalse(result["fixed_setup_retries"])
                self.assertFalse(result["bandit_update_committed"])
                self.assertEqual(result["replayed_transitions"],0)
                self.assertEqual(controller.steps,0 if fallback_status=="nonconvergence" else 1)
                if fallback_status=="nonconvergence":
                    np.testing.assert_array_equal(controller.b,initial)


if __name__ == "__main__":
    unittest.main()
