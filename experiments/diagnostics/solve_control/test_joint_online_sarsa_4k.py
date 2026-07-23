from __future__ import annotations

import _project_paths  # noqa: F401

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import legacy_joint_studies as legacy_studies
from run_joint_online_sarsa_4k import (
    BEHAVIOR_MODES,
    BATCHED_LSVI_METHOD,
    LSVI_METHOD,
    LSVI_METHODS,
    RECURSIVE_LCB_METHODS,
    RECURSIVE_LCB_PPO_METHODS,
    RECURSIVE_LSTDQ_METHOD,
    RECURSIVE_LSTDQ_METHODS,
    RECURSIVE_LSTDQ_V2_METHOD,
    RECURSIVE_MC_METHOD,
    REFERENCE_METHODS,
    RECALIBRATED_LSVI_METHOD,
    SOLVE_CONTROLLER_SEED_OFFSETS,
    SOLVE_CONTROLLER_SCREEN_METHODS,
    STRUCTURED_MODEL_BASED_METHOD,
    _comparison_windows,
    _empty_stream_summary,
    _make_lsvi_controller,
    _make_recursive_lstdq_controller,
    _make_recursive_lstdq_v2_controller,
    _make_recalibrated_lsvi_controller,
    _make_structured_model_based_controller,
    _validate_lsvi_protocol,
    _write_solve_screen_report,
    candidate_grid,
)
from joint_online_common import _build_paired_instance_stream


class JointOnlineSarsa4KTests(unittest.TestCase):
    @staticmethod
    def _lsvi_args() -> SimpleNamespace:
        return SimpleNamespace(
            tol=1.0e-6,
            max_cycles=50,
            c_max=1000.0,
            weights="1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0",
            action_rbf_centers="1.0,1.25,1.5,1.75,2.0",
            action_rbf_sigma=0.2,
            epsilon_start=0.3,
            epsilon_final=0.03,
            epsilon_decay_steps=20_000.0,
            lsvi_ridge=1.0,
            lsvi_beta=2.0,
            lsvi_residual_floor_sec=1.0e-3,
            lsvi_refit_interval_episodes=100,
            recursive_mc_ridge=1.0,
            recursive_mc_beta=2.0,
            recursive_mc_episode_half_life=500.0,
            recursive_lstdq_ridge=1.0,
            recursive_lstdq_beta=2.0,
            recursive_lstdq_lambda=0.8,
            recursive_lstdq_residual_floor_sec=1.0e-3,
            recursive_lstdq_lcb_lower_bound_sec=0.0,
            recursive_lstdq_v2_beta=2.0,
            recursive_lstdq_v2_coverage_ridge=1.0,
            recursive_lstdq_v2_residual_window=2048,
            recursive_lstdq_v2_min_samples=32,
            structured_model_ridge=1.0,
            structured_model_min_samples=32,
            structured_model_scale_window=2048,
            recalibrated_lsvi_beta=2.0,
            recalibrated_lsvi_refit_sweeps=3,
            recalibrated_lsvi_shrinkage_samples=32.0,
        )

    def test_two_behaviors_cross_six_alpha_lambda_candidates(self) -> None:
        candidates = candidate_grid(
            alphas=(0.001, 0.005),
            trace_lambdas=(0.2, 0.5, 0.8),
        )
        self.assertEqual(len(candidates), 12)
        self.assertEqual({candidate.behavior_mode for candidate in candidates}, set(BEHAVIOR_MODES))
        self.assertEqual(len({candidate.name for candidate in candidates}), 12)
        self.assertEqual(len(REFERENCE_METHODS) + len(candidates), 15)

    def test_selected_configuration_has_one_candidate_per_behavior(self) -> None:
        candidates = candidate_grid(
            alphas=(0.001,),
            trace_lambdas=(0.8,),
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            {candidate.behavior_mode for candidate in candidates},
            set(BEHAVIOR_MODES),
        )
        self.assertTrue(
            all(candidate.alpha == 0.001 for candidate in candidates)
        )
        self.assertTrue(
            all(candidate.trace_lambda == 0.8 for candidate in candidates)
        )

    def test_legacy_study_resolver_owns_every_frozen_roster(self) -> None:
        expected_methods = {
            "lsvi_lcb": LSVI_METHODS,
            "recursive_lcb_suite": RECURSIVE_LCB_METHODS,
            "recursive_lcb_ppo": RECURSIVE_LCB_PPO_METHODS,
            "recursive_lstdq_lcb": RECURSIVE_LSTDQ_METHODS,
            "solve_controller_screen": SOLVE_CONTROLLER_SCREEN_METHODS,
        }
        for study_mode, methods in expected_methods.items():
            with self.subTest(study_mode=study_mode):
                study = legacy_studies.resolve_legacy_study(
                    SimpleNamespace(
                        study_mode=study_mode,
                        include_default_setup_baseline=False,
                    )
                )
                self.assertEqual(study.methods, methods)
                self.assertEqual(study.candidates, ())
                self.assertEqual(study.bandit_methods, methods)
                self.assertEqual(
                    tuple(study.family_by_method),
                    methods,
                )

        sarsa = legacy_studies.resolve_legacy_study(
            SimpleNamespace(
                study_mode="sarsa",
                include_default_setup_baseline=False,
                alphas="0.001",
                trace_lambdas="0.8",
            )
        )
        self.assertEqual(len(sarsa.candidates), len(BEHAVIOR_MODES))
        self.assertEqual(
            sarsa.methods,
            (
                *REFERENCE_METHODS,
                *[candidate.name for candidate in sarsa.candidates],
            ),
        )

    def test_legacy_recursive_runtime_delegates_to_bundle_helpers(self) -> None:
        args = SimpleNamespace(
            study_mode="recursive_lcb_suite",
            include_default_setup_baseline=False,
            controller_seed=100,
            lsvi_refit_interval_episodes=37,
        )
        study = legacy_studies.resolve_legacy_study(args)
        mc_bundle = object()
        lstdq_bundle = object()
        lsvi_bundle = object()
        with (
            patch.object(
                legacy_studies,
                "build_recursive_mc_controller_bundle",
                return_value=mc_bundle,
            ) as build_mc,
            patch.object(
                legacy_studies,
                "build_recursive_lstdq_controller_bundle",
                return_value=lstdq_bundle,
            ) as build_lstdq,
            patch.object(
                legacy_studies,
                "build_lsvi_controller_bundle",
                return_value=lsvi_bundle,
            ) as build_lsvi,
        ):
            runtime = legacy_studies.build_legacy_solve_runtime(
                args,
                study=study,
                make_ppo_runner=Mock(),
            )

        self.assertEqual(
            runtime.controller_bundles,
            {
                RECURSIVE_MC_METHOD: mc_bundle,
                RECURSIVE_LSTDQ_METHOD: lstdq_bundle,
                BATCHED_LSVI_METHOD: lsvi_bundle,
            },
        )
        build_mc.assert_called_once_with(args, seed=100)
        build_lstdq.assert_called_once_with(args, seed=1109)
        build_lsvi.assert_called_once_with(
            args,
            seed=2118,
            refit_interval_episodes=37,
        )
        self.assertIsNone(runtime.ppo_runner)

    def test_legacy_lsvi_runtime_keeps_per_episode_refit_override(self) -> None:
        args = SimpleNamespace(
            study_mode="lsvi_lcb",
            include_default_setup_baseline=False,
            controller_seed=250,
        )
        study = legacy_studies.resolve_legacy_study(args)
        bundle = object()
        with patch.object(
            legacy_studies,
            "build_lsvi_controller_bundle",
            return_value=bundle,
        ) as build_lsvi:
            runtime = legacy_studies.build_legacy_solve_runtime(
                args,
                study=study,
                make_ppo_runner=Mock(),
            )

        self.assertEqual(
            runtime.controller_bundles,
            {LSVI_METHOD: bundle},
        )
        build_lsvi.assert_called_once_with(
            args,
            seed=250,
            refit_interval_episodes=1,
        )

    def test_legacy_screen_runtime_preserves_seed_offsets(self) -> None:
        args = SimpleNamespace(
            study_mode="solve_controller_screen",
            include_default_setup_baseline=False,
            controller_seed=400,
        )
        study = legacy_studies.resolve_legacy_study(args)
        patched_factories = {
            RECURSIVE_LSTDQ_METHOD: (
                "build_recursive_lstdq_controller_bundle"
            ),
            RECURSIVE_LSTDQ_V2_METHOD: (
                "build_recursive_lstdq_v2_controller_bundle"
            ),
            STRUCTURED_MODEL_BASED_METHOD: (
                "build_structured_model_based_controller_bundle"
            ),
            RECALIBRATED_LSVI_METHOD: (
                "build_recalibrated_lsvi_controller_bundle"
            ),
        }
        mocks = {
            method: Mock(return_value=object())
            for method in patched_factories
        }
        with (
            patch.object(
                legacy_studies,
                patched_factories[RECURSIVE_LSTDQ_METHOD],
                mocks[RECURSIVE_LSTDQ_METHOD],
            ),
            patch.object(
                legacy_studies,
                patched_factories[RECURSIVE_LSTDQ_V2_METHOD],
                mocks[RECURSIVE_LSTDQ_V2_METHOD],
            ),
            patch.object(
                legacy_studies,
                patched_factories[STRUCTURED_MODEL_BASED_METHOD],
                mocks[STRUCTURED_MODEL_BASED_METHOD],
            ),
            patch.object(
                legacy_studies,
                patched_factories[RECALIBRATED_LSVI_METHOD],
                mocks[RECALIBRATED_LSVI_METHOD],
            ),
        ):
            runtime = legacy_studies.build_legacy_solve_runtime(
                args,
                study=study,
                make_ppo_runner=Mock(),
            )

        self.assertEqual(
            set(runtime.controller_bundles),
            set(patched_factories),
        )
        for method, factory in mocks.items():
            factory.assert_called_once_with(
                args,
                seed=400 + SOLVE_CONTROLLER_SEED_OFFSETS[method],
            )

    def test_legacy_sarsa_runtime_builds_bundles_and_ppo_once(self) -> None:
        args = SimpleNamespace(
            study_mode="sarsa",
            include_default_setup_baseline=False,
            alphas="0.001",
            trace_lambdas="0.8",
            controller_seed=700,
        )
        study = legacy_studies.resolve_legacy_study(args)
        build_bundle = Mock(side_effect=(object(), object()))
        ppo_runner = object()
        make_ppo = Mock(return_value=ppo_runner)
        with patch.object(
            legacy_studies,
            "build_sarsa_controller_bundle",
            build_bundle,
        ):
            runtime = legacy_studies.build_legacy_solve_runtime(
                args,
                study=study,
                make_ppo_runner=make_ppo,
            )

        self.assertEqual(
            list(runtime.controller_bundles),
            [candidate.name for candidate in study.candidates],
        )
        self.assertEqual(
            [call.kwargs["seed"] for call in build_bundle.call_args_list],
            [700, 1709],
        )
        make_ppo.assert_called_once_with(args)
        self.assertIs(runtime.ppo_runner, ppo_runner)

    def test_locked_4k_stream_hash_and_partition(self) -> None:
        args = SimpleNamespace(
            train_seed_groups=(
                "39800039,39806039,39812039,39818039,"
                "39824039,39830039,39836039,39842039"
            ),
            train_shuffle_seeds="39848039",
            train_cases_per_seed=500,
            train_group_take=4000,
            train_cases=4000,
            instance_offset=0,
            grid_n=40,
            c_min=1.0,
            c_max=1000.0,
            seed=39860039,
        )
        stream, manifest = _build_paired_instance_stream(args)
        self.assertEqual(len(stream[:2000]), 2000)
        self.assertEqual(len(stream[2000:]), 2000)
        self.assertEqual(
            manifest["sha256"],
            "64b8b0d63e59c7addb835e671b4dbcf498effa9c79084e3218e52bc9a92ec03f",
        )
        self.assertEqual(manifest["segments"][0]["window"], [0, 500])

    def test_joint_from_scratch_windows_cover_the_full_4k_stream(self) -> None:
        windows = _comparison_windows(4000)
        self.assertEqual(windows["all_4000"], (0, 4000))
        self.assertEqual(windows["first_2000"], (0, 2000))
        self.assertEqual(windows["last_2000"], (2000, 4000))
        self.assertEqual(windows["last_1000"], (3000, 4000))

    def test_solve_controller_screen_uses_canonical_60_cubed_stream(self) -> None:
        args = SimpleNamespace(
            train_seed_groups=(
                "40800039,40806039,40812039,40818039,"
                "40824039,40830039,40836039,40842039"
            ),
            train_shuffle_seeds="40848039",
            train_cases_per_seed=500,
            train_group_take=4000,
            train_cases=4000,
            instance_offset=0,
            grid_n=60,
            c_min=1.0,
            c_max=1000.0,
            seed=40800039,
        )
        stream, manifest = _build_paired_instance_stream(args)
        self.assertEqual(len(stream), 4000)
        self.assertEqual(
            manifest["sha256"],
            "156e6fdbbed6733d98e2c5f7e550e5d230c45217d0833ac64567563b5434459b",
        )

    def test_empty_warmup_summary_is_finite_and_zero(self) -> None:
        summary = _empty_stream_summary()
        self.assertEqual(summary["cases"], 0)
        self.assertEqual(summary["failed_count"], 0)
        self.assertTrue(all(value == 0.0 for value in summary["means_sec"].values()))
        self.assertTrue(all(value == 0.0 for value in summary["totals_sec"].values()))

    def test_lsvi_mode_has_only_three_requested_methods(self) -> None:
        self.assertEqual(
            LSVI_METHODS,
            ("bandit_default", "bandit_fixed_w1.6", LSVI_METHOD),
        )

    def test_recursive_lcb_suite_has_exactly_five_requested_methods(self) -> None:
        self.assertEqual(
            RECURSIVE_LCB_METHODS,
            (
                "bandit_default",
                "bandit_fixed_w1.6",
                RECURSIVE_MC_METHOD,
                RECURSIVE_LSTDQ_METHOD,
                BATCHED_LSVI_METHOD,
            ),
        )

    def test_recursive_lcb_ppo_has_exactly_five_requested_methods(self) -> None:
        self.assertEqual(
            RECURSIVE_LCB_PPO_METHODS,
            (
                "bandit_default",
                "bandit_fixed_w1.6",
                "bandit_ppo",
                RECURSIVE_MC_METHOD,
                RECURSIVE_LSTDQ_METHOD,
            ),
        )

    def test_recursive_lstdq_mode_has_only_three_requested_methods(self) -> None:
        self.assertEqual(
            RECURSIVE_LSTDQ_METHODS,
            (
                "bandit_default",
                "bandit_fixed_w1.6",
                RECURSIVE_LSTDQ_METHOD,
            ),
        )

    def test_solve_controller_screen_has_exactly_five_requested_methods(self) -> None:
        self.assertEqual(
            SOLVE_CONTROLLER_SCREEN_METHODS,
            (
                "bandit_fixed_w1.6",
                RECURSIVE_LSTDQ_METHOD,
                RECURSIVE_LSTDQ_V2_METHOD,
                STRUCTURED_MODEL_BASED_METHOD,
                RECALIBRATED_LSVI_METHOD,
            ),
        )
        self.assertEqual(
            SOLVE_CONTROLLER_SEED_OFFSETS,
            {
                RECURSIVE_LSTDQ_METHOD: 1009,
                RECURSIVE_LSTDQ_V2_METHOD: 2018,
                STRUCTURED_MODEL_BASED_METHOD: 3027,
                RECALIBRATED_LSVI_METHOD: 4036,
            },
        )

    def test_new_controller_factories_share_encoder_and_action_grid(self) -> None:
        args = self._lsvi_args()
        args.shared_action_profile = "1to3_step0p05"
        args.weights = None
        args.action_rbf_centers = None
        _validate_lsvi_protocol(args)
        factories = (
            _make_recursive_lstdq_v2_controller,
            _make_structured_model_based_controller,
            _make_recalibrated_lsvi_controller,
        )
        for offset, factory in enumerate(factories):
            controller, encoder = factory(args, seed=40866039 + offset)
            self.assertEqual(controller.feature_dim, encoder.feature_dim)
            self.assertEqual(controller.weights.size, 41)

    def test_screen_report_contains_all_locked_runtime_components(self) -> None:
        means = {
            "setup_runtime": 0.1,
            "native_solve_runtime": 0.2,
            "native_total_runtime": 0.3,
            "controller_runtime": 0.01,
            "setup_bandit_overhead": 0.02,
            "end_to_end_runtime": 0.33,
        }
        summary = {
            "means_sec": means,
            "mean_iterations": 12.0,
            "primary_failure_count": 1,
            "recovered_failure_count": 1,
            "unrecovered_failure_count": 0,
        }
        comparison = {
            "same_setup_rate": 1.0,
            "end_to_end_runtime": {
                "candidate_improvement_pct": 1.0,
                "candidate_improvement_95pct": [0.5, 1.5],
            },
        }
        window = {
            "methods": {
                "bandit_fixed_w1.6": summary,
                RECURSIVE_LSTDQ_V2_METHOD: summary,
            },
            "comparisons": {
                "vs_fixed_w1.6": {RECURSIVE_LSTDQ_V2_METHOD: comparison}
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            _write_solve_screen_report(
                path,
                window_results={"all_4000": window},
                window_actions={},
            )
            report = path.read_text(encoding="utf-8")
        self.assertIn("native total", report)
        self.assertIn("cycles", report)
        self.assertIn("330.000", report)

    def test_recursive_lstdq_factory_uses_locked_confidence_controls(self) -> None:
        args = self._lsvi_args()
        controller, encoder = _make_recursive_lstdq_controller(
            args,
            seed=39967048,
        )
        self.assertEqual(controller.feature_dim, encoder.feature_dim)
        self.assertEqual(controller.spec.lcb_lower_bound_sec, 0.0)
        self.assertEqual(controller.summary()["covariance_residual"], "postfit_unclipped")

    def test_locked_lsvi_controller_matches_selected_screen(self) -> None:
        args = self._lsvi_args()
        _validate_lsvi_protocol(args)
        controller, encoder = _make_lsvi_controller(args, seed=39866039)
        self.assertEqual(controller.feature_dim, encoder.feature_dim)
        self.assertEqual(controller.weights.size, 11)
        for actual, expected in zip(
            controller.weights,
            [1.0 + 0.1 * i for i in range(11)],
        ):
            self.assertAlmostEqual(float(actual), expected)
        self.assertEqual(controller.action_basis.shape[0], 11)
        self.assertEqual(controller.spec.uncertainty_beta, 2.0)

    def test_extended_shared_action_profile_has_41_related_actions(self) -> None:
        args = self._lsvi_args()
        args.shared_action_profile = "1to3_step0p05"
        args.weights = None
        args.action_rbf_centers = None
        _validate_lsvi_protocol(args)
        controller, _encoder = _make_lsvi_controller(args, seed=39966039)
        self.assertEqual(controller.weights.size, 41)
        self.assertEqual(controller.action_basis.shape, (41, 9))
        self.assertAlmostEqual(float(controller.weights[0]), 1.0)
        self.assertAlmostEqual(float(controller.weights[-1]), 3.0)
        self.assertGreater(
            float(np.dot(controller.action_basis[4], controller.action_basis[5])),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
