from __future__ import annotations

from solve.tests import _project_paths  # noqa: F401

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from solve.controllers.common import (
    OnlineSolveCase,
    SharedActionSpec,
    SolveStateSpec,
)
from solve.controllers.recursive_lstdq import (
    RecursiveLstdqLcbSpec,
    RecursiveLstdqV2LcbSpec,
    RecursiveLstdqV3LcbSpec,
)
from solve.registry import (
    COMPOSABLE_SOLVE_KINDS,
    ONLINE_SOLVE_KINDS,
    OnlineControllerBuildSpec,
    build_online_solve_controller,
    make_online_controller_spec,
    solve_kind_registration,
)


class _SetupObservationEncoder:
    observed_keys = ("strong_threshold", "interp_type")
    defaults = {"strong_threshold": 0.25, "interp_type": 6}

    @staticmethod
    def encode(_params: object) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)


class SolveRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = SolveStateSpec(
            tol=1.0e-6,
            max_cycles=50,
            c_max=1000.0,
        )
        self.actions = SharedActionSpec(
            weights=(1.0, 1.5, 2.0),
            rbf_centers=(1.0, 1.5, 2.0),
            rbf_sigma=0.2,
        )
        self.setup_encoder = _SetupObservationEncoder()

    def test_registry_covers_every_composable_solve_kind(self) -> None:
        self.assertEqual(
            COMPOSABLE_SOLVE_KINDS,
            (
                "default",
                "fixed",
                "ppo",
                "recursive_mc",
                "recursive_lstdq_v1",
                "recursive_lstdq_v2",
                "recursive_lstdq_v3",
                "rblspi",
                "stagewise_lsvi",
                "structured_model_based",
                "recalibrated_lsvi",
            ),
        )
        self.assertEqual(
            ONLINE_SOLVE_KINDS,
            COMPOSABLE_SOLVE_KINDS[3:],
        )

    def test_every_online_kind_builds_its_registered_controller(self) -> None:
        for index, kind in enumerate(ONLINE_SOLVE_KINDS):
            with self.subTest(kind=kind):
                trace_lambda = (
                    0.8
                    if solve_kind_registration(kind).trace_lambda_from_request
                    else None
                )
                request = make_online_controller_spec(
                    kind=kind,
                    state=self.state,
                    actions=self.actions,
                    algorithm_parameters={},
                    trace_lambda=trace_lambda,
                )
                bundle = build_online_solve_controller(
                    request,
                    setup_obs_encoder=self.setup_encoder,
                    seed=100 + index,
                )
                registration = solve_kind_registration(kind)
                self.assertIs(type(bundle.controller), registration.controller_type)
                self.assertEqual(
                    bundle.controller.feature_dim,
                    bundle.encoder.feature_dim,
                )
                metadata = bundle.protocol_metadata()
                self.assertEqual(metadata["kind"], kind)
                self.assertEqual(metadata["family"], registration.family)
                self.assertEqual(
                    metadata["state_encoder"]["problem_context_mode"],
                    "canonical",
                )
                self.assertEqual(
                    metadata["state_encoder"]["feature_dim"],
                    bundle.encoder.feature_dim,
                )
                json.dumps(metadata)

    def test_online_registrations_delegate_to_family_factories(self) -> None:
        for kind in ONLINE_SOLVE_KINDS:
            with self.subTest(kind=kind):
                registration = solve_kind_registration(kind)
                self.assertIsNotNone(registration.factory)
                self.assertTrue(
                    registration.factory.__module__.endswith(".factory"),
                    registration.factory.__module__,
                )
                self.assertIn(
                    f".{registration.family}.",
                    registration.factory.__module__,
                )

    def test_rblspi_disables_epsilon_without_a_second_config_path(self) -> None:
        request = make_online_controller_spec(
            kind="rblspi",
            state=self.state,
            actions=self.actions,
            algorithm_parameters={},
        )
        controller = build_online_solve_controller(
            request,
            setup_obs_encoder=self.setup_encoder,
            seed=7,
        ).controller
        self.assertEqual(controller.config.epsilon_start, 0.0)
        self.assertEqual(controller.config.epsilon_final, 0.0)
        self.assertEqual(controller.config.trace_lambda, 0.0)

    def test_versioned_lstdq_specs_cannot_be_silently_cross_routed(self) -> None:
        with self.assertRaisesRegex(TypeError, "requires RecursiveLstdqLcbSpec"):
            build_online_solve_controller(
                OnlineControllerBuildSpec(
                    kind="recursive_lstdq_v1",
                    state=self.state,
                    actions=self.actions,
                    algorithm=RecursiveLstdqV2LcbSpec(),
                    trace_lambda=0.8,
                ),
                setup_obs_encoder=self.setup_encoder,
                seed=7,
            )
        with self.assertRaisesRegex(TypeError, "requires RecursiveLstdqV2LcbSpec"):
            build_online_solve_controller(
                OnlineControllerBuildSpec(
                    kind="recursive_lstdq_v2",
                    state=self.state,
                    actions=self.actions,
                    algorithm=RecursiveLstdqLcbSpec(),
                    trace_lambda=0.8,
                ),
                setup_obs_encoder=self.setup_encoder,
                seed=7,
            )
        with self.assertRaisesRegex(TypeError, "requires RecursiveLstdqV3LcbSpec"):
            build_online_solve_controller(
                OnlineControllerBuildSpec(
                    kind="recursive_lstdq_v3",
                    state=self.state,
                    actions=self.actions,
                    algorithm=RecursiveLstdqV2LcbSpec(),
                    trace_lambda=0.8,
                ),
                setup_obs_encoder=self.setup_encoder,
                seed=7,
            )
        with self.assertRaisesRegex(TypeError, "requires RecursiveLstdqV2LcbSpec"):
            build_online_solve_controller(
                OnlineControllerBuildSpec(
                    kind="recursive_lstdq_v2",
                    state=self.state,
                    actions=self.actions,
                    algorithm=RecursiveLstdqV3LcbSpec(),
                    trace_lambda=0.8,
                ),
                setup_obs_encoder=self.setup_encoder,
                seed=7,
            )

    def test_bundle_owns_summary_save_and_protocol_metadata(self) -> None:
        request = make_online_controller_spec(
            kind="recursive_lstdq_v2",
            state=self.state,
            actions=self.actions,
            algorithm_parameters={
                "ridge": 2.0,
                "coverage_ridge": 3.0,
                "residual_scale_window": 64,
                "residual_scale_min_samples": 8,
            },
            trace_lambda=0.7,
        )
        bundle = build_online_solve_controller(
            request,
            setup_obs_encoder=self.setup_encoder,
            seed=17,
        )

        native_summary = bundle.controller.summary()
        summary = bundle.summary()
        self.assertEqual(summary["steps"], 0)
        self.assertEqual(summary["episodes"], 0)
        self.assertEqual(summary["epsilon"], bundle.controller.epsilon)
        for key, value in native_summary.items():
            self.assertEqual(summary[key], value)

        metadata = bundle.protocol_metadata()
        self.assertEqual(metadata["kind"], "recursive_lstdq_v2")
        self.assertEqual(metadata["family"], "recursive_lstdq")
        self.assertEqual(metadata["state"], "setup_full")
        self.assertEqual(metadata["actions"], [1.0, 1.5, 2.0])
        self.assertEqual(metadata["trace_lambda"], 0.7)
        self.assertEqual(metadata["algorithm"]["ridge"], 2.0)
        self.assertEqual(metadata["coverage_ridge"], 3.0)
        json.dumps(metadata)
        metadata["algorithm"]["ridge"] = -1.0
        self.assertEqual(
            bundle.protocol_metadata()["algorithm"]["ridge"],
            2.0,
        )

        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = Path(tmp) / "bundle.npz"
            native_path = Path(tmp) / "native.npz"
            bundle.save(bundle_path)
            bundle.controller.save(native_path)
            with np.load(bundle_path, allow_pickle=False) as bundled, np.load(
                native_path,
                allow_pickle=False,
            ) as native:
                self.assertEqual(set(bundled.files), set(native.files))
                for key in bundled.files:
                    np.testing.assert_array_equal(bundled[key], native[key])

    def test_v3_protocol_declares_episode_cluster_parameter_uncertainty(
        self,
    ) -> None:
        bundle = build_online_solve_controller(
            make_online_controller_spec(
                kind="recursive_lstdq_v3",
                state=self.state,
                actions=self.actions,
                algorithm_parameters={"uncertainty_beta": 3.0},
                trace_lambda=0.8,
            ),
            setup_obs_encoder=self.setup_encoder,
            seed=19,
        )
        metadata = bundle.protocol_metadata()
        self.assertEqual(
            metadata["uncertainty"],
            "episode-cluster post-fit sandwich covariance",
        )
        self.assertEqual(
            metadata["uncertainty_semantics"],
            "parameter uncertainty",
        )
        self.assertEqual(
            metadata["cluster_unit"],
            "complete committed AMG solve episode",
        )

    def test_bundle_run_case_is_a_thin_typed_td_adapter(self) -> None:
        bundle = build_online_solve_controller(
            make_online_controller_spec(
                kind="recursive_mc",
                state=self.state,
                actions=self.actions,
                algorithm_parameters={},
            ),
            setup_obs_encoder=self.setup_encoder,
            seed=5,
        )
        fallback = lambda: {"runtime": 1.0}  # noqa: E731
        case = OnlineSolveCase(
            mkw={"nx": 4},
            params={"strong_threshold": 0.5},
            solve_tol=1.0e-6,
            solve_max_cycles=50,
            learn=True,
            explore=True,
            problem_context=(
                1.0,
                0.1,
                0.2,
                0.3,
                0.2,
                0.4,
                0.5,
                0.6,
            ),
            epsilon=0.2,
            defer_monte_carlo_update=True,
            record_action_metadata=True,
            initial_environment_weight_override=1.25,
            fallback_attempt=fallback,
            failure_penalty_sec=0.5,
        )
        expected = {"runtime": 0.25, "failed": False}
        with patch(
            "solve.core.episode.run_td_episode",
            return_value=expected,
        ) as run_episode:
            self.assertIs(bundle.run_case(case), expected)
        run_episode.assert_called_once_with(
            mkw={"nx": 4},
            params={"strong_threshold": 0.5},
            controller=bundle.controller,
            encoder=bundle.encoder,
            problem_context=case.problem_context,
            solve_tol=1.0e-6,
            solve_max_cycles=50,
            learn=True,
            explore=True,
            epsilon=0.2,
            defer_monte_carlo_update=True,
            record_action_metadata=True,
            initial_environment_weight_override=1.25,
            fallback_attempt=fallback,
            failure_penalty_sec=0.5,
        )

    def test_episode_legacy_imports_share_the_core_implementation(self) -> None:
        from solve.core.episode import run_td_episode
        from solve.controllers.sarsa import run_td_episode as family_episode
        from solve.controllers.sarsa.online_td_lambda import (
            run_td_episode as legacy_episode,
        )

        self.assertIs(family_episode, run_td_episode)
        self.assertIs(legacy_episode, run_td_episode)

    def test_td_config_legacy_imports_share_the_common_class(self) -> None:
        from solve.controllers.common.td_config import ExpectedSarsaLambdaConfig
        from solve.controllers.sarsa import (
            ExpectedSarsaLambdaConfig as FamilyConfig,
        )
        from solve.controllers.sarsa.config import (
            ExpectedSarsaLambdaConfig as LegacyConfig,
        )

        self.assertIs(FamilyConfig, ExpectedSarsaLambdaConfig)
        self.assertIs(LegacyConfig, ExpectedSarsaLambdaConfig)

    def test_specs_remain_available_from_legacy_implementation_modules(self) -> None:
        from solve.controllers.lsvi.config import (
            HierarchicalLsviLcbSpec as HierarchicalConfig,
        )
        from solve.controllers.lsvi.config import (
            StagewiseLsviLcbSpec as StagewiseConfig,
        )
        from solve.controllers.lsvi.hierarchical import (
            HierarchicalLsviLcbSpec as LegacyHierarchical,
        )
        from solve.controllers.lsvi.stagewise import (
            StagewiseLsviLcbSpec as LegacyStagewise,
        )
        from solve.controllers.recursive_lstdq.config import (
            RecursiveLstdqLcbSpec as LstdqConfig,
        )
        from solve.controllers.recursive_lstdq.config import (
            RecursiveLstdqV2LcbSpec as LstdqV2Config,
        )
        from solve.controllers.recursive_lstdq.config import (
            RecursiveLstdqV3LcbSpec as LstdqV3Config,
        )
        from solve.controllers.recursive_lstdq.v1 import (
            RecursiveLstdqLcbSpec as LegacyLstdq,
        )
        from solve.controllers.recursive_lstdq.v2 import (
            RecursiveLstdqV2LcbSpec as LegacyLstdqV2,
        )
        from solve.controllers.recursive_lstdq.v3 import (
            RecursiveLstdqV3LcbSpec as LegacyLstdqV3,
        )

        self.assertIs(LegacyLstdq, LstdqConfig)
        self.assertIs(LegacyLstdqV2, LstdqV2Config)
        self.assertIs(LegacyLstdqV3, LstdqV3Config)
        self.assertIs(LegacyStagewise, StagewiseConfig)
        self.assertIs(LegacyHierarchical, HierarchicalConfig)


if __name__ == "__main__":
    unittest.main()
