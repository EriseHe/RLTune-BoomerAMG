from __future__ import annotations

from solve.tests import _project_paths  # noqa: F401

import unittest

import numpy as np

from solve.controllers.common import SharedActionSpec, SolveStateSpec
from solve.controllers.recursive_lstdq import (
    RecursiveLstdqLcbSpec,
    RecursiveLstdqV2LcbSpec,
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
                self.assertEqual(bundle.controller.feature_dim, bundle.encoder.feature_dim)

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

    def test_v2_spec_cannot_be_silently_routed_to_v1(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
