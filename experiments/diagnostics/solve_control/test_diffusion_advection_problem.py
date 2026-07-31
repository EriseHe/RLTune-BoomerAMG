from __future__ import annotations

import _project_paths  # noqa: F401

import json
from pathlib import Path
import unittest

import numpy as np

import run_joint_experiment as high_level
from joint_method_spec import ComposableMethodSpec
from joint_online_common import (
    _build_paired_instance_stream,
    _problem_stream_spec,
    _sampled_advection_range,
)
from problems.amg import build_normalized_cell_peclet_from_matrix_kwargs
from problems.registry import (
    CANONICAL_NO_C_MEAN_SETUP_CONTEXT,
    CANONICAL_PECLET_ONLY_SETUP_CONTEXT,
    CANONICAL_WITH_A_MEAN_SETUP_CONTEXT,
    CANONICAL_WITH_MEANS_AND_PECLET_SETUP_CONTEXT,
    SCALAR_ANISOTROPIC_DIFFUSION,
    SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
    context_for_setup_method,
    learning_context_for_problem,
    learning_context_for_setup,
)
from setup.learners import SharedLinUCB_AMG_v5
from setup.learners.linucb.SharedLinUCB_AMG_v5 import (
    LINUCB_V5_CONTEXT_FIELDS,
    LINUCB_V5_RECOMMENDED_AGG_INTERP_TYPES,
    LINUCB_V5_RECOMMENDED_COARSEN_TYPES,
    LINUCB_V5_RECOMMENDED_INTERP_TYPES,
)
from solve.controllers.common import SolveStateEncoder
from setup.space import SetupConfigurationSpace
from setup_aware_compare_common import build_online_linucb_branch


class DiffusionAdvectionProblemTests(unittest.TestCase):
    @staticmethod
    def _config() -> dict:
        config_path = (
            Path(__file__).resolve().parents[2]
            / "joint"
            / "solve_control"
            / "configs"
            / "n40_setup_space_compare_joint4k.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["output_dir"] = "/tmp/diffusion-advection-problem-test"
        config["problem"] = {
            "kind": SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
            "grid": [12, 12, 12],
            "c_min": 1.0,
            "c_max": 10.0,
            "advection_min": 2.0,
            "advection_max": 20.0,
        }
        config["stream"] = {
            "cases": 4,
            "warmup_cases": 0,
            "online_cases": 4,
            "instance_offset": 0,
            "seed_groups": [123],
            "shuffle_seeds": [456],
            "cases_per_seed": 4,
            "group_take": 4,
            "smoke": True,
        }
        return config

    def test_high_level_config_resolves_sampled_advection(self) -> None:
        args = high_level.config_to_args(self._config())
        problem, grid, fixed_advection = _problem_stream_spec(args)
        sampled_range = _sampled_advection_range(args, problem=problem)

        self.assertEqual(
            problem,
            SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
        )
        self.assertEqual(grid, (12, 12, 12))
        self.assertEqual(fixed_advection, (0.0, 0.0, 0.0))
        self.assertEqual(sampled_range, (2.0, 20.0))

    def test_stream_manifest_and_context_expose_advection(self) -> None:
        args = high_level.config_to_args(self._config())
        stream, manifest = _build_paired_instance_stream(args)

        self.assertEqual(len(stream), 4)
        self.assertEqual(
            manifest["advection"],
            {
                "mode": "independent_uniform_per_component",
                "range": [2.0, 20.0],
            },
        )
        contexts = np.asarray(
            [context for _matrix_kwargs, context in stream],
            dtype=float,
        )
        self.assertEqual(contexts.shape, (4, 8))
        self.assertGreater(np.unique(contexts[:, 5:], axis=0).shape[0], 1)

        learning_context = learning_context_for_problem(
            SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION
        )
        self.assertEqual(learning_context.dimension, 8)
        self.assertEqual(
            learning_context.interaction_indices,
            (1, 2, 3, 4, 5, 6, 7),
        )
        solve_encoder = SolveStateEncoder(
            tol=1.0e-6,
            max_cycles=50,
            c_max=20.0,
        )
        self.assertEqual(
            tuple(solve_encoder.problem_context_fields),
            tuple(LINUCB_V5_CONTEXT_FIELDS),
        )

    def test_v4_and_v5_receive_distinct_context_contracts(self) -> None:
        args = high_level.config_to_args(self._config())
        stream, _manifest = _build_paired_instance_stream(args)
        matrix_kwargs, stream_context = stream[0]

        v4_context = context_for_setup_method(
            problem_kind=args.problem,
            setup_kind="linucb",
            matrix_kwargs=matrix_kwargs,
            stream_context=stream_context,
            grid_norm_div=float(args.grid_n),
        )
        v5_context = context_for_setup_method(
            problem_kind=args.problem,
            setup_kind="linucb_v5",
            matrix_kwargs=matrix_kwargs,
            stream_context=stream_context,
            grid_norm_div=float(args.grid_n),
        )

        np.testing.assert_array_equal(v4_context[:5], v5_context[:5])
        np.testing.assert_array_equal(v4_context[5:], [1.0, 1.0, 1.0])
        self.assertFalse(np.array_equal(v4_context[5:], v5_context[5:]))
        self.assertEqual(
            learning_context_for_setup(args.problem, "linucb"),
            learning_context_for_setup(
                "scalar_anisotropic_diffusion",
                "linucb",
            ),
        )
        self.assertEqual(
            learning_context_for_setup(args.problem, "linucb_v5"),
            learning_context_for_problem(args.problem),
        )

    def test_no_c_mean_ablation_drops_only_the_redundant_field(self) -> None:
        args = high_level.config_to_args(self._config())
        stream, _manifest = _build_paired_instance_stream(args)
        matrix_kwargs, stream_context = stream[0]

        context = context_for_setup_method(
            problem_kind=args.problem,
            setup_kind="linucb",
            setup_context=CANONICAL_NO_C_MEAN_SETUP_CONTEXT,
            matrix_kwargs=matrix_kwargs,
            stream_context=stream_context,
            grid_norm_div=float(args.grid_n),
        )
        np.testing.assert_array_equal(
            context,
            stream_context[[0, 1, 2, 3, 5, 6, 7]],
        )
        contract = learning_context_for_setup(
            args.problem,
            "linucb",
            CANONICAL_NO_C_MEAN_SETUP_CONTEXT,
        )
        self.assertEqual(contract.dimension, 7)
        self.assertEqual(contract.interaction_indices, (1, 2, 3, 4, 5, 6))

        with self.assertRaisesRegex(ValueError, "frozen LinUCB v5"):
            learning_context_for_setup(
                args.problem,
                "linucb_v5",
                CANONICAL_NO_C_MEAN_SETUP_CONTEXT,
            )

    def test_context_feature_ablation_layouts_and_cell_peclet(self) -> None:
        matrix_kwargs = {
            "nx": 3,
            "ny": 4,
            "nz": 9,
            "k": 2.0,
            "c": 4.0,
            "a0": 5.0,
            "a1": 4.0,
            "a2": -10.0,
            "a3": 0.0,
        }
        canonical = np.asarray(
            [1.0, 0.1, 0.2, 0.3, 0.2, 0.4, -0.5, 0.6],
            dtype=float,
        )
        advection_mean = float(np.mean(canonical[5:8]))
        # The x and y directional ratios both equal 0.5, which maps to 1/3.
        peclet = build_normalized_cell_peclet_from_matrix_kwargs(
            matrix_kwargs
        )
        self.assertAlmostEqual(peclet, 1.0 / 3.0)

        expected_by_mode = {
            CANONICAL_WITH_A_MEAN_SETUP_CONTEXT: np.concatenate(
                (canonical, [advection_mean])
            ),
            CANONICAL_WITH_MEANS_AND_PECLET_SETUP_CONTEXT: np.concatenate(
                (canonical, [advection_mean, peclet])
            ),
            CANONICAL_PECLET_ONLY_SETUP_CONTEXT: np.concatenate(
                (canonical[[0, 1, 2, 3, 5, 6, 7]], [peclet])
            ),
        }
        expected_dimensions = {
            CANONICAL_WITH_A_MEAN_SETUP_CONTEXT: 9,
            CANONICAL_WITH_MEANS_AND_PECLET_SETUP_CONTEXT: 10,
            CANONICAL_PECLET_ONLY_SETUP_CONTEXT: 8,
        }
        for mode, expected in expected_by_mode.items():
            with self.subTest(mode=mode):
                actual = context_for_setup_method(
                    problem_kind=SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
                    setup_kind="linucb",
                    setup_context=mode,
                    matrix_kwargs=matrix_kwargs,
                    stream_context=canonical,
                )
                np.testing.assert_allclose(actual, expected)
                contract = learning_context_for_setup(
                    SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
                    "linucb",
                    mode,
                )
                dimension = expected_dimensions[mode]
                self.assertEqual(contract.dimension, dimension)
                self.assertEqual(
                    contract.interaction_indices,
                    tuple(range(1, dimension)),
                )

    def test_cell_peclet_rejects_nonpositive_diffusion(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            build_normalized_cell_peclet_from_matrix_kwargs(
                {
                    "nx": 4,
                    "ny": 4,
                    "nz": 4,
                    "k": 1.0,
                    "c": 0.0,
                    "a0": 1.0,
                    "a1": 1.0,
                    "a2": 1.0,
                    "a3": 1.0,
                }
            )

    def test_scalar_diffusion_v5_uses_full_eight_dimensional_contract(self) -> None:
        v4_context = learning_context_for_setup(
            SCALAR_ANISOTROPIC_DIFFUSION,
            "linucb",
        )
        v5_context = learning_context_for_setup(
            SCALAR_ANISOTROPIC_DIFFUSION,
            "linucb_v5",
        )

        self.assertEqual(v4_context.dimension, 8)
        self.assertEqual(v4_context.interaction_indices, (1, 2, 3, 4))
        self.assertEqual(v5_context.dimension, 8)
        self.assertEqual(
            v5_context.interaction_indices,
            (1, 2, 3, 4, 5, 6, 7),
        )

    def test_scalar_stream_uses_same_fields_with_zero_advection(self) -> None:
        config = self._config()
        config["problem"] = {
            "kind": SCALAR_ANISOTROPIC_DIFFUSION,
            "grid": [12, 12, 12],
            "c_min": 1.0,
            "c_max": 10.0,
        }
        args = high_level.runtime_config_from_spec(
            high_level.parse_joint_experiment_config(config)
        )
        stream, _manifest = _build_paired_instance_stream(args)
        matrix_kwargs, stream_context = stream[0]

        np.testing.assert_array_equal(
            stream_context[5:],
            [0.0, 0.0, 0.0],
        )
        legacy_context = context_for_setup_method(
            problem_kind=args.problem,
            setup_kind="linucb",
            matrix_kwargs=matrix_kwargs,
            stream_context=stream_context,
            grid_norm_div=float(args.grid_n),
        )
        np.testing.assert_array_equal(
            legacy_context[5:],
            [1.0, 1.0, 1.0],
        )
        self.assertEqual(
            learning_context_for_problem(args.problem),
            learning_context_for_setup(args.problem, "linucb_v5"),
        )

    def test_sampled_problem_rejects_ambiguous_fixed_advection(self) -> None:
        config = self._config()
        config["problem"]["advection"] = [1.0, 0.0, 0.0]
        args = high_level.config_to_args(config)
        with self.assertRaisesRegex(ValueError, "sampled advection"):
            _problem_stream_spec(args)

    @staticmethod
    def _recommended_space(
        *,
        coarsen_types=LINUCB_V5_RECOMMENDED_COARSEN_TYPES,
    ) -> SetupConfigurationSpace:
        return SetupConfigurationSpace(
            name="recommended",
            coarsen_types=coarsen_types,
            interp_types=LINUCB_V5_RECOMMENDED_INTERP_TYPES,
            agg_interp_types=LINUCB_V5_RECOMMENDED_AGG_INTERP_TYPES,
        )

    def test_runner_builds_v5_only_for_the_final_contract(self) -> None:
        branch, _config = build_online_linucb_branch(
            seed=101,
            learner_kind="linucb_v5",
            tune_dim=7,
            tune7_variant="categorical",
            action_space_mode="full_cartesian",
            parameter_resolution=2,
            configuration_space=self._recommended_space(),
            context_dim=8,
            context_interaction_indices=(1, 2, 3, 4, 5, 6, 7),
        )

        self.assertIsInstance(branch.policy.model, SharedLinUCB_AMG_v5)
        self.assertEqual(branch.family, "Shared LinUCB v5")
        self.assertEqual(branch.tune_set, "recommended")

        method = ComposableMethodSpec.from_mapping(
            {
                "id": "paper_linucb_v5",
                "setup": "linucb_v5",
                "setup_space": "recommended",
                "candidate_sampling": "structured512",
                "solve": "default",
            }
        )
        self.assertEqual(
            method.label,
            "Online LinUCB v5 + default solve",
        )

    def test_runner_rejects_a_drifted_v5_recommended_space(self) -> None:
        with self.assertRaisesRegex(ValueError, "coarsen_types"):
            build_online_linucb_branch(
                seed=101,
                learner_kind="linucb_v5",
                tune_dim=7,
                tune7_variant="categorical",
                action_space_mode="full_cartesian",
                parameter_resolution=2,
                configuration_space=self._recommended_space(
                    coarsen_types=(0, 3, 6, 8, 10),
                ),
                context_dim=8,
                context_interaction_indices=(1, 2, 3, 4, 5, 6, 7),
            )


if __name__ == "__main__":
    unittest.main()
