from __future__ import annotations

import unittest

import numpy as np

from problems.amg import (
    DIFFUSION_ADVECTION_PHYSICS_COORDINATE_FIELDS,
    DIFFUSION_ADVECTION_QUADRATIC_CONTEXT_DIM,
    DIFFUSION_ADVECTION_QUADRATIC_CONTEXT_FIELDS,
    build_diffusion_advection_physics_coordinates,
    build_diffusion_advection_physics_context,
    build_diffusion_advection_quadratic_context,
    build_directional_cell_peclet_from_matrix_kwargs,
)
from problems.registry import (
    DIFFUSION_CONVECTION,
    SCALAR_ANISOTROPIC_DIFFUSION,
    SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
    PHYSICS_LINEAR_SETUP_CONTEXT,
    context_for_setup_method,
    learning_context_for_setup,
)


class QuadraticDiffusionAdvectionContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix_kwargs = {
            "nx": 9,
            "ny": 4,
            "nz": 1,
            "k": 2.0,
            "c": 4.0,
            "a0": 8.0,
            "a1": 20.0,
            "a2": -10.0,
            "a3": 0.0,
        }
        self.canonical_context = np.asarray(
            [1.0, 0.2, 0.4, 0.8, 1.4 / 3.0, 0.5, -0.4, 0.0],
            dtype=float,
        )

    def test_six_physics_coordinates_are_nonredundant_and_bounded(self) -> None:
        directional = build_directional_cell_peclet_from_matrix_kwargs(
            self.matrix_kwargs
        )
        np.testing.assert_allclose(directional, [1.0, -0.5, 0.0])

        coordinates = build_diffusion_advection_physics_coordinates(
            canonical_context=self.canonical_context,
            matrix_kwargs=self.matrix_kwargs,
        )
        np.testing.assert_allclose(
            coordinates,
            [
                1.4 / 3.0,
                -0.2 / np.sqrt(2.0),
                -1.0 / np.sqrt(6.0),
                0.5,
                -1.0 / 3.0,
                0.0,
            ],
        )
        self.assertEqual(
            len(DIFFUSION_ADVECTION_PHYSICS_COORDINATE_FIELDS),
            6,
        )

    def test_complete_quadratic_basis_has_stable_28_field_order(self) -> None:
        coordinates = build_diffusion_advection_physics_coordinates(
            canonical_context=self.canonical_context,
            matrix_kwargs=self.matrix_kwargs,
        )
        context = build_diffusion_advection_quadratic_context(
            canonical_context=self.canonical_context,
            matrix_kwargs=self.matrix_kwargs,
        )
        expected_pairs = np.asarray(
            [
                coordinates[left] * coordinates[right]
                for left in range(6)
                for right in range(left + 1, 6)
            ]
        )

        self.assertEqual(DIFFUSION_ADVECTION_QUADRATIC_CONTEXT_DIM, 28)
        self.assertEqual(
            len(DIFFUSION_ADVECTION_QUADRATIC_CONTEXT_FIELDS),
            28,
        )
        self.assertEqual(
            DIFFUSION_ADVECTION_QUADRATIC_CONTEXT_FIELDS[-1],
            "cell_peclet_y*cell_peclet_z",
        )
        np.testing.assert_allclose(context[0], 1.0)
        np.testing.assert_allclose(context[1:7], coordinates)
        np.testing.assert_allclose(context[7:13], coordinates**2)
        np.testing.assert_allclose(context[13:], expected_pairs)

    def test_physics_linear_ablation_is_bias_plus_same_six_coordinates(
        self,
    ) -> None:
        coordinates = build_diffusion_advection_physics_coordinates(
            canonical_context=self.canonical_context,
            matrix_kwargs=self.matrix_kwargs,
        )
        expected = np.concatenate(([1.0], coordinates))
        direct = build_diffusion_advection_physics_context(
            canonical_context=self.canonical_context,
            matrix_kwargs=self.matrix_kwargs,
        )
        via_registry = context_for_setup_method(
            problem_kind=SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
            setup_kind="linucb",
            setup_context=PHYSICS_LINEAR_SETUP_CONTEXT,
            matrix_kwargs=self.matrix_kwargs,
            stream_context=self.canonical_context,
        )
        np.testing.assert_allclose(direct, expected)
        np.testing.assert_allclose(via_registry, expected)

        contract = learning_context_for_setup(
            SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
            "linucb",
            PHYSICS_LINEAR_SETUP_CONTEXT,
        )
        self.assertEqual(contract.dimension, 7)
        self.assertEqual(contract.interaction_indices, tuple(range(1, 7)))

    def test_v6_registry_contract_supports_both_scalar_problem_kinds(self) -> None:
        for problem in (
            SCALAR_ANISOTROPIC_DIFFUSION,
            SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
        ):
            with self.subTest(problem=problem):
                contract = learning_context_for_setup(
                    problem,
                    "linucb_v6",
                )
                self.assertEqual(contract.dimension, 28)
                self.assertEqual(
                    contract.interaction_indices,
                    tuple(range(1, 28)),
                )

        actual = context_for_setup_method(
            problem_kind=SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
            setup_kind="linucb_v6",
            matrix_kwargs=self.matrix_kwargs,
            stream_context=self.canonical_context,
        )
        expected = build_diffusion_advection_quadratic_context(
            canonical_context=self.canonical_context,
            matrix_kwargs=self.matrix_kwargs,
        )
        np.testing.assert_allclose(actual, expected)

        with self.assertRaisesRegex(ValueError, "only defined"):
            learning_context_for_setup(
                DIFFUSION_CONVECTION,
                "linucb_v6",
            )

    def test_scalar_diffusion_uses_same_basis_with_zero_peclet(self) -> None:
        matrix_kwargs = dict(self.matrix_kwargs)
        matrix_kwargs.update(a1=0.0, a2=0.0, a3=0.0)
        canonical = self.canonical_context.copy()
        canonical[5:] = 0.0

        context = context_for_setup_method(
            problem_kind=SCALAR_ANISOTROPIC_DIFFUSION,
            setup_kind="linucb_v6",
            matrix_kwargs=matrix_kwargs,
            stream_context=canonical,
        )
        self.assertEqual(context.shape, (28,))
        np.testing.assert_array_equal(context[4:7], np.zeros(3))


if __name__ == "__main__":
    unittest.main()
