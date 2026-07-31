from __future__ import annotations

import unittest

import numpy as np

from problems.amg import (
    build_context_difconv,
    build_context_diffusion_advection_from_matrix_kwargs,
    build_matrix_kwargs_difconv,
)
from problems.scalar_anisotropic_diffusion_advection import (
    SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_CONTEXT_DIM,
    build_context_scalar_anisotropic_diffusion_advection,
    build_legacy_linucb_v4_context,
    build_matrix_kwargs_scalar_anisotropic_diffusion_advection,
    stencil_0_scalar_anisotropic_diffusion_advection_rl,
)
from problems.streams import (
    generate_difconv_instances,
    generate_scalar_anisotropic_diffusion_advection_instances,
)


class ScalarAnisotropicDiffusionAdvectionTests(unittest.TestCase):
    def test_native_mapping_reuses_difconv(self) -> None:
        kwargs = {
            "nx": 30,
            "ny": 31,
            "nz": 32,
            "cx": 2.0,
            "cy": 3.0,
            "cz": 4.0,
            "ax": 5.0,
            "ay": 6.0,
            "az": 7.0,
            "rhs_seed": 123,
            "rhs_type": 1,
        }
        self.assertEqual(
            build_matrix_kwargs_scalar_anisotropic_diffusion_advection(
                **kwargs
            ),
            build_matrix_kwargs_difconv(**kwargs),
        )

    def test_context_uses_diffusion_and_advection_without_fixed_grid(self) -> None:
        common = {
            "cx": 2.0,
            "cy": 3.0,
            "cz": 4.0,
            "nx": 60,
            "ny": 60,
            "nz": 60,
            "grid_norm_div": 60.0,
            "c_norm_div": 1000.0,
        }
        diffusion_context = build_context_difconv(**common)
        context = build_context_scalar_anisotropic_diffusion_advection(
            **common,
            ax=5.0,
            ay=6.0,
            az=7.0,
            a_norm_div=1000.0,
        )

        self.assertEqual(
            context.size,
            SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_CONTEXT_DIM,
        )
        np.testing.assert_allclose(context[:5], diffusion_context[:5])
        self.assertEqual(context[5:].shape, (3,))

    def test_context_is_derived_from_the_exact_matrix_kwargs(self) -> None:
        matrix_kwargs = (
            build_matrix_kwargs_scalar_anisotropic_diffusion_advection(
                nx=60,
                ny=60,
                nz=60,
                cx=2.0,
                cy=3.0,
                cz=4.0,
                ax=5.0,
                ay=6.0,
                az=7.0,
                rhs_seed=123,
            )
        )
        expected = build_context_scalar_anisotropic_diffusion_advection(
            cx=2.0,
            cy=3.0,
            cz=4.0,
            ax=5.0,
            ay=6.0,
            az=7.0,
            nx=60,
            ny=60,
            nz=60,
            grid_norm_div=60.0,
            c_norm_div=1000.0,
            a_norm_div=1000.0,
        )

        np.testing.assert_array_equal(
            build_context_diffusion_advection_from_matrix_kwargs(
                matrix_kwargs,
                grid_norm_div=60.0,
                c_norm_div=1000.0,
                a_norm_div=1000.0,
            ),
            expected,
        )

    def test_legacy_v4_projection_restores_fixed_grid_features(self) -> None:
        context = build_context_scalar_anisotropic_diffusion_advection(
            cx=2.0,
            cy=3.0,
            cz=4.0,
            ax=5.0,
            ay=6.0,
            az=7.0,
            nx=60,
            ny=60,
            nz=60,
            grid_norm_div=60.0,
            c_norm_div=1000.0,
            a_norm_div=1000.0,
        )
        legacy = build_legacy_linucb_v4_context(
            stream_context=context,
            nx=60,
            ny=60,
            nz=60,
            grid_norm_div=60.0,
        )

        np.testing.assert_array_equal(legacy[:5], context[:5])
        np.testing.assert_array_equal(legacy[5:], [1.0, 1.0, 1.0])

    def test_stream_is_deterministic_and_varies_both_coefficient_sets(
        self,
    ) -> None:
        kwargs = {
            "count": 8,
            "seed": 20260728,
            "grid_choices": [(12, 12, 12)],
            "c_min": 1.0,
            "c_max": 10.0,
            "advection_min": 2.0,
            "advection_max": 20.0,
        }
        first = generate_scalar_anisotropic_diffusion_advection_instances(
            **kwargs
        )
        second = generate_scalar_anisotropic_diffusion_advection_instances(
            **kwargs
        )
        diffusion_only = generate_difconv_instances(
            count=kwargs["count"],
            seed=kwargs["seed"],
            grid_choices=kwargs["grid_choices"],
            c_min=kwargs["c_min"],
            c_max=kwargs["c_max"],
        )

        self.assertEqual(
            [matrix_kwargs for matrix_kwargs, _context in first],
            [matrix_kwargs for matrix_kwargs, _context in second],
        )
        for (_first_kwargs, first_context), (
            _second_kwargs,
            second_context,
        ) in zip(first, second):
            np.testing.assert_array_equal(first_context, second_context)

        diffusion = np.asarray(
            [
                [matrix_kwargs[key] for key in ("k", "c", "a0")]
                for matrix_kwargs, _context in first
            ],
            dtype=float,
        )
        advection = np.asarray(
            [
                [matrix_kwargs[key] for key in ("a1", "a2", "a3")]
                for matrix_kwargs, _context in first
            ],
            dtype=float,
        )
        self.assertTrue(np.all((1.0 <= diffusion) & (diffusion <= 10.0)))
        self.assertTrue(np.all((2.0 <= advection) & (advection <= 20.0)))
        self.assertGreater(np.unique(diffusion, axis=0).shape[0], 1)
        self.assertGreater(np.unique(advection, axis=0).shape[0], 1)
        for (matrix_kwargs, _context), (
            diffusion_kwargs,
            _diffusion_context,
        ) in zip(first, diffusion_only):
            for key in ("k", "c", "a0", "rhs_seed", "rhs_type"):
                self.assertEqual(matrix_kwargs[key], diffusion_kwargs[key])
        self.assertTrue(
            all(
                context.size
                == SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION_CONTEXT_DIM
                for _matrix_kwargs, context in first
            )
        )

    def test_invalid_advection_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "a_min <= a_max"):
            stencil_0_scalar_anisotropic_diffusion_advection_rl(
                rng=np.random.default_rng(1),
                nx=8,
                ny=8,
                nz=8,
                c_min=1.0,
                c_max=10.0,
                a_min=2.0,
                a_max=1.0,
            )


if __name__ == "__main__":
    unittest.main()
