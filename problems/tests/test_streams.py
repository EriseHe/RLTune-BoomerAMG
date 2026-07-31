from __future__ import annotations

import unittest

import numpy as np

from problems.amg import build_matrix_kwargs_difconv
from problems.scalar_anisotropic_diffusion import (
    build_matrix_kwargs_scalar_anisotropic_diffusion,
)
from problems.streams import (
    generate_difconv_instances,
    generate_scalar_anisotropic_diffusion_instances,
)


class DifConvStreamTests(unittest.TestCase):
    def test_locked_exp44_prefix_is_unchanged(self) -> None:
        instances = generate_difconv_instances(
            count=3,
            seed=39393939,
            grid_choices=[(40, 40, 40)],
            c_min=1.0,
            c_max=1000.0,
        )

        matrix_kwargs, context = instances[0]
        self.assertEqual(matrix_kwargs["rhs_seed"], 3747094179491211093)
        self.assertAlmostEqual(matrix_kwargs["k"], 670.7765768050205)
        self.assertAlmostEqual(matrix_kwargs["c"], 461.52722391983355)
        self.assertAlmostEqual(matrix_kwargs["a0"], 244.4799501330346)
        np.testing.assert_allclose(
            context,
            np.asarray(
                [
                    1.0,
                    0.942192629614899,
                    0.8880657745568643,
                    0.7960810827706223,
                    0.8754464956474619,
                    1.0,
                    1.0,
                    1.0,
                ]
            ),
            rtol=0.0,
            atol=1.0e-15,
        )

    def test_scalar_diffusion_uses_the_same_native_mapping(self) -> None:
        kwargs = dict(
            nx=30,
            ny=31,
            nz=32,
            cx=2.0,
            cy=3.0,
            cz=4.0,
            ax=0.0,
            ay=0.0,
            az=0.0,
            rhs_seed=123,
            rhs_type=1,
        )
        self.assertEqual(
            build_matrix_kwargs_scalar_anisotropic_diffusion(**kwargs),
            build_matrix_kwargs_difconv(**kwargs),
        )

    def test_scalar_stream_uses_canonical_zero_advection_context(self) -> None:
        instances = generate_scalar_anisotropic_diffusion_instances(
            count=2,
            seed=39393939,
            grid_choices=[(40, 40, 40)],
            c_min=1.0,
            c_max=1000.0,
        )

        for matrix_kwargs, context in instances:
            np.testing.assert_array_equal(context[5:], [0.0, 0.0, 0.0])
            np.testing.assert_array_equal(
                [
                    matrix_kwargs["a1"],
                    matrix_kwargs["a2"],
                    matrix_kwargs["a3"],
                ],
                [0.0, 0.0, 0.0],
            )


if __name__ == "__main__":
    unittest.main()
