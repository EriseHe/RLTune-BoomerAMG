from __future__ import annotations

from solve.tests import _project_paths  # noqa: F401

import unittest

from solve.controllers.lsvi import LsviFamilySpecs
from solve.controllers.model_based import StructuredModelBasedSpec
from solve.controllers.rblspi import RecursiveBlstdqSpec
from solve.controllers.recursive_lstdq import RecursiveLstdqFamilySpecs
from solve.controllers.recursive_mc import RecursiveMonteCarloLcbSpec


class FamilyConfigDecodingTests(unittest.TestCase):
    def test_recursive_lstdq_owns_shared_and_v2_aliases(self) -> None:
        shared = {
            "ridge": 2.0,
            "beta": 3.0,
            "trace_lambda": 0.7,
            "residual_floor_sec": 0.004,
            "lcb_lower_bound_sec": 0.02,
        }
        v2 = {
            "beta": 4.0,
            "coverage_ridge": 5.0,
            "residual_window": 64,
            "min_samples": 8,
        }

        decoded = RecursiveLstdqFamilySpecs.from_mappings(shared, v2)

        self.assertEqual(decoded.trace_lambda, 0.7)
        self.assertEqual(decoded.v1.ridge, 2.0)
        self.assertEqual(decoded.v1.uncertainty_beta, 3.0)
        self.assertEqual(decoded.v1.residual_floor_sec, 0.004)
        self.assertEqual(decoded.v1.lcb_lower_bound_sec, 0.02)
        self.assertEqual(decoded.v2.ridge, 2.0)
        self.assertEqual(decoded.v2.residual_floor_sec, 0.004)
        self.assertEqual(decoded.v2.lcb_lower_bound_sec, 0.02)
        self.assertEqual(decoded.v2.uncertainty_beta, 4.0)
        self.assertEqual(decoded.v2.coverage_ridge, 5.0)
        self.assertEqual(decoded.v2.residual_scale_window, 64)
        self.assertEqual(decoded.v2.residual_scale_min_samples, 8)

    def test_lsvi_owns_shared_and_hierarchical_aliases(self) -> None:
        decoded = LsviFamilySpecs.from_mappings(
            {
                "ridge": 2.0,
                "beta": 3.0,
                "residual_floor_sec": 0.004,
                "refit_interval_episodes": 11,
            },
            {
                "beta": 4.0,
                "refit_sweeps": 5,
                "shrinkage_samples": 6.0,
            },
            horizon=37,
        )

        self.assertEqual(decoded.stagewise.horizon, 37)
        self.assertEqual(decoded.stagewise.ridge, 2.0)
        self.assertEqual(decoded.stagewise.uncertainty_beta, 3.0)
        self.assertEqual(decoded.stagewise.residual_floor_sec, 0.004)
        self.assertEqual(decoded.stagewise.refit_interval_episodes, 11)
        self.assertEqual(decoded.hierarchical.horizon, 37)
        self.assertEqual(decoded.hierarchical.ridge, 2.0)
        self.assertEqual(decoded.hierarchical.uncertainty_beta, 4.0)
        self.assertEqual(decoded.hierarchical.residual_floor_sec, 0.004)
        self.assertEqual(decoded.hierarchical.refit_interval_episodes, 11)
        self.assertEqual(decoded.hierarchical.refit_sweeps, 5)
        self.assertEqual(
            decoded.hierarchical.residual_shrinkage_samples,
            6.0,
        )

    def test_simple_families_own_aliases_and_defaults(self) -> None:
        recursive_mc = RecursiveMonteCarloLcbSpec.from_mapping(
            {
                "ridge": 2.0,
                "beta": 3.0,
                "residual_floor_sec": 0.004,
                "episode_half_life": 25.0,
            }
        )
        self.assertEqual(recursive_mc.ridge, 2.0)
        self.assertEqual(recursive_mc.uncertainty_beta, 3.0)
        self.assertEqual(recursive_mc.residual_floor_sec, 0.004)
        self.assertEqual(recursive_mc.episode_half_life, 25.0)

        rblspi = RecursiveBlstdqSpec.from_mapping(
            {
                "prior_precision": 2.0,
                "noise_precision": 3.0,
                "gram_ridge": 4.0,
            }
        )
        self.assertEqual(rblspi.prior_precision, 2.0)
        self.assertEqual(rblspi.noise_precision, 3.0)
        self.assertEqual(rblspi.gram_ridge, 4.0)

        model = StructuredModelBasedSpec.from_mapping(
            {
                "ridge": 2.0,
                "min_samples": 3,
                "scale_window": 4,
            }
        )
        self.assertEqual(model.ridge, 2.0)
        self.assertEqual(model.minimum_samples, 3)
        self.assertEqual(model.scale_window, 4)

        self.assertEqual(
            RecursiveMonteCarloLcbSpec.from_mapping({"beta": None}),
            RecursiveMonteCarloLcbSpec(),
        )
        self.assertEqual(
            RecursiveBlstdqSpec.from_mapping({"gram_ridge": None}),
            RecursiveBlstdqSpec(),
        )
        self.assertEqual(
            StructuredModelBasedSpec.from_mapping({"min_samples": None}),
            StructuredModelBasedSpec(),
        )

    def test_family_decoders_reject_unknown_fields_at_the_owner(self) -> None:
        cases = (
            (
                lambda: RecursiveMonteCarloLcbSpec.from_mapping(
                    {"not_mc": 1}
                ),
                "Unknown solve.recursive_mc keys",
            ),
            (
                lambda: RecursiveLstdqFamilySpecs.from_mappings(
                    {"not_lstdq": 1},
                    {},
                ),
                "Unknown solve.lstdq keys",
            ),
            (
                lambda: RecursiveLstdqFamilySpecs.from_mappings(
                    {},
                    {"ridge": 1},
                ),
                "Unknown solve.lstdq_v2 keys",
            ),
            (
                lambda: RecursiveBlstdqSpec.from_mapping(
                    {"not_rblspi": 1}
                ),
                "Unknown solve.rblspi keys",
            ),
            (
                lambda: LsviFamilySpecs.from_mappings(
                    {"not_lsvi": 1},
                    {},
                    horizon=50,
                ),
                "Unknown solve.lsvi keys",
            ),
            (
                lambda: LsviFamilySpecs.from_mappings(
                    {},
                    {"ridge": 1},
                    horizon=50,
                ),
                "Unknown solve.recalibrated_lsvi keys",
            ),
            (
                lambda: StructuredModelBasedSpec.from_mapping(
                    {"not_model": 1}
                ),
                "Unknown solve.structured_model keys",
            ),
        )
        for decode, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    decode()

    def test_joint_specific_family_defaults_remain_stable(self) -> None:
        lstdq = RecursiveLstdqFamilySpecs.from_mappings({}, {})
        self.assertEqual(lstdq.v1.uncertainty_beta, 2.0)
        self.assertEqual(lstdq.v2.uncertainty_beta, 2.0)
        self.assertEqual(lstdq.v2.residual_scale_window, 2048)
        self.assertEqual(lstdq.trace_lambda, 0.8)

        lsvi = LsviFamilySpecs.from_mappings({}, {}, horizon=50)
        self.assertEqual(lsvi.stagewise.refit_interval_episodes, 100)
        self.assertEqual(lsvi.hierarchical.refit_interval_episodes, 100)


if __name__ == "__main__":
    unittest.main()
