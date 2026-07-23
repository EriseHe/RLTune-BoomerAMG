from __future__ import annotations

from setup.tests import _project_paths  # noqa: F401

import tempfile
import unittest
from pathlib import Path

import numpy as np

from setup.learners import SharedLinTS_AMG_v2
from setup.learners.common import (
    AOTCandidateSchedule,
    CompactActionCatalog,
    FactorizedActionFeatureCache,
    GenericActionFeatureEncoder,
    ParameterSpaceSpec,
    ParameterSpec,
)
from setup.utils.setup_amg import build_actions_from_spec


class SharedLinTSV2Tests(unittest.TestCase):
    @staticmethod
    def _numeric_model(*, posterior_seed: int = 19) -> SharedLinTS_AMG_v2:
        parameter_spec = ParameterSpaceSpec(
            (
                ParameterSpec(
                    name="weight",
                    kind="continuous",
                    values=(1.0, 1.5, 2.0),
                    default=1.5,
                    center=1.5,
                    scale=0.5,
                ),
            )
        )
        return SharedLinTS_AMG_v2(
            build_actions_from_spec(parameter_spec),
            context_dim=2,
            parameter_spec=parameter_spec,
            context_interaction_indices=(0, 1),
            candidate_pool_size=3,
            relative_sampling_scale=0.2,
            loss_scale_prior=0.1,
            seed=7,
            posterior_seed=posterior_seed,
        )

    def test_rank_one_factor_matches_precision_and_posterior_covariance(self) -> None:
        model = self._numeric_model()
        contexts = (
            np.asarray([1.0, 0.25]),
            np.asarray([1.0, 0.75]),
            np.asarray([0.5, 1.0]),
        )
        for index, context in enumerate(contexts):
            model.predict(context)
            model.update(0.08 + 0.01 * index)

        precision = np.linalg.inv(model.A_inv)
        np.testing.assert_allclose(
            model._precision_cholesky @ model._precision_cholesky.T,
            precision,
            rtol=2.0e-12,
            atol=2.0e-12,
        )

        model.reference_loss = 0.1
        scale = model._effective_sampling_scale()
        draws = np.asarray([model._sample_parameter()[0] for _ in range(20_000)])
        empirical = np.cov(draws, rowvar=False, ddof=1)
        expected = scale * scale * model.A_inv
        np.testing.assert_allclose(empirical, expected, rtol=0.08, atol=2.0e-5)

    def test_posterior_rng_does_not_advance_candidate_rng(self) -> None:
        model = self._numeric_model()
        candidate_state = model.rng.bit_generator.state
        model._sample_parameter()
        self.assertEqual(model.rng.bit_generator.state, candidate_state)

    def test_checkpoint_restores_factor_scale_and_next_posterior_draw(self) -> None:
        source = self._numeric_model()
        source.predict(np.asarray([1.0, 0.5]))
        source.update(0.125)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lints_v2.npz"
            source.save_mutable_state(path, metadata={"stream_hash": "paired"})
            restored = self._numeric_model(posterior_seed=99)
            metadata = restored.load_mutable_state(path)

        self.assertEqual(metadata["stream_hash"], "paired")
        self.assertEqual(restored.reference_loss, source.reference_loss)
        np.testing.assert_allclose(
            restored._precision_cholesky, source._precision_cholesky
        )
        np.testing.assert_allclose(
            restored._sample_parameter()[0], source._sample_parameter()[0]
        )

    def test_compact_categorical_features_and_per_case_aot_pairing(self) -> None:
        parameter_spec = ParameterSpaceSpec(
            (
                ParameterSpec(
                    name="weight",
                    kind="continuous",
                    values=(1.0, 1.5, 2.0),
                    default=1.5,
                    center=1.5,
                    scale=0.5,
                ),
                ParameterSpec(
                    name="kind",
                    kind="categorical",
                    values=(0, 1, 2),
                    default=1,
                ),
            )
        )
        catalog = CompactActionCatalog(parameter_spec)
        encoder = GenericActionFeatureEncoder(parameter_spec)
        cache = FactorizedActionFeatureCache(catalog, encoder)
        arm_zero = catalog.index_of({"weight": 1.5, "kind": 0})
        arm_two = catalog.index_of({"weight": 1.5, "kind": 2})
        self.assertFalse(
            np.allclose(cache.feature(arm_zero), cache.feature(arm_two))
        )

        with tempfile.TemporaryDirectory() as directory:
            schedule_a = AOTCandidateSchedule(
                directory=Path(directory),
                catalog=catalog,
                rounds=9,
                pool_size=4,
                seed=23,
            )
            schedule_b = AOTCandidateSchedule(
                directory=Path(directory),
                catalog=catalog,
                rounds=9,
                pool_size=4,
                seed=23,
            )
            first_a = schedule_a.next_arms()
            first_b = schedule_b.next_arms()
            np.testing.assert_array_equal(first_a, first_b)

            schedule_a.next_arms()  # learner A needed one recovery attempt
            schedule_a.finish_case(max_selections=3)
            schedule_b.finish_case(max_selections=3)
            np.testing.assert_array_equal(
                schedule_a.next_arms(), schedule_b.next_arms()
            )


if __name__ == "__main__":
    unittest.main()
