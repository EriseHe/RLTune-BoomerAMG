"""Encoder-only regressions; no PDE experiments are executed."""
from __future__ import annotations

import _project_paths  # noqa: F401

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from composable_joint_4k import build_composable_solve_runtime
from joint_controller_build import make_setup_obs_encoder
from joint_experiment_config import parse_joint_experiment_config, runtime_config_from_spec
from setup.space import DEFAULT_SETUP_PARAMS, SetupConfigurationSpace
from solve.controllers.common import SolveStateSpec


CONFIGS = Path(__file__).resolve().parents[2] / "joint/solve_control/configs"
PRESET = "paper_final_n60_canonical8d_lstdq_v3_seed_stability_5k"


class SolveStateEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.old = json.loads((CONFIGS / f"{PRESET}.json").read_text())
        cls.new = json.loads((CONFIGS / f"{PRESET}_encoderfix.json").read_text())

    def runtime(self, raw):
        config = runtime_config_from_spec(parse_joint_experiment_config(raw))
        return build_composable_solve_runtime(config, specs=config.methods)

    def features(self, encoder, weight=1.0, **params):
        return encoder.encode(
            mkw={"nx": 60, "ny": 60, "nz": 60, "k": 100, "c": 10, "a0": 1},
            initial_residual=1.0, residual=0.01, previous_residual=0.05,
            cycle=4, last_weight=weight, last_cycle_time=0.005,
            setup_params={**DEFAULT_SETUP_PARAMS, **params},
        )

    def test_paper_rerun_changes_only_encoder_and_artifact_identity(self):
        old, new = copy.deepcopy(self.old), copy.deepcopy(self.new)
        for key in ("name", "description", "output_dir"):
            old.pop(key)
            new.pop(key)
        self.assertEqual(new["solve"].pop("state_encoding"), "space_aware_v2")
        self.assertEqual(old, new)

    def test_new_runtime_observes_every_actual_category_and_weight(self):
        runtime = self.runtime(self.new)
        space = self.new["setup"]["configuration_spaces"]["recommended"]
        for bundle in runtime.controller_bundles.values():
            encoder = bundle.encoder
            self.assertEqual(encoder.feature_dim, 35)
            self.assertEqual(bundle.controller.joint_dim, 315)
            self.assertEqual(encoder.encoding_version, "space_aware_v2")
            self.assertEqual(encoder.weight_bounds, (1.0, 3.0))
            for key, values in (("coarsen_type", space["coarsen_types"]),
                                ("interp_type", space["interp_types"])):
                slot = encoder._setup_offset + encoder.setup_obs_encoder.observed_keys.index(key)
                encoded = [self.features(encoder, **{key: v})[slot] for v in values]
                self.assertEqual(len(set(encoded)), len(values))
                for _ in range(2):
                    with self.assertRaisesRegex(ValueError, "Unknown setup category"):
                        self.features(encoder, **{key: 999})
            encoded_weights = [self.features(encoder, weight=w)[6]
                               for w in bundle.controller.weights]
            np.testing.assert_allclose(encoded_weights, np.linspace(-1.0, 1.0, 41))
            self.assertEqual(len(set(encoded_weights)), 41)

    def test_legacy_encoding_and_other_state_coordinates_are_unchanged(self):
        old = next(iter(self.runtime(self.old).controller_bundles.values()))
        new = next(iter(self.runtime(self.new).controller_bundles.values()))
        self.assertEqual(old.encoder.encoding_version, "legacy_v1")
        np.testing.assert_array_equal(self.features(old.encoder, weight=2.0),
                                      self.features(old.encoder, weight=3.0))
        np.testing.assert_array_equal(self.features(old.encoder, interp_type=17),
                                      self.features(old.encoder, interp_type=6))
        before = self.features(old.encoder, weight=2.7, strong_threshold=0.95)
        after = self.features(new.encoder, weight=2.7, strong_threshold=0.95)
        # Default categorical references stay zero; only last-weight changes.
        np.testing.assert_array_equal(np.delete(before, 6), np.delete(after, 6))
        for attr in ("config", "spec"):
            self.assertEqual(getattr(old.controller, attr), getattr(new.controller, attr))
        np.testing.assert_array_equal(old.controller.action_basis, new.controller.action_basis)

    def test_default_outside_named_subset_is_not_aliased(self):
        setup = make_setup_obs_encoder(
            configuration_space=SetupConfigurationSpace("subset", (3, 6), (8, 17)),
            strict_categories=True,
        )
        self.assertNotEqual(setup._encode_one("coarsen_type", 3),
                            setup._encode_one("coarsen_type", 10))
        self.assertNotEqual(setup._encode_one("interp_type", 8),
                            setup._encode_one("interp_type", 6))
        with self.assertRaisesRegex(ValueError, "encoding_version"):
            SolveStateSpec(tol=1e-6, max_cycles=50, c_max=1000, encoding_version="typo")
        with self.assertRaisesRegex(ValueError, "weight bounds"):
            SolveStateSpec(tol=1e-6, max_cycles=50, c_max=1000,
                           encoding_version="space_aware_v2").build_encoder(setup_obs_encoder=setup)

    def test_checkpoint_roundtrip_rejects_mixed_encoders(self):
        bundle = next(iter(self.runtime(self.new).controller_bundles.values()))
        controller = bundle.controller
        x = self.features(bundle.encoder, weight=2.7, coarsen_type=3, interp_type=17)
        controller.start_episode(initial_environment_weight=1.0)
        controller.update(features=x, action_index=34, cost=0.01,
                          next_features=x, terminal=True)
        controller.finish_episode(learned=True)
        expected_values = controller._values(x, cycle=4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "controller.npz"
            bundle.save(path)
            restored = next(iter(self.runtime(self.new).controller_bundles.values()))
            restored.load(path)
            for actual, expected in zip(restored.controller._values(x, cycle=4), expected_values):
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(restored.controller.theta, controller.theta)
            self.assertEqual(restored.controller.rng.bit_generator.state,
                             controller.rng.bit_generator.state)
            legacy = next(iter(self.runtime(self.old).controller_bundles.values()))
            with self.assertRaisesRegex(ValueError, "does not match"):
                legacy.load(path)
            plain = Path(tmp) / "legacy.npz"
            legacy.save(plain)
            legacy.load(plain)
            with self.assertRaisesRegex(ValueError, "requires checkpoint encoder metadata"):
                restored.load(plain)


if __name__ == "__main__":
    unittest.main()
