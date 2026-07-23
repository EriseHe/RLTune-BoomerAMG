from __future__ import annotations

from setup.tests import _project_paths  # noqa: F401

import unittest
from importlib import import_module
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from setup.learners.common import (
    LinTSV2Spec,
    LinUCBV4Spec,
    SharedSetupLearnerSpec,
)
from setup.registry import (
    COMPOSABLE_SETUP_KINDS,
    ONLINE_SETUP_KINDS,
    SetupLearnerBuildSpec,
    build_online_setup_learner,
    make_setup_learner_spec,
    setup_kind_registration,
)
from setup.learners import SharedLinTS_AMG_v2, SharedLinUCB_AMG_v4
from setup.learners.common import ParameterSpaceSpec, ParameterSpec
from setup.utils.setup_amg import build_actions_from_spec


class SetupRegistryTests(unittest.TestCase):
    @staticmethod
    def _shared(*, seed: int = 7) -> SharedSetupLearnerSpec:
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
        return SharedSetupLearnerSpec(
            actions=build_actions_from_spec(parameter_spec),
            context_dim=2,
            parameter_spec=parameter_spec,
            context_interaction_indices=(0, 1),
            candidate_pool_size=3,
            always_include_arms=(1,),
            elite_cache_size=1,
            initial_guess=(1.5,),
            initial_guess_rounds=1,
            seed=seed,
        )

    def test_registry_covers_composable_and_online_setup_kinds(self) -> None:
        self.assertEqual(
            COMPOSABLE_SETUP_KINDS,
            ("default", "linucb", "lints"),
        )
        self.assertEqual(ONLINE_SETUP_KINDS, ("linucb", "lints"))
        self.assertFalse(setup_kind_registration("random").online)
        self.assertNotIn("random", COMPOSABLE_SETUP_KINDS)
        self.assertIs(
            setup_kind_registration("linucbv4").learner_type,
            SharedLinUCB_AMG_v4,
        )
        self.assertIs(
            setup_kind_registration("lints_v2").learner_type,
            SharedLinTS_AMG_v2,
        )

    def test_legacy_imports_alias_canonical_classes_and_modules(self) -> None:
        compatibility_root = Path(__file__).resolve().parents[2] / "SetupPhase"
        inserted = str(compatibility_root) not in sys.path
        if inserted:
            sys.path.insert(0, str(compatibility_root))
        try:
            qualified = import_module("SetupPhase.learners")
            bare = import_module("learners")
            canonical_module = import_module(
                "setup.learners.linucb.SharedLinUCB_AMG_v4"
            )
            qualified_module = import_module(
                "SetupPhase.learners.linucb.SharedLinUCB_AMG_v4"
            )
            bare_module = import_module(
                "learners.linucb.SharedLinUCB_AMG_v4"
            )

            self.assertIs(
                SharedLinUCB_AMG_v4,
                qualified.SharedLinUCB_AMG_v4,
            )
            self.assertIs(SharedLinUCB_AMG_v4, bare.SharedLinUCB_AMG_v4)
            self.assertIs(canonical_module, qualified_module)
            self.assertIs(canonical_module, bare_module)
        finally:
            if inserted:
                sys.path.remove(str(compatibility_root))

    def test_import_order_preserves_identity_in_fresh_processes(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        compatibility_root = repo_root / "SetupPhase"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            (
                str(compatibility_root),
                str(repo_root),
                environment.get("PYTHONPATH", ""),
            )
        )
        program = """
import importlib
import os
from pathlib import Path
import sys

first_name = {
    "canonical": "setup.learners",
    "qualified": "SetupPhase.learners",
    "bare": "learners",
}[os.environ["SETUP_IMPORT_ORDER"]]
first = importlib.import_module(first_name).SharedLinUCB_AMG_v4
canonical = importlib.import_module("setup.learners")
qualified = importlib.import_module("SetupPhase.learners")
bare = importlib.import_module("learners")
setup_phase = importlib.import_module("SetupPhase")
canonical_leaf = importlib.import_module(
    "setup.learners.linucb.SharedLinUCB_AMG_v4"
)
qualified_leaf = importlib.import_module(
    "SetupPhase.learners.linucb.SharedLinUCB_AMG_v4"
)
bare_leaf = importlib.import_module(
    "learners.linucb.SharedLinUCB_AMG_v4"
)
assert first is canonical.SharedLinUCB_AMG_v4
assert first is qualified.SharedLinUCB_AMG_v4
assert first is bare.SharedLinUCB_AMG_v4
assert setup_phase.learners is canonical
assert canonical_leaf is qualified_leaf is bare_leaf
assert importlib.import_module("setup.registry") is importlib.import_module(
    "SetupPhase.registry"
)
assert importlib.import_module("setup.utils.setup_amg") is importlib.import_module(
    "SetupPhase.utils.setup_amg"
)
assert setup_phase.utils is importlib.import_module("setup.utils")
assert importlib.import_module("setup.utils.setup_amg") is importlib.import_module(
    "utils.setup_amg"
)
assert str(Path.cwd() / "setup") not in sys.path
"""
        for order in ("canonical", "qualified", "bare"):
            with self.subTest(order=order):
                environment["SETUP_IMPORT_ORDER"] = order
                subprocess.run(
                    (sys.executable, "-c", program),
                    check=True,
                    cwd=repo_root,
                    env=environment,
                    capture_output=True,
                    text=True,
                )

    def test_canonical_import_installs_pickle_module_aliases(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(repo_root), environment.get("PYTHONPATH", ""))
        )
        program = """
import importlib
import pickle
from pathlib import Path
import sys

import setup

canonical = importlib.import_module(
    "setup.learners.linucb.SharedLinUCB_AMG_v4"
)
legacy = importlib.import_module("learners.SharedLinUCB_AMG_v4")
assert canonical is legacy
payload = b"clearners.SharedLinUCB_AMG_v4\\nSharedLinUCB_AMG_v4\\n."
assert pickle.loads(payload) is canonical.SharedLinUCB_AMG_v4
assert str(Path.cwd() / "setup") not in sys.path
assert str(Path.cwd() / "SetupPhase") not in sys.path
"""
        subprocess.run(
            (sys.executable, "-c", program),
            check=True,
            cwd=repo_root,
            env=environment,
            capture_output=True,
            text=True,
        )

    def test_linucb_factory_matches_direct_constructor(self) -> None:
        shared = self._shared()
        built = build_online_setup_learner(
            make_setup_learner_spec(kind="linucb", shared=shared)
        )
        direct = SharedLinUCB_AMG_v4(
            shared.actions,
            context_dim=shared.context_dim,
            **shared.learner_kwargs(),
        )

        context = np.asarray((1.0, 0.25), dtype=float)
        self.assertIsInstance(built, SharedLinUCB_AMG_v4)
        self.assertEqual(built.predict(context), direct.predict(context))
        np.testing.assert_allclose(built.A_inv, direct.A_inv)
        np.testing.assert_allclose(built.b, direct.b)

    def test_lints_factory_matches_direct_constructor(self) -> None:
        shared = self._shared()
        algorithm = {
            "relative_sampling_scale": 0.2,
            "loss_scale_prior": 0.125,
            "posterior_seed": 19,
        }
        built = build_online_setup_learner(
            make_setup_learner_spec(
                kind="lints_v2",
                shared=shared,
                algorithm_parameters=algorithm,
            )
        )
        direct = SharedLinTS_AMG_v2(
            shared.actions,
            context_dim=shared.context_dim,
            **shared.learner_kwargs(),
            **LinTSV2Spec(**algorithm).learner_kwargs(),
        )

        context = np.asarray((1.0, 0.25), dtype=float)
        self.assertIsInstance(built, SharedLinTS_AMG_v2)
        self.assertEqual(built.predict(context), direct.predict(context))
        np.testing.assert_allclose(
            built._precision_cholesky,
            direct._precision_cholesky,
        )
        self.assertEqual(
            built.posterior_rng.bit_generator.state,
            direct.posterior_rng.bit_generator.state,
        )

    def test_factory_rejects_algorithm_spec_from_another_kind(self) -> None:
        with self.assertRaisesRegex(TypeError, "requires LinUCBV4Spec"):
            build_online_setup_learner(
                SetupLearnerBuildSpec(
                    kind="linucb",
                    shared=self._shared(),
                    algorithm=LinTSV2Spec(),
                )
            )

        with self.assertRaisesRegex(TypeError, "requires LinTSV2Spec"):
            build_online_setup_learner(
                SetupLearnerBuildSpec(
                    kind="lints",
                    shared=self._shared(),
                    algorithm=LinUCBV4Spec(),
                )
            )


if __name__ == "__main__":
    unittest.main()
