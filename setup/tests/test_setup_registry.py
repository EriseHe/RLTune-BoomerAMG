from __future__ import annotations

from setup.tests import _project_paths  # noqa: F401

import unittest
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from setup.learners.common import (
    LinTSV2Spec,
    LinUCBV4Spec,
    LinUCBV5RBFSpec,
    LinUCBV5Spec,
    LinUCBV6Spec,
    SetupLearnerFactoryRequest,
    SharedSetupLearnerSpec,
)
from setup.learners.linucb.config import LinUCBV4Spec as FamilyLinUCBV4Spec
from setup.learners.linucb.config import LinUCBV5Spec as FamilyLinUCBV5Spec
from setup.learners.linucb.config import (
    LinUCBV5RBFSpec as FamilyLinUCBV5RBFSpec,
)
from setup.learners.linucb.config import LinUCBV6Spec as FamilyLinUCBV6Spec
from setup.learners.linucb.factory import (
    build_linucb_v4_learner,
    build_linucb_v5_learner,
    build_linucb_v5_rbf_learner,
    build_linucb_v6_learner,
)
from setup.learners.thompson.config import LinTSV2Spec as FamilyLinTSV2Spec
from setup.learners.thompson.factory import build_lints_v2_learner
from setup.registry import (
    COMPOSABLE_SETUP_KINDS,
    ONLINE_SETUP_KINDS,
    SetupLearnerBuildSpec,
    build_online_setup_learner,
    make_setup_learner_spec,
    setup_kind_registration,
)
from setup.learners import (
    SharedLinTS_AMG_v2,
    SharedLinUCB_AMG_v4,
    SharedLinUCB_AMG_v5,
    SharedLinUCB_AMG_v5_RBF,
    SharedLinUCB_AMG_v6,
)
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
            (
                "default",
                "linucb",
                "linucb_v5",
                "linucb_v5_rbf",
                "linucb_v6",
                "lints",
            ),
        )
        self.assertEqual(
            ONLINE_SETUP_KINDS,
            (
                "linucb",
                "linucb_v5",
                "linucb_v5_rbf",
                "linucb_v6",
                "lints",
            ),
        )
        self.assertFalse(setup_kind_registration("random").online)
        self.assertNotIn("random", COMPOSABLE_SETUP_KINDS)
        self.assertIs(
            setup_kind_registration("linucbv4").learner_type,
            SharedLinUCB_AMG_v4,
        )
        self.assertIs(
            setup_kind_registration("linucbv5").learner_type,
            SharedLinUCB_AMG_v5,
        )
        self.assertIs(
            setup_kind_registration("linucbv5rbf").learner_type,
            SharedLinUCB_AMG_v5_RBF,
        )
        self.assertIs(
            setup_kind_registration("linucbv6").learner_type,
            SharedLinUCB_AMG_v6,
        )
        self.assertIs(
            setup_kind_registration("lints_v2").learner_type,
            SharedLinTS_AMG_v2,
        )
        self.assertIs(
            setup_kind_registration("linucb").factory,
            build_linucb_v4_learner,
        )
        self.assertIs(
            setup_kind_registration("linucb_v5").factory,
            build_linucb_v5_learner,
        )
        self.assertIs(
            setup_kind_registration("linucb_v5_rbf").factory,
            build_linucb_v5_rbf_learner,
        )
        self.assertIs(
            setup_kind_registration("linucb_v6").factory,
            build_linucb_v6_learner,
        )
        self.assertIs(
            setup_kind_registration("lints").factory,
            build_lints_v2_learner,
        )

    def test_algorithm_specs_are_family_owned_compatibility_reexports(self) -> None:
        self.assertIs(LinUCBV4Spec, FamilyLinUCBV4Spec)
        self.assertIs(LinUCBV5Spec, FamilyLinUCBV5Spec)
        self.assertIs(LinUCBV5RBFSpec, FamilyLinUCBV5RBFSpec)
        self.assertIs(LinUCBV6Spec, FamilyLinUCBV6Spec)
        self.assertIs(LinTSV2Spec, FamilyLinTSV2Spec)
        self.assertEqual(
            LinUCBV4Spec.__module__,
            "setup.learners.linucb.config",
        )
        self.assertEqual(
            LinUCBV5Spec.__module__,
            "setup.learners.linucb.config",
        )
        self.assertEqual(
            LinUCBV5RBFSpec.__module__,
            "setup.learners.linucb.config",
        )
        self.assertEqual(
            LinUCBV6Spec.__module__,
            "setup.learners.linucb.config",
        )
        self.assertEqual(
            LinTSV2Spec.__module__,
            "setup.learners.thompson.config",
        )

    def test_bare_checkpoint_alias_preserves_identity_in_fresh_process(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            (
                str(repo_root),
                environment.get("PYTHONPATH", ""),
            )
        )
        program = """
import importlib
from pathlib import Path
import sys

canonical = importlib.import_module("setup.learners")
bare = importlib.import_module("learners")
canonical_leaf = importlib.import_module(
    "setup.learners.linucb.SharedLinUCB_AMG_v4"
)
bare_leaf = importlib.import_module(
    "learners.linucb.SharedLinUCB_AMG_v4"
)
assert canonical.SharedLinUCB_AMG_v4 is bare.SharedLinUCB_AMG_v4
assert canonical_leaf is bare_leaf
canonical_linucb_config = importlib.import_module(
    "setup.learners.linucb.config"
)
assert canonical_linucb_config is importlib.import_module(
    "learners.linucb.config"
)
canonical_linucb_factory = importlib.import_module(
    "setup.learners.linucb.factory"
)
assert canonical_linucb_factory is importlib.import_module(
    "learners.linucb.factory"
)
canonical_lints_config = importlib.import_module(
    "setup.learners.thompson.config"
)
assert canonical_lints_config is importlib.import_module(
    "learners.thompson.config"
)
canonical_lints_factory = importlib.import_module(
    "setup.learners.thompson.factory"
)
assert canonical_lints_factory is importlib.import_module(
    "learners.thompson.factory"
)
assert str(Path.cwd() / "setup") not in sys.path
"""
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

        family_built = build_linucb_v4_learner(
            SetupLearnerFactoryRequest(
                shared=shared,
                algorithm=LinUCBV4Spec(),
            )
        )
        self.assertIsInstance(family_built, SharedLinUCB_AMG_v4)
        np.testing.assert_allclose(family_built.A_inv, direct.A_inv)

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

        family_built = build_lints_v2_learner(
            SetupLearnerFactoryRequest(
                shared=shared,
                algorithm=LinTSV2Spec(**algorithm),
            )
        )
        self.assertIsInstance(family_built, SharedLinTS_AMG_v2)
        np.testing.assert_allclose(
            family_built._precision_cholesky,
            direct._precision_cholesky,
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
