if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import ctypes
import os
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[3]

from hypre.bindings import create_env  # noqa: E402
from hypre.bindings.config import configure_smoother_profile  # noqa: E402


class PreparedEnvironmentDefaultWeightTest(unittest.TestCase):
    def test_native_profile_preserves_stage_smoothers_while_tuning_weight(self) -> None:
        with patch.dict(os.environ, {"AMG_RELAX_TYPE": "18", "SETUP_RELAX_TYPE": "18"}):
            configure_smoother_profile("hypre_default")
            self.assertNotIn("AMG_RELAX_TYPE", os.environ)
            self.assertNotIn("SETUP_RELAX_TYPE", os.environ)
            env = create_env(nx=8, ny=8, nz=8, rhs_seed=123)
            try:
                prepared = env.prepare_rl({})
                self.assertAlmostEqual(prepared.initial_relax_weight, 1.0)
                self.assertEqual(env.cycle_relax_types, (13, 14, 9))
                for weight in (1.0, 1.1, 1.6):
                    env.step_rl(relax_weight=weight, sweeps_down=1, sweeps_up=1)
                    self.assertEqual(env.cycle_relax_types, (13, 14, 9))
            finally:
                env.close()

    def test_legacy_profile_retains_explicit_jacobi_override(self) -> None:
        with patch.dict(os.environ):
            configure_smoother_profile("hypre_default")
            configure_smoother_profile("legacy_l1_jacobi")
            env = create_env(nx=8, ny=8, nz=8, rhs_seed=123)
            try:
                env.prepare_rl({})
                self.assertEqual(env.cycle_relax_types, (18, 18, 18))
            finally:
                env.close()

    def test_direct_coarse_profile_survives_weight_and_type_updates(self) -> None:
        with patch.dict(os.environ):
            configure_smoother_profile("l1_jacobi_direct_coarse")
            env = create_env(nx=8, ny=8, nz=8, rhs_seed=123)
            try:
                env.prepare_rl({})
                self.assertEqual(env.cycle_relax_types, (18, 18, 9))
                for weight in (1.0, 1.3, 1.6):
                    env.step_rl(
                        relax_weight=weight, sweeps_down=1, sweeps_up=1,
                        relax_type=18,
                    )
                    self.assertEqual(env.cycle_relax_types, (18, 18, 9))
            finally:
                env.close()
            configure_smoother_profile("legacy_l1_jacobi")
            self.assertNotIn("AMG_COARSE_RELAX_TYPE", os.environ)
            env = create_env(nx=8, ny=8, nz=8, rhs_seed=123)
            try:
                env.prepare_rl({})
                self.assertEqual(env.cycle_relax_types, (18, 18, 18))
            finally:
                env.close()

    def test_native_full_solve_matches_default_weight_cycle_path(self) -> None:
        with patch.dict(os.environ):
            for profile, stages in (
                ("hypre_default", (13, 14, 9)),
                ("legacy_l1_jacobi", (18, 18, 18)),
                ("l1_jacobi_direct_coarse", (18, 18, 9)),
            ):
                with self.subTest(profile=profile):
                    configure_smoother_profile(profile)
                    env = create_env(nx=8, ny=8, nz=8, rhs_seed=123)
                    try:
                        solved = env.solve({}, tol=1.0e-6, max_iter=50)
                        env.prepare_rl({})
                        for _ in range(solved.iterations):
                            residual, _ = env.step_rl(
                                relax_weight=1.0, sweeps_down=1, sweeps_up=1,
                                tol=1.0e-6, max_cycles=50,
                            )
                        self.assertLessEqual(solved.residual_norm, 1.0e-6)
                        self.assertAlmostEqual(residual, solved.residual_norm, delta=1.0e-12)
                        self.assertEqual(env.cycle_relax_types, stages)
                    finally:
                        env.close()

    def test_prepare_clears_an_unrelated_sticky_hypre_error(self) -> None:
        hypre = ctypes.CDLL(
            str(REPO_ROOT / "hypre" / "install" / "lib" / "libHYPRE.dylib")
        )
        hypre.HYPRE_GetError.restype = ctypes.c_int
        hypre.HYPRE_ClearAllErrors.restype = ctypes.c_int
        hypre.HYPRE_BoomerAMGSetTol.restype = ctypes.c_int
        hypre.HYPRE_BoomerAMGSetTol.argtypes = [ctypes.c_void_p, ctypes.c_double]

        hypre.HYPRE_ClearAllErrors()
        self.assertNotEqual(hypre.HYPRE_BoomerAMGSetTol(None, 1.0e-6), 0)
        self.assertNotEqual(hypre.HYPRE_GetError(), 0)

        env = create_env(nx=8, ny=8, nz=8, rhs_seed=123)
        try:
            prepared = env.prepare_rl({"relax_wt": 1.3})
            self.assertAlmostEqual(prepared.initial_relax_weight, 1.3)
            self.assertEqual(hypre.HYPRE_GetError(), 0)
        finally:
            env.close()
            hypre.HYPRE_ClearAllErrors()


if __name__ == "__main__":
    unittest.main()
