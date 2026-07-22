import ctypes
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP_ROOT = REPO_ROOT / "SetupPhase"
if str(SETUP_ROOT) not in sys.path:
    sys.path.insert(0, str(SETUP_ROOT))

from hypre.bindings import create_env  # noqa: E402


class PreparedEnvironmentDefaultWeightTest(unittest.TestCase):
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
import _project_paths  # noqa: F401
