"""Integration tests for the native AMG runtime binding."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from hypre.bindings import (
    AMGNativeError,
    AttemptStatus,
    NativeCode,
    SolveStatus,
    create_env,
    run_with_default_fallback,
    solve,
)
from hypre.bindings.config import configure_smoother_profile
from solve.core.outcomes import classify_rl_failure


class AMGRuntimeBindingTests(unittest.TestCase):
    def test_hierarchy_fingerprint_matches_rebuild_and_detects_rhs_changes(self):
        signatures = []
        for seed in (23, 23, 24):
            with create_env(nx=8, ny=8, nz=8, rhs_type=1, rhs_seed=seed) as env:
                env.prepare_rl()
                signature = env.hierarchy_fingerprint()
                self.assertEqual(env.cycle, 0)
                self.assertEqual(signature, env.hierarchy_fingerprint())
                signatures.append(signature)
                env.step_rl(relax_weight=1., sweeps_down=1, sweeps_up=1, tol=1e-6, max_cycles=50)
                with self.assertRaises(RuntimeError):
                    env.hierarchy_fingerprint()
        self.assertEqual(signatures[0], signatures[1])
        self.assertNotEqual(signatures[0], signatures[2])

    def test_native_timers_cover_completed_work_on_success_and_failure(self):
        root = Path(__file__).resolve().parents[2]
        compiler = shutil.which("mpicc")
        self.assertIsNotNone(compiler, "The native timing regression requires the MPI compiler")
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "test_rl_timing"
            subprocess.run([
                compiler, "-O2", "-I" + str(root / "hypre/install/include"),
                "-I" + str(root / "hypre/interfaces"),
                str(root / "hypre/interfaces/tests/test_rl_timing.c"),
                str(root / "hypre/interfaces/amg_rl_shared.c"),
                "-lm", "-o", str(executable),
            ], check=True, capture_output=True, text=True)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_full_solve_reports_typed_status_and_timing_identity(self):
        result = solve(
            nx=8,
            ny=8,
            nz=8,
            stencil=7,
            rhs_type=0,
            tol=1.0e-6,
            max_iter=50,
        )
        self.assertIs(result.status, SolveStatus.CONVERGED)
        self.assertAlmostEqual(
            result.runtime_sec,
            result.setup_runtime_sec + result.solve_runtime_sec,
            places=12,
        )
        self.assertGreater(result.iterations, 0)
        self.assertLessEqual(result.residual_norm, 1.0e-6)

    def test_max_cycle_status_does_not_poison_the_next_solve(self):
        limited = solve(
            nx=8,
            ny=8,
            nz=8,
            stencil=7,
            rhs_type=1,
            rhs_seed=19,
            tol=1.0e-14,
            max_iter=1,
        )
        self.assertIs(limited.status, SolveStatus.MAX_CYCLES)

        recovered = solve(
            nx=8,
            ny=8,
            nz=8,
            stencil=7,
            rhs_type=0,
            tol=1.0e-6,
            max_iter=50,
        )
        self.assertIs(recovered.status, SolveStatus.CONVERGED)

    @patch.dict(os.environ, {"AMG_RELAX_TYPE": "18", "AMG_COARSE_RELAX_TYPE": "9"})
    def test_prepare_and_step_share_typed_status_across_matrix_families(self):
        cases = (
            dict(nx=7, ny=7, nz=7, stencil=27, rhs_type=0, rhs_seed=23),
            dict(
                nx=8,
                ny=8,
                nz=8,
                stencil=0,
                rhs_type=1,
                rhs_seed=29,
                k=1.0,
                c=10.0,
                a0=100.0,
                a1=0.0,
                a2=0.0,
                a3=0.0,
            ),
        )
        for matrix_kwargs in cases:
            with self.subTest(stencil=matrix_kwargs["stencil"]):
                with create_env(**matrix_kwargs) as env:
                    prepared = env.prepare_rl(params={})
                    self.assertGreaterEqual(prepared.setup_runtime_sec, 0.0)
                    self.assertEqual(prepared.initial_residual_norm, 1.0)
                    self.assertGreater(
                        prepared.absolute_initial_residual_norm,
                        0.0,
                    )
                    for _ in range(50):
                        residual, runtime = env.step_rl(
                            relax_weight=prepared.initial_relax_weight,
                            sweeps_down=1,
                            sweeps_up=1,
                            tol=1.0e-6,
                            max_cycles=50,
                        )
                        self.assertGreaterEqual(runtime, 0.0)
                        if env.last_step.status is not SolveStatus.CONTINUE:
                            break
                    self.assertIn(
                        env.last_step.status,
                        {SolveStatus.CONVERGED, SolveStatus.MAX_CYCLES},
                    )
                    self.assertEqual(residual, env.last_step.residual_norm)
                    self.assertAlmostEqual(
                        residual,
                        env.last_step.absolute_residual_norm
                        / prepared.absolute_initial_residual_norm,
                        places=12,
                    )

    def test_prepared_step_uses_the_native_relative_residual_criterion(self):
        with create_env(
            nx=8,
            ny=8,
            nz=8,
            stencil=7,
            rhs_type=1,
            rhs_seed=31,
        ) as env:
            prepared = env.prepare_rl(params={})
            tolerance = 0.5
            residual, _runtime = env.step_rl(
                relax_weight=prepared.initial_relax_weight,
                sweeps_down=1,
                sweeps_up=1,
                tol=tolerance,
                max_cycles=5,
            )

            expected_status = (
                SolveStatus.CONVERGED
                if residual < tolerance
                else SolveStatus.CONTINUE
            )
            self.assertIs(env.last_step.status, expected_status)
            self.assertAlmostEqual(
                env.last_step.absolute_residual_norm,
                residual * prepared.absolute_initial_residual_norm,
                places=12,
            )

    def test_default_and_cycle_control_stop_at_the_same_relative_target(self):
        matrix = dict(
            nx=10,
            ny=10,
            nz=10,
            stencil=7,
            rhs_type=1,
            rhs_seed=73,
        )
        tolerance = 1.0e-6
        native = solve(tol=tolerance, max_iter=50, **matrix)

        with create_env(**matrix) as env:
            prepared = env.prepare_rl(params={})
            for cycles in range(1, 51):
                controlled_residual, _runtime = env.step_rl(
                    relax_weight=prepared.initial_relax_weight,
                    sweeps_down=1,
                    sweeps_up=1,
                    tol=tolerance,
                    max_cycles=50,
                )
                if env.last_step.status is not SolveStatus.CONTINUE:
                    break

        self.assertIs(native.status, SolveStatus.CONVERGED)
        self.assertIs(env.last_step.status, SolveStatus.CONVERGED)
        self.assertEqual(cycles, native.iterations)
        self.assertAlmostEqual(
            controlled_residual,
            native.residual_norm,
            places=12,
        )

    def test_setup_error_is_typed_and_cleared(self):
        with self.assertRaises(AMGNativeError) as raised:
            solve(nx=6, ny=6, nz=6, tol=2.0, max_iter=5)
        self.assertIs(raised.exception.code, NativeCode.SETUP_ERROR)
        self.assertEqual(raised.exception.operation, "setup")

        result = solve(nx=6, ny=6, nz=6, tol=1.0e-6, max_iter=50)
        self.assertIs(result.status, SolveStatus.CONVERGED)

    def test_cycle_limit_matches_linked_boomeramg_before_at_and_after_target(self):
        matrix = dict(nx=8, ny=8, nz=8, stencil=7, rhs_type=1, rhs_seed=73)
        tolerance = 1.0e-6
        with patch.dict(os.environ):
            for profile in ("hypre_default", "legacy_l1_jacobi", "l1_jacobi_direct_coarse"):
                configure_smoother_profile(profile)
                reference = solve(tol=tolerance, max_iter=100, **matrix)
                self.assertIs(reference.status, SolveStatus.CONVERGED)
                hit = reference.iterations
                self.assertGreater(hit, 1)
                for limit in (hit-1, hit, hit+1):
                    with self.subTest(profile=profile, first_target_cycle=hit, limit=limit):
                        native = solve(tol=tolerance, max_iter=limit, **matrix)
                        with create_env(**matrix) as env:
                            prepared = env.prepare_rl({})
                            for cycles in range(1, limit+1):
                                residual, _ = env.step_rl(
                                    relax_weight=prepared.initial_relax_weight,
                                    sweeps_down=1, sweeps_up=1,
                                    tol=tolerance, max_cycles=limit,
                                )
                                if env.last_step.status is not SolveStatus.CONTINUE:
                                    break
                        # Use the actual linked full-solve result as the oracle.
                        self.assertIs(env.last_step.status, native.status)
                        self.assertEqual(cycles, native.iterations)
                        self.assertAlmostEqual(residual, native.residual_norm, delta=1e-12)
                        reason = classify_rl_failure(
                            residual_norm=residual, iterations=cycles,
                            solve_tol=tolerance, solve_max_cycles=limit,
                        )
                        self.assertEqual(bool(reason), native.status is SolveStatus.MAX_CYCLES)
                        if limit == hit:
                            self.assertLess(native.residual_norm, tolerance)
                            self.assertIs(native.status, SolveStatus.MAX_CYCLES)
                            self.assertNotIn("residual_above_solve_tol", reason)

    def test_step_equality_with_tolerance_continues_as_boomeramg_loop_does(self):
        with patch.dict(os.environ):
            configure_smoother_profile("l1_jacobi_direct_coarse")
            with create_env(nx=8, ny=8, nz=8, stencil=7, rhs_type=1, rhs_seed=73) as env:
                env.prepare_rl({})
                tolerance, _ = env.step_rl(relax_weight=1.0, sweeps_down=1, sweeps_up=1)
                self.assertGreater(tolerance, 0.0)
                self.assertLess(tolerance, 1.0)
                env.prepare_rl({})
                residual, _ = env.step_rl(
                    relax_weight=1.0, sweeps_down=1, sweeps_up=1,
                    tol=tolerance, max_cycles=50,
                )
                self.assertEqual(residual, tolerance)
                self.assertIs(env.last_step.status, SolveStatus.CONTINUE)
                self.assertTrue(classify_rl_failure(
                    residual_norm=residual, iterations=1,
                    solve_tol=tolerance, solve_max_cycles=50,
                ))

    def test_real_setup_failure_runs_exactly_one_successful_fallback(self):
        fallback_calls = 0

        def fallback():
            nonlocal fallback_calls
            fallback_calls += 1
            result = solve(nx=8, ny=8, nz=8, tol=1.0e-6, max_iter=50)
            return {
                "runtime": result.runtime_sec,
                "setup_runtime": result.setup_runtime_sec,
                "solve_runtime": result.solve_runtime_sec,
                "failed": result.status is not SolveStatus.CONVERGED,
                "native_status": result.status.name.lower(),
                "residual_norm": result.residual_norm,
                "iterations": result.iterations,
            }

        recovery = run_with_default_fallback(
            lambda: solve(
                params={"strong_threshold": 2.0},
                nx=8,
                ny=8,
                nz=8,
                tol=1.0e-6,
                max_iter=50,
            ),
            fallback,
        )
        self.assertIs(recovery.primary.status, AttemptStatus.SETUP_FAILURE)
        self.assertTrue(recovery.recovered)
        self.assertEqual(fallback_calls, 1)
        self.assertAlmostEqual(
            recovery.end_to_end_runtime_sec,
            recovery.primary.end_to_end_runtime_sec
            + recovery.fallback.end_to_end_runtime_sec,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
