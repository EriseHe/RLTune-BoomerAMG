from __future__ import annotations

import unittest

from hypre.bindings import (
    AMGNativeError,
    AttemptStatus,
    NativeCode,
    SolveStatus,
    create_env,
    run_with_default_fallback,
    solve,
)


class AMGRuntimeBindingTests(unittest.TestCase):
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

    def test_setup_error_is_typed_and_cleared(self):
        with self.assertRaises(AMGNativeError) as raised:
            solve(nx=6, ny=6, nz=6, tol=2.0, max_iter=5)
        self.assertIs(raised.exception.code, NativeCode.SETUP_ERROR)
        self.assertEqual(raised.exception.operation, "setup")

        result = solve(nx=6, ny=6, nz=6, tol=1.0e-6, max_iter=50)
        self.assertIs(result.status, SolveStatus.CONVERGED)

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
