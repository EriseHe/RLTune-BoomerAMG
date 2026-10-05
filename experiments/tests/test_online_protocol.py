"""Frozen official configuration streams and the active online recovery protocol."""

from pathlib import Path
import copy
import json
import os
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import numpy as np
from experiments.paper_final.online import runner as runner
from experiments.paper_final.online import execution as engine
from experiments.paper_final.online import suite as suite_runner
from experiments.paper_final.online.configuration import (
    parse_joint_experiment_config,
    runtime_config_from_spec,
)
from experiments.paper_final.online.run import _validate_resolved
from experiments.paper_final.online.methods import build_composable_solve_runtime
from solve.controllers.common import ControllerBundle
from setup.learners.linucb.SharedLinUCB_AMG_v4 import SharedLinUCB_AMG_v4
from hypre.bindings import AttemptOutcome, RecoveryOutcome

ROOT = Path(__file__).resolve().parents[2]


def _mocked_online_run(destination, family):
    """Use real learners and deterministic costs to exercise the orchestration."""
    repo = ROOT
    rollback_outcomes = []
    original_rollback = SharedLinUCB_AMG_v4.rollback_recovery_transaction

    def checked_rollback(model):
        before = copy.deepcopy(model._recovery_transaction)
        rng_before = copy.deepcopy(model.rng.bit_generator.state)
        original_rollback(model)
        if before is not None:
            unchanged = all(
                np.array_equal(getattr(model, name), before[name])
                for name in ("A_inv", "b", "failure_b")
            )
            unchanged &= (
                model.t == before["t"]
                and model.failure_observation_count
                == before["failure_observation_count"]
            )
            unchanged &= model.rng.bit_generator.state == rng_before
            rollback_outcomes.append(bool(unchanged))

    raw = json.loads(
        (
            repo
            / "experiments/paper_final/04_online/20260920_formal"
            / ("advection_60.json" if family == "advection" else "diffusion_60.json")
        ).read_text()
    )
    raw["stream"].update(
        cases=8, online_cases=8, cases_per_seed=1, group_take=8, expected_sha256=""
    )
    raw["methods"][2]["solve_activation_case"] = 2
    raw["reporting"]["progress_every"] = 8
    runtime = runtime_config_from_spec(
        parse_joint_experiment_config(raw, output_dir_override=destination / "run")
    )
    clock = {"tick": 1_000_000_000, "sec_calls": 0, "ns_calls": 0}

    def sec():
        clock["tick"] += 1_000_000
        clock["sec_calls"] += 1
        return clock["tick"] / 1e9

    def ns():
        clock["tick"] += 1_000_000
        clock["ns_calls"] += 1
        return clock["tick"]

    count = {"native": 0, "rl": 0}
    call_log = []
    bundles = []

    def native(*, params, mkw, **kwargs):
        count["native"] += 1
        i = count["native"]
        call_log.append({"kind": "native", "i": i, "params": params, "mkw": mkw})
        status = (
            "setup_failure"
            if i in {2, 3, 4, 9, 15, 16, 17}
            else "solve_failure"
            if i in {7, 18}
            else "success"
        )
        setup = 0.010 + (i % 3) * 0.001
        solve = 0.0 if status == "setup_failure" else 0.020 + (i % 4) * 0.002
        return {
            "runtime": setup + solve,
            "native_runtime": setup + solve,
            "setup_runtime": setup,
            "solve_runtime": solve,
            "native_solve_runtime": solve,
            "infer_runtime": 0.0,
            "failed": status != "success",
            "attempt_status": status,
            "failure_origin": "solver",
            "failure_reason": "setup_failure"
            if status == "setup_failure"
            else "nonconvergence"
            if status == "solve_failure"
            else "",
            "residual_norm": 1e-8 if status == "success" else 1.0,
            "iterations": 0
            if status == "setup_failure"
            else 50
            if status == "solve_failure"
            else 2,
            "initial_residual_norm": 1.0,
            "final_residual_norm": 1e-8 if status == "success" else 1.0,
        }

    def baseline(*, mkw, **kwargs):
        primary = {
            "runtime": 0.040,
            "native_runtime": 0.040,
            "setup_runtime": 0.015,
            "solve_runtime": 0.025,
            "native_solve_runtime": 0.025,
            "infer_runtime": 0.0,
            "failed": False,
            "attempt_status": "success",
            "residual_norm": 1e-8,
            "iterations": 3,
        }
        out = RecoveryOutcome(primary=AttemptOutcome.from_mapping(primary)).to_result()
        out["recovery_protocol_applied"] = True
        out["native_status"] = "success"
        return out

    def run_case(self, case):
        count["rl"] += 1
        bundles.append(self)
        primary = native(params=case.params, mkw=case.mkw)
        if primary["attempt_status"] == "setup_failure":
            return primary
        before = self.controller.snapshot_learning_state()
        self.controller.start_episode(initial_environment_weight=1.0)
        features = self.encoder.encode(
            mkw=case.mkw,
            problem_context=case.problem_context,
            initial_residual=1.0,
            residual=1.0,
            previous_residual=1.0,
            cycle=0,
            last_weight=1.0,
            last_cycle_time=0.0,
            setup_params=case.params,
        )
        ai, w, meta = self.controller.select_action(features, explore=True, cycle=0)
        fallback = case.fallback_attempt() if primary["failed"] else None
        recovery = RecoveryOutcome(
            primary=AttemptOutcome.from_mapping(primary),
            fallback=None
            if fallback is None
            else AttemptOutcome.from_mapping(fallback),
        )
        if not recovery.unrecovered_failure:
            self.controller.update(
                features=features,
                action_index=ai,
                cost=recovery.native_runtime_sec,
                next_features=np.zeros_like(features),
                terminal=True,
                native_cycle_cost=primary["solve_runtime"],
                residual_ratio=0.1,
            )
            self.controller.finish_episode(learned=True)
        else:
            self.controller.restore_learning_state(before)
        out = recovery.to_result()
        out.update(
            recovery_protocol_applied=True,
            controller_update_committed=not recovery.unrecovered_failure,
            cycle_actions=[w],
            cycle_times=[primary["solve_runtime"]],
            cycle_action_metadata=[meta],
            infer_runtime=0.004,
            feature_runtime=0.001,
            decision_runtime=0.001,
            update_runtime=0.001,
            lifecycle_runtime=0.001,
            native_runtime=recovery.native_runtime_sec,
            native_solve_runtime=recovery.to_result()["solve_runtime"],
        )
        out["runtime"] = out["native_runtime"] + 0.004
        out["solve_runtime"] = out["native_solve_runtime"] + 0.004
        call_log.append(
            {
                "kind": "rl",
                "n": count["rl"],
                "action": ai,
                "weight": w,
                "rng": copy.deepcopy(self.controller.rng.bit_generator.state),
            }
        )
        return out

    with (
        patch.object(
            SharedLinUCB_AMG_v4, "rollback_recovery_transaction", checked_rollback
        ),
        patch.object(time, "perf_counter", side_effect=sec),
        patch.object(time, "perf_counter_ns", side_effect=ns),
        patch.object(engine, "solve_no_rl_case", side_effect=native),
        patch.object(engine, "solve_default_baseline_case", side_effect=baseline),
        patch.object(ControllerBundle, "run_case", run_case),
        patch.object(runner, "_git_revision", return_value="same-scientific-source"),
    ):
        result = runner.run(runtime)
    records = {
        p.stem: [json.loads(line) for line in p.read_text().splitlines()]
        for p in (destination / "run" / "trajectories").glob("*.jsonl")
    }
    return result, records, count, clock, rollback_outcomes


class OfficialOnlineProtocolTests(unittest.TestCase):
    def test_all_captured_configs_keep_exact_streams_and_guard(self):
        paths = []
        for base in (
            ROOT / "experiments/paper_final/04_online/20260920_formal",
            ROOT / "experiments/paper_final/reproduction/online",
        ):
            for path in sorted(base.rglob("*.json")):
                raw = json.loads(path.read_text())
                if "methods" in raw and "problem" in raw:
                    paths.append((path, raw))
        self.assertEqual(len(paths), 42)
        with patch.dict(os.environ, dict(os.environ), clear=True):
            for path, raw in paths:
                with self.subTest(config=path.name):
                    runtime = runtime_config_from_spec(
                        parse_joint_experiment_config(raw)
                    )
                    validated = _validate_resolved(runtime)
                    self.assertEqual(
                        validated["stream"]["sha256"], raw["stream"]["expected_sha256"]
                    )
                    self.assertEqual(validated["aot_max_selections_per_case"], 3)
                    self.assertEqual(
                        [m.name for m in runtime.methods],
                        list(suite_runner.METHOD_LABELS),
                    )
        with patch.object(suite_runner, "file_hash", return_value="changed"):
            with self.assertRaisesRegex(
                ValueError, "pinned official protocol reference changed"
            ):
                suite_runner._reference_protocol()

    def test_solve_starts_after_problem_1000_with_prescribed_seed(self):
        raw = json.loads(
            (
                ROOT
                / "experiments/paper_final/04_online/20260920_formal/diffusion_60.json"
            ).read_text()
        )
        runtime = runtime_config_from_spec(parse_joint_experiment_config(raw))
        bundle = build_composable_solve_runtime(
            runtime, specs=runtime.methods
        ).controller_bundles["bandit_lstdq"]
        self.assertEqual(
            bundle.controller.rng.bit_generator.state,
            np.random.default_rng(runtime.controller_seed + 2018).bit_generator.state,
        )
        native = {
            "runtime": 0.03,
            "setup_runtime": 0.01,
            "solve_runtime": 0.02,
            "failed": False,
        }
        dummy = Mock()
        dummy.run_case.return_value = native
        common = dict(
            method="bandit_lstdq",
            solve=engine.SolveExecutionConfig(1e-6, 50),
            mkw={},
            case_progress=0.0,
            controller_bundles={"bandit_lstdq": dummy},
            composable_specs={m.name: m for m in runtime.methods},
        )
        with patch.object(engine, "solve_no_rl_case", return_value=native) as default:
            engine.make_method_solver(case_index=999, **common)({})
            dummy.run_case.assert_not_called()
            engine.make_method_solver(case_index=1000, **common)({})
            dummy.run_case.assert_called_once()
            default.assert_called_once()

    def test_reselection_fallback_rollback_and_timing_in_both_contexts(self):
        for family in ("diffusion", "advection"):
            with (
                self.subTest(family=family),
                tempfile.TemporaryDirectory() as directory,
                patch.dict(os.environ, dict(os.environ), clear=True),
            ):
                result, rows, count, clock, rollbacks = _mocked_online_run(
                    Path(directory), family
                )
                self.assertEqual(count, {"native": 24, "rl": 7})
                self.assertEqual(
                    clock, {"tick": 1194000000, "sec_calls": 72, "ns_calls": 122}
                )
                self.assertEqual(
                    result["bandit_online_steps"],
                    {"bandit_default": 8, "bandit_lstdq": 8},
                )
                construction = rows["bandit_lstdq"][0]["outcome"]
                self.assertEqual(construction["primary_attempt_count"], 3)
                self.assertTrue(construction["fallback_used"])
                self.assertTrue(construction["recovered"])
                solve_failure = rows["bandit_lstdq"][1]["outcome"]
                self.assertEqual(solve_failure["primary_attempt_count"], 1)
                self.assertTrue(solve_failure["fallback_used"])
                unrecovered = rows["bandit_default"][4]["outcome"]
                self.assertEqual(unrecovered["primary_attempt_count"], 3)
                self.assertTrue(unrecovered["unrecovered_failure"])
                self.assertFalse(unrecovered["bandit_update_committed"])
                self.assertEqual(unrecovered["bandit_observation_count"], 0)
                self.assertEqual(rollbacks, [True])
                expected = np.random.default_rng(56772120)
                methods = tuple(suite_runner.METHOD_LABELS)
                for recorded in rows["method_order"]:
                    self.assertEqual(
                        recorded["method_order"],
                        [methods[int(i)] for i in expected.permutation(3)],
                    )


if __name__ == "__main__":
    unittest.main()
