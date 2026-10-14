"""Verify sweep wiring with simulated outcomes; never execute a native solve."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.paper_final import run_06_beta_sweep as sweep

import numpy as np

from experiments.paper_final.common.artifacts import checkpoint_payload_hash, dump, file_hash, read, ready
from experiments.paper_final.online.configuration import parse_joint_experiment_config
from experiments.paper_final.online.setup_branches import GenericBanditPolicy
from experiments.paper_final.test_06_recovery import outcome
from problems.heterogeneous_3d import context_from_matrix_kwargs
from setup.tests import test_linucb
from solve.tests import test_recursive_lstdq


class BetaConfigurationTests(unittest.TestCase):
    def test_version_two_has_one_beta_and_rejects_legacy_overrides(self):
        for beta in (0.5, 1.0, 2.0, 3.75):
            raw = read(sweep.CONFIG)
            raw["solve"]["lstdq"]["beta"] = beta
            parsed = parse_joint_experiment_config(raw)
            self.assertEqual(parsed.solve.recursive_lstdq.uncertainty_beta, beta)
            for legacy in ("lstdq_v2", "lstdq_v3"):
                changed = copy.deepcopy(raw)
                changed["solve"][legacy] = {"beta": 99}
                with self.assertRaisesRegex(ValueError, "uses solve.lstdq only"):
                    parse_joint_experiment_config(changed)

    def test_beta_values_are_variable_but_branch_names_must_be_distinct(self):
        self.assertEqual(sweep.branch_betas([0.5, 1, 2]), sweep.BETAS)
        self.assertEqual(list(sweep.branch_betas([0, 0.25, 3.75]).values()), [0, 0.25, 3.75])
        for values in ((1, 1, 2), (1, 2), (1, 2, float("nan")),
                       (-1, 1, 2), (1, 2, float("inf")), (1, 1+1e-14, 2)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                sweep.branch_betas(values)

    def test_prepare_only_pairs_streams_and_real_untrained_controllers(self):
        forbidden = AssertionError("Preparation tried to run an experiment")
        with tempfile.TemporaryDirectory() as temporary, \
             patch("solve.core.episode.run_td_episode", side_effect=forbidden), \
             patch("experiments.paper_final.online.native_evaluation.solve_no_rl_case", side_effect=forbidden), \
             patch("hypre.bindings.create_env", side_effect=forbidden), \
             patch.object(sweep, "execute_case", side_effect=forbidden), \
             patch.object(sweep, "build_named_setup_branches", side_effect=forbidden), \
             patch.object(sweep.subprocess, "Popen", side_effect=forbidden), \
             patch.object(sweep.subprocess, "check_output", return_value="test-revision"), \
             patch.object(sweep, "native_hashes", return_value={}), \
             patch.object(sweep, "source_hashes", return_value={}):
            directory = Path(temporary) / "prepared"
            protocol = sweep.prepare(directory)
            self.assertEqual(protocol["betas"], sweep.BETAS)
            self.assertEqual(protocol["baselines"], sweep.BASELINES)
            self.assertFalse(protocol["rl_recovery_cost"])
            self.assertEqual(protocol["total_attempt_budget"], 4)
            status = read(directory / "status.json")
            self.assertEqual(status["status"], "prepared")
            self.assertFalse(status["experiment_started"])
            self.assertEqual(status["native_solves_executed"], 0)
            self.assertEqual(len((directory / "inputs.jsonl").read_text().splitlines()), 5000)
            baseline = None
            for name, beta in sweep.BETAS.items():
                raw = read(directory / "configs" / f"{name}.json")
                runtime = sweep.decode(raw, directory / name)
                self.assertEqual(runtime.bundle.controller.spec.uncertainty_beta, beta)
                self.assertEqual(runtime.bundle.controller.steps, 0)
                self.assertEqual(runtime.bundle.controller.episodes, 0)
                self.assertEqual(runtime.method.solve_activation_case, 1000)
                self.assertEqual(len(runtime.stream), 5000)
                check = protocol["preparation_checks"][name]
                self.assertEqual(runtime.manifest["sha256"], check["stream_sha256"])
                for mkw, context in runtime.stream:
                    np.testing.assert_array_equal(context, context_from_matrix_kwargs(mkw))
                normalized = sweep.normalized_configuration(raw)
                if baseline is None:
                    baseline = normalized
                self.assertEqual(normalized, baseline)
            self.assertEqual(len({protocol["preparation_checks"][name]["initial_controller_sha256"] for name in sweep.BETAS}), 1)
            for name in sweep.BASELINES:
                runtime = sweep.decode(read(directory / "configs" / f"{name}.json"), directory / name)
                self.assertIsNone(runtime.bundle)
                self.assertEqual(runtime.method.solve_kind, "default")
                self.assertEqual(runtime.manifest["sha256"], protocol["preparation_checks"]["beta_2"]["stream_sha256"])
                self.assertEqual(len(runtime.stream), 5000)
            self.assertFalse((directory / "prefix").exists())
            self.assertFalse((directory / "workers.json").exists())
            self.assertFalse((directory / "joint_start.flag").exists())
            self.assertEqual(list(directory.glob("*.log")), [])

    def test_decode_checks_the_prepared_stream_hash(self):
        raw = read(sweep.CONFIG)
        raw["stream"]["expected_sha256"] = "unrecognized-wrong-hash"
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "prepared input hash"):
                sweep.decode(raw, Path(temporary))


class SweepRecoveryWiringTests(unittest.TestCase):
    def runtime(self):
        model = test_linucb.SharedLinUCBTests._model()
        controller = test_recursive_lstdq.RecursiveLstdqEpisodeTests._controller()
        mkw = dict(nx=60, ny=60, nz=60, stencil=102, rhs_type=1, rhs_seed=42,
                   k=10**-2.5, c=0, a0=0, a1=0, a2=0, a3=0)
        context = context_from_matrix_kwargs(mkw)
        return SimpleNamespace(
            args=SimpleNamespace(problem="rs_convection_3d_type2", tol=1e-6, max_cycles=50),
            method=SimpleNamespace(solve_activation_case=1000, setup_kind="linucb", setup_context="rs_diffusion"),
            branch=SimpleNamespace(policy=GenericBanditPolicy(model), parameter_space={}),
            bundle=SimpleNamespace(controller=controller, encoder=object()),
            stream=[(mkw, context)]*1001, rows=Mock(), name="beta_2")

    @staticmethod
    def update_one_episode(controller):
        controller.start_episode(initial_environment_weight=1.0)
        features = np.array([1.0, 0.2, 0.4])
        controller.update(features=features, action_index=0, cost=0.02,
                          next_features=features, terminal=True, next_action_index=None)
        controller.finish_episode(learned=True)

    def simulate(self, runtime, index, statuses, *, fallback_success=True):
        seen = []
        def episode(**kwargs):
            # run_td_episode must retain provisional attempt updates, with no
            # internal default solve or terminal recovery/penalty target.
            self.assertIsNone(kwargs["fallback_attempt"])
            self.assertEqual(kwargs["failure_penalty_sec"], 0.0)
            self.assertTrue(kwargs["learn"] and kwargs["explore"])
            self.assertTrue(kwargs["record_action_metadata"])
            controller = kwargs["controller"]
            status = statuses[len(seen)]
            seen.append(kwargs["params"])
            if status == "setup_failure":
                return outcome(status, solve=0.0)
            self.update_one_episode(controller)
            result = outcome(status)
            beta = controller.spec.uncertainty_beta
            result.update(iterations=1, cycle_actions=[1.0], cycle_times=[0.02],
                cycle_residuals=[result["residual_norm"]], cycle_explored=[False],
                cycle_epsilons=[0.3], cycle_selected_uncertainties=[0.001],
                cycle_selected_q_values=[0.02], cycle_selection_scores=[max(0.02-beta*0.001, 0)],
                cycle_forced_default_actions=[False], td_errors=[0.02], postfit_td_errors=[0.01])
            return result
        native_status = "success" if fallback_success else "nonconvergence"
        with patch("solve.core.episode.run_td_episode", side_effect=episode) as rl, \
             patch("experiments.paper_final.online.native_evaluation.solve_no_rl_case",
                   return_value=outcome(native_status, setup=0.03, solve=0.04)) as native:
            row, _ = sweep.execute_case(runtime, index, 0.0)
        return row, rl, native

    def test_input_1000_is_setup_only_and_1001_activates_rl(self):
        runtime = self.runtime()
        row, rl, native = self.simulate(runtime, 999, [])
        self.assertFalse(row["rl_active"])
        rl.assert_not_called()
        native.assert_called_once()
        self.assertEqual(runtime.bundle.controller.steps, 0)
        row, rl, native = self.simulate(runtime, 1000, ["success"])
        self.assertTrue(row["rl_active"])
        rl.assert_called_once()
        native.assert_not_called()
        self.assertEqual(runtime.bundle.controller.steps, 1)

    def test_three_failed_rl_attempts_then_default_do_not_change_rl_targets(self):
        runtime = self.runtime()
        row, rl, native = self.simulate(runtime, 1000, ["nonconvergence"]*3)
        result = row["outcome"]
        self.assertEqual(rl.call_count, 3)
        native.assert_called_once()
        self.assertFalse(result["failed"])
        self.assertTrue(result["fallback_used"])
        self.assertAlmostEqual(result["runtime"], 0.16)
        self.assertEqual(result["replayed_transitions"], 0)
        self.assertEqual(result["target_replay_runtime"], 0)
        self.assertTrue(all(a["rl_recovery_cost"] == 0 for a in result["primary_attempts"]))
        self.assertTrue(result["controller_update_committed"])
        self.assertEqual(runtime.branch.policy.model.t, 3)
        self.assertTrue(all(a["learning_cost"] == 2 * a["attempt_cost"] for a in result["primary_attempts"]))
        expected = test_recursive_lstdq.RecursiveLstdqEpisodeTests._controller()
        for _ in range(3):
            self.update_one_episode(expected)
        actual = runtime.bundle.controller
        self.assertEqual((actual.steps, actual.episodes), (3, 3))
        for key in ("a_matrix", "a_inverse", "b", "theta", "episode_moment_covariance"):
            np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))

    def test_final_failure_rolls_back_rl_but_keeps_setup_feedback_and_actual_cost(self):
        runtime = self.runtime()
        bandit = runtime.branch.policy.model
        original = (bandit.A_inv.copy(), bandit.b.copy())
        row, rl, native = self.simulate(runtime, 1000, ["nonconvergence"]*3, fallback_success=False)
        self.assertTrue(row["outcome"]["failed"])
        self.assertEqual(rl.call_count, 3)
        native.assert_called_once()
        self.assertAlmostEqual(row["outcome"]["runtime"], 0.16)
        self.assertEqual((runtime.bundle.controller.steps, runtime.bundle.controller.episodes), (0, 0))
        self.assertEqual(bandit.t, 3)
        for actual, before in zip((bandit.A_inv, bandit.b), original):
            self.assertFalse(np.array_equal(actual, before))

    def test_setup_failure_and_rl_failure_share_three_attempts(self):
        runtime = self.runtime()
        row, rl, native = self.simulate(runtime, 1000, ["setup_failure", "nonconvergence", "success"])
        self.assertEqual(rl.call_count, 3)
        native.assert_not_called()
        self.assertFalse(row["outcome"]["failed"])
        self.assertEqual(runtime.bundle.controller.episodes, 2)

    def test_audit_rejects_scores_from_the_wrong_beta(self):
        row, _, _ = self.simulate(self.runtime(), 1000, ["success"])
        with self.assertRaisesRegex(ValueError, "does not match the branch beta"):
            sweep.audit_case(row, 0.5)

    def test_linucb_control_never_activates_rl_and_uses_the_same_recovery_budget(self):
        runtime = self.runtime()
        runtime.name, runtime.bundle = "linucb", None
        runtime.stream = [runtime.stream[0]]*5000
        for index in (0, 999, 1000, 4999):
            row, rl, native = self.simulate(runtime, index, [])
            rl.assert_not_called()
            native.assert_called_once()
            self.assertFalse(row["rl_active"])
            self.assertEqual(row["shared_prefix"], index < 1000)
            self.assertFalse(row["outcome"]["controller_update_committed"])
        with patch("solve.core.episode.run_td_episode", side_effect=AssertionError("RL in control")), \
             patch("experiments.paper_final.online.native_evaluation.solve_no_rl_case",
                   side_effect=[outcome("nonconvergence")]*3+[outcome("success")]) as native:
            row, _ = sweep.execute_case(runtime, 1000, 0.0)
        self.assertEqual(native.call_count, 4)
        self.assertEqual(row["outcome"]["primary_attempt_count"], 3)
        self.assertTrue(row["outcome"]["fallback_used"])
        self.assertFalse(row["outcome"]["failed"])

    def test_default_control_runs_one_ordinary_solve_even_when_it_fails(self):
        runtime = self.runtime()
        for status in ("success", "nonconvergence", "setup_failure"):
            with patch("experiments.paper_final.online.native_evaluation.solve_no_rl_case",
                       return_value=outcome(status)) as native, \
                 patch("solve.core.episode.run_td_episode", side_effect=AssertionError("RL in default")):
                row = sweep.execute_default_case(runtime, 0)
            native.assert_called_once()
            self.assertEqual(row["outcome"]["failed"], status != "success")
            self.assertEqual(row["outcome"]["primary_attempt_count"], 1)
            self.assertFalse(row["outcome"]["fallback_used"])
            self.assertFalse(row["shared_prefix"])
            self.assertEqual(sweep.metrics([row])["rl_actions"], 0)


class PairingBarrierTests(unittest.TestCase):
    def test_release_requires_identical_prefix_candidates_and_rng_states(self):
        prefix = {key: "same" for key in (
            "stream_sha256", "candidate_sha256", "initial_bandit_sha256",
            "initial_controller_sha256", "prefix_bandit_sha256", "native_sha256")}
        states = {name: dict(prefix, effective_beta=beta, controller_steps=0, candidate_cursor=3000)
                  for name, beta in sweep.BETAS.items()}
        states["linucb"] = dict(prefix, effective_beta=None, initial_controller_sha256=None,
                                controller_steps=0, candidate_cursor=3000)
        states["default"] = dict(prefix, effective_beta=None, controller_steps=0, candidate_cursor=None,
            candidate_sha256=None, initial_bandit_sha256=None, initial_controller_sha256=None,
            prefix_bandit_sha256=None)
        sweep.verify_pairing(states, prefix)
        for key, wrong in (("prefix_bandit_sha256", "different"), ("candidate_sha256", "different"),
                           ("initial_controller_sha256", "different"), ("effective_beta", 99),
                           ("candidate_cursor", 2999), ("controller_steps", 1)):
            changed = copy.deepcopy(states)
            changed["beta_0p5"][key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                sweep.verify_pairing(changed, prefix)
        for name in sweep.BASELINES:
            changed = copy.deepcopy(states)
            changed.pop(name)
            with self.subTest(missing=name), self.assertRaisesRegex(ValueError, "both controls"):
                sweep.verify_pairing(changed, prefix)


class PrefixReuseTests(unittest.TestCase):
    def test_reuse_verifies_inputs_checkpoint_candidates_and_unchanged_solver_code(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(sweep, "PREFIX", 2):
            source, output = Path(temporary) / "source_run", Path(temporary) / "new_run"
            (source / "prefix").mkdir(parents=True)
            runtime = SweepRecoveryWiringTests.runtime(self)
            rows = []
            for index in range(2):
                row, _, _ = SweepRecoveryWiringTests.simulate(self, runtime, index, [])
                rows.append(row)
            trajectory = source / "prefix/trajectory.jsonl"
            trajectory.write_text("".join(json.dumps(ready(row))+"\n" for row in rows))
            runtime.branch.policy.model.save_mutable_state(source / "prefix/final_bandit.npz")
            candidate = source / "prefix/aot_candidate_schedules/recommended/structured512/candidate_ids.npy"
            candidate.parent.mkdir(parents=True)
            candidate.write_bytes(b"fixture-candidates")
            pairing = dict(stream_sha256="same-stream", initial_controller_sha256="same-controller",
                prefix_bandit_sha256=checkpoint_payload_hash(source / "prefix/final_bandit.npz", exclude=("metadata",)),
                candidate_sha256=file_hash(candidate))
            result = dict(complete=True, cases=2, controller=dict(steps=0, episodes=0), pairing=pairing,
                          trajectory_sha256=file_hash(trajectory))
            protocol = dict(reference_branch="beta_2", betas=sweep.BETAS, native_sha256={},
                            variant=asdict(sweep.VARIANT), preparation_checks={"beta_2": pairing})
            config = read(sweep.CONFIG)
            old_source = {"solve/core/episode.py": "same", "experiments/paper_final/run_06_beta_sweep.py": "old"}
            new_source = {**old_source, "experiments/paper_final/run_06_beta_sweep.py": "new"}
            for directory in (source, output):
                dump(directory / "protocol.json", protocol)
                dump(directory / "configs/beta_2.json", config)
            dump(source / "prefix/result.json", result)
            dump(source / "source_manifest.json", old_source)
            dump(output / "source_manifest.json", new_source)
            (output / "inputs.jsonl").write_text("".join(json.dumps(dict(mkw=r["mkw"],context=r["context"]))+"\n" for r in rows))
            candidate.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "candidate schedule changed"):
                sweep.reuse_prefix(output, source)
            candidate.write_bytes(b"fixture-candidates")
            dump(source / "source_manifest.json", {**old_source, "solve/core/episode.py": "different"})
            with self.assertRaisesRegex(ValueError, "different solver or learner code"):
                sweep.reuse_prefix(output, source)
            dump(source / "source_manifest.json", old_source)
            sweep.reuse_prefix(output, source)
            self.assertEqual(read(output / "protocol.json")["reused_prefix"]["cases"], 2)
            self.assertEqual(file_hash(output / "prefix/trajectory.jsonl"), result["trajectory_sha256"])
            self.assertEqual(checkpoint_payload_hash(output / "prefix/final_bandit.npz", exclude=("metadata",)), pairing["prefix_bandit_sha256"])


class FiveMethodReportTests(unittest.TestCase):
    def test_report_and_figure_include_default_linucb_and_all_three_betas(self):
        helper = SweepRecoveryWiringTests("test_input_1000_is_setup_only_and_1001_activates_rl")
        with tempfile.TemporaryDirectory() as temporary, patch.object(sweep, "CASES", 6), patch.object(sweep, "PREFIX", 2):
            output = Path(temporary)
            prefix_runtime = helper.runtime()
            prefix_runtime.method.solve_activation_case = 2
            prefix = [helper.simulate(prefix_runtime, index, [])[0] for index in range(2)]
            checkpoint = output / "prefix/final_bandit.npz"
            prefix_runtime.branch.policy.model.save_mutable_state(checkpoint)
            pairing = {key: "same" for key in ("stream_sha256", "candidate_sha256", "initial_bandit_sha256",
                "initial_controller_sha256", "prefix_bandit_sha256", "native_sha256")}
            dump(output / "prefix/result.json", dict(pairing=pairing))
            dump(output / "protocol.json", dict(betas=sweep.BETAS, baselines=sweep.BASELINES, reference_branch="beta_2", family="rs_convection_3d_type2"))
            for name in [*sweep.BETAS, *sweep.BASELINES]:
                runtime = helper.runtime()
                runtime.name = name
                runtime.method.solve_activation_case = 2
                runtime.stream = [runtime.stream[0]]*6
                runtime.branch.policy.model.load_mutable_state(checkpoint)
                if name in sweep.BETAS:
                    runtime.bundle.controller.spec = replace(runtime.bundle.controller.spec, uncertainty_beta=sweep.BETAS[name])
                else:
                    runtime.bundle = None
                if name == "default":
                    rows = []
                    for index in range(6):
                        status = "nonconvergence" if index == 5 else "success"
                        with patch("experiments.paper_final.online.native_evaluation.solve_no_rl_case", return_value=outcome(status)):
                            rows.append(sweep.execute_default_case(runtime, index))
                    state = dict(pairing, effective_beta=None, controller_steps=0, candidate_cursor=None,
                        candidate_sha256=None, initial_bandit_sha256=None, initial_controller_sha256=None, prefix_bandit_sha256=None)
                else:
                    rows = [dict(row, method=name) for row in prefix]
                    for index in range(2, 6):
                        rows.append(helper.simulate(runtime, index, ["success"])[0])
                    state = dict(pairing, effective_beta=sweep.BETAS.get(name), controller_steps=0, candidate_cursor=6)
                    if name == "linucb":
                        state["initial_controller_sha256"] = None
                directory = output / name
                directory.mkdir()
                trajectory = directory / "trajectory.jsonl"
                trajectory.write_text("".join(json.dumps(ready(row))+"\n" for row in rows))
                dump(directory / "ready.json", state)
                dump(directory / "result.json", dict(complete=True, trajectory_sha256=file_hash(trajectory),
                    bandit_observations=0 if name == "default" else runtime.branch.policy.model.t,
                    controller=dict(steps=runtime.bundle.controller.steps, episodes=runtime.bundle.controller.episodes)
                    if runtime.bundle is not None else dict(steps=0, episodes=0)))
            sweep.analyze(output)
            comparison = read(output / "comparison.json")
            self.assertEqual({r["branch"] for r in comparison["rows"]}, set(sweep.BETAS) | set(sweep.BASELINES))
            self.assertEqual(comparison["common_success_cases"], 5)
            totals = {r["branch"]:r for r in comparison["rows"] if r["window"] == "all_5000"}
            self.assertEqual(totals["default"]["failures"], 1)
            self.assertEqual(totals["linucb"]["rl_actions"], 0)
            self.assertEqual(totals["beta_2"]["rl_actions"], 4)
            self.assertGreater((output / "comparison.png").stat().st_size, 1000)
            self.assertGreater((output / "comparison.pdf").stat().st_size, 1000)
if __name__ == "__main__":
    unittest.main()
