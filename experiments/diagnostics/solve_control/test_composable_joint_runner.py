from __future__ import annotations

import _project_paths  # noqa: F401

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import run_joint_experiment as high_level
import run_joint_online_sarsa_4k as runner
import plot_joint_online_sarsa_4k as plotter
from joint_online_common import _problem_stream_spec
from setup_action_space import SetupConfigurationSpace
from setup_aware_compare_common import DEFAULT_SETUP_PARAMS


class ComposableJointRunnerTests(unittest.TestCase):
    def _parse(self, *extra: str):
        return runner.build_parser().parse_args(
            [
                "--output-dir",
                "/tmp/composable-joint-runner-test",
                "--study-mode",
                "composable",
                "--weights",
                "1.0,1.05,1.10",
                "--action-rbf-centers",
                "1.0,1.25,1.5",
                *extra,
            ]
        )

    def test_composable_method_parser_maps_requested_families(self) -> None:
        default = runner._parse_composable_method(
            "default_setup:default:default"
        )
        bandit_default = runner._parse_composable_method(
            "bandit_default:linucb:default"
        )
        default_v2 = runner._parse_composable_method(
            "default_setup_recursive_lstdq_v2_lcb:default:recursive_lstdq_v2"
        )
        bandit_v2 = runner._parse_composable_method(
            "bandit_recursive_lstdq_v2_lcb:linucb:recursive_lstdq_v2"
        )
        fixed = runner._parse_composable_method(
            "bandit_fixed_w1.6:linucb:fixed@1.6"
        )

        self.assertEqual(default.family, "default_setup")
        self.assertEqual(bandit_default.family, "default")
        self.assertEqual(
            default_v2.family,
            "default_setup_recursive_lstdq_v2_lcb",
        )
        self.assertEqual(bandit_v2.family, "recursive_lstdq_v2_lcb")
        self.assertEqual(fixed.family, "fixed_w1.6")
        self.assertAlmostEqual(float(fixed.fixed_weight), 1.6)
        self.assertEqual(
            default_v2.label,
            "Default setup + Recursive LSTDQ v2-LCB",
        )

        named_space = runner._parse_composable_method(
            "linucb_recommended:linucb@recommended:default"
        )
        self.assertEqual(named_space.setup_space, "recommended")
        self.assertEqual(named_space.candidate_sampling, "uniform512")
        self.assertEqual(
            named_space.label,
            "Online LinUCB [recommended; uniform-512] + default solve",
        )
        structured = runner._parse_composable_method(
            "linucb_structured:linucb@recommended@structured512:default"
        )
        self.assertEqual(structured.candidate_sampling, "structured512")
        self.assertEqual(
            structured.label,
            "Online LinUCB [recommended; structured-512] + default solve",
        )
        lin_ts = runner._parse_composable_method(
            "lints_original:lints@original:default"
        )
        self.assertEqual(lin_ts.family, "default")
        self.assertEqual(lin_ts.setup_space, "original")
        self.assertEqual(
            lin_ts.label,
            "Online LinTS v2 [original; uniform-512] + default solve",
        )
        rblspi = runner._parse_composable_method(
            "lints_rblspi:lints@recommended@structured512:rblspi"
        )
        self.assertEqual(rblspi.family, "rblspi")
        self.assertEqual(rblspi.solve_kind, "rblspi")

    def test_composable_method_parser_rejects_ambiguous_or_unsafe_specs(self) -> None:
        invalid_specs = (
            "missing-fields",
            "name:unknown:default",
            "name:linucb:unknown",
            "name:linucb:fixed",
            "name:linucb:fixed@0",
            "name:linucb:fixed@nan",
            "name:linucb@recommended@unknown:default",
            "unsafe/name:linucb:default",
        )
        for raw in invalid_specs:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                runner._parse_composable_method(raw)

    def test_validation_preserves_method_order_and_requires_unique_names(self) -> None:
        args = self._parse(
            "--method",
            "default_setup:default:default",
            "--method",
            "bandit_default:linucb:default",
            "--method",
            "bandit_fixed_w1.6:linucb:fixed@1.6",
        )
        specs = runner._validate_composable_protocol(args)
        self.assertEqual(
            tuple(spec.name for spec in specs),
            ("default_setup", "bandit_default", "bandit_fixed_w1.6"),
        )

        args.method_specs.append("bandit_default:linucb:recursive_lstdq_v2")
        with self.assertRaisesRegex(ValueError, "unique"):
            runner._validate_composable_protocol(args)

    def test_validation_requires_named_setup_space_references(self) -> None:
        args = self._parse(
            "--method",
            "default_setup:default:default",
            "--method",
            "linucb_original:linucb@original:default",
        )
        args.setup_configuration_spaces = {
            "original": SetupConfigurationSpace(
                name="original",
                coarsen_types=(0, 2, 6, 8, 10),
                interp_types=(6, 8),
            )
        }
        specs = runner._validate_composable_protocol(args)
        self.assertEqual(specs[1].setup_space, "original")

        args.method_specs[1] = "linucb_original:linucb@missing:default"
        with self.assertRaisesRegex(ValueError, "unknown setup spaces"):
            runner._validate_composable_protocol(args)

    def test_validation_requires_explicit_finite_increasing_action_bases(self) -> None:
        args = self._parse("--method", "candidate:linucb:recursive_lstdq_v2")
        for attribute, value in (
            ("weights", None),
            ("action_rbf_centers", None),
            ("weights", "1.0,1.0,1.1"),
            ("weights", "1.0,nan,1.1"),
            ("action_rbf_centers", "1.0,0.5"),
        ):
            candidate = self._parse(
                "--method",
                "candidate:linucb:recursive_lstdq_v2",
            )
            setattr(candidate, attribute, value)
            with self.subTest(attribute=attribute, value=value):
                with self.assertRaises(ValueError):
                    runner._validate_composable_protocol(candidate)

        args.action_rbf_sigma = 0.0
        with self.assertRaisesRegex(ValueError, "positive"):
            runner._validate_composable_protocol(args)

    def test_shared_parser_exposes_problem_grid_hash_and_composition(self) -> None:
        args = self._parse(
            "--problem",
            "diffusion_convection",
            "--grid-shape",
            "40,48,56",
            "--advection",
            "1.0,-2.0,0.5",
            "--expected-stream-hash",
            "abc123",
            "--method",
            "default_setup:default:default",
            "--method",
            "bandit_rl:linucb:recursive_lstdq_v2",
        )

        self.assertEqual(args.problem, "diffusion_convection")
        self.assertEqual(args.grid_shape, "40,48,56")
        self.assertEqual(args.advection, "1.0,-2.0,0.5")
        self.assertEqual(args.expected_stream_hash, "abc123")
        self.assertEqual(
            args.method_specs,
            [
                "default_setup:default:default",
                "bandit_rl:linucb:recursive_lstdq_v2",
            ],
        )
        problem, grid, advection = _problem_stream_spec(args)
        self.assertEqual(problem, "diffusion_convection")
        self.assertEqual(grid, (40, 48, 56))
        self.assertEqual(advection, (1.0, -2.0, 0.5))

    def test_scalar_diffusion_rejects_nonzero_advection(self) -> None:
        args = self._parse(
            "--problem",
            "scalar_anisotropic_diffusion",
            "--advection",
            "1,0,0",
            "--method",
            "default_setup:default:default",
        )
        with self.assertRaisesRegex(ValueError, "zero advection"):
            _problem_stream_spec(args)

    def test_default_setup_v2_routes_to_its_controller_without_linucb(self) -> None:
        spec = runner._parse_composable_method(
            "default_setup_recursive_lstdq_v2_lcb:default:recursive_lstdq_v2"
        )
        solver = Mock(
            return_value={
                "runtime": 0.2,
                "solve_runtime": 0.1,
                "infer_runtime": 0.01,
            }
        )
        args = self._parse("--method", f"{spec.name}:default:{spec.solve_kind}")

        with patch.object(
            runner,
            "_report_online_outcome",
            side_effect=lambda outcome, *, bandit_timing: dict(outcome),
        ):
            outcome = runner._run_default_setup_method(
                spec=spec,
                solver_fn=solver,
                args=args,
                mkw={"nx": 40, "ny": 40, "nz": 40},
                controller_methods=(spec.name,),
            )

        solver.assert_called_once_with(dict(DEFAULT_SETUP_PARAMS))
        self.assertFalse(outcome["bandit_update_committed"])

    def test_composable_ppo_path_is_checked_only_when_ppo_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.zip"
            args = self._parse(
                "--ppo-model",
                str(missing),
                "--method",
                "bandit_rl:linucb:recursive_lstdq_v2",
            )
            runner._validate_composable_protocol(args)

            args.method_specs = ["bandit_ppo:linucb:ppo"]
            with self.assertRaises(FileNotFoundError):
                runner._validate_composable_protocol(args)

    def test_high_level_decimal_grid_is_inclusive_and_exact(self) -> None:
        self.assertEqual(
            high_level._expand_grid(
                {"start": 1.0, "stop": 1.2, "step": 0.05},
                name="grid",
            ),
            (1.0, 1.05, 1.1, 1.15, 1.2),
        )
        with self.assertRaisesRegex(ValueError, "exactly"):
            high_level._expand_grid(
                {"start": 1.0, "stop": 1.2, "step": 0.07},
                name="grid",
            )

    def test_runtime_breakdown_sorts_by_native_total_descending(self) -> None:
        window = {
            "bandit_default": {
                "means_sec": {
                    "setup_runtime": 0.03,
                    "native_solve_runtime": 0.04,
                },
            },
            "default_setup": {
                "means_sec": {
                    "setup_runtime": 0.08,
                    "native_solve_runtime": 0.03,
                },
            },
        }
        self.assertEqual(
            plotter._methods_by_native_runtime(
                ("bandit_default", "default_setup"),
                window,
            ),
            ("default_setup", "bandit_default"),
        )

    def test_setup_space_reporting_does_not_require_fixed_solve_branch(self) -> None:
        methods = ("default_setup", "linucb_original")
        records = {method: [] for method in methods}
        self.assertEqual(
            plotter._same_setup_audit(records, methods, expected_cases=4000),
            {},
        )
        self.assertEqual(
            plotter._summary_reference_method(
                [{"method": method} for method in methods],
                methods,
            ),
            "default_setup",
        )
        self.assertEqual(
            plotter._method_label(
                "linucb_original",
                {"linucb_original": "default"},
                {"linucb_original": "Online LinUCB [original] + default solve"},
            ),
            "Online LinUCB [original] + default solve",
        )

    def test_canonical_n40_high_level_config_resolves_protocol(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[2]
            / "joint"
            / "solve_control"
            / "configs"
            / "n40_lstdq_v2_factorial_joint4k.json"
        )
        args = high_level.config_to_args(
            json.loads(config_path.read_text(encoding="utf-8"))
        )
        specs = runner._validate_composable_protocol(args)
        self.assertEqual(args.grid_shape, "40,40,40")
        self.assertEqual(args.online_cases, 4000)
        self.assertEqual(len(args.weights.split(",")), 41)
        self.assertEqual(len(args.action_rbf_centers.split(",")), 9)
        self.assertEqual(
            tuple(spec.name for spec in specs),
            (
                "default_setup",
                "bandit_default",
                "default_setup_recursive_lstdq_v2_lcb",
                "bandit_recursive_lstdq_v2_lcb",
                "bandit_fixed_w1.6",
            ),
        )

    def test_n40_setup_space_comparison_uses_default_solve_only(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[2]
            / "joint"
            / "solve_control"
            / "configs"
            / "n40_setup_space_compare_joint4k.json"
        )
        args = high_level.config_to_args(
            json.loads(config_path.read_text(encoding="utf-8"))
        )
        specs = runner._validate_composable_protocol(args)

        self.assertEqual(args.grid_shape, "40,40,40")
        self.assertEqual(args.online_cases, 4000)
        self.assertEqual(
            tuple(spec.name for spec in specs),
            (
                "default_setup",
                "linucb_original",
                "linucb_recommended",
                "linucb_expanded",
            ),
        )
        self.assertEqual({spec.solve_kind for spec in specs}, {"default"})
        self.assertEqual(
            {
                name: space.as_dict()
                for name, space in args.setup_configuration_spaces.items()
            },
            {
                "original": {
                    "coarsen_types": [0, 2, 6, 8, 10],
                    "interp_types": [6, 8],
                    "agg_interp_types": [4],
                },
                "recommended": {
                    "coarsen_types": [0, 3, 6, 8, 10],
                    "interp_types": [0, 2, 3, 6, 8, 17],
                    "agg_interp_types": [4],
                },
                "expanded": {
                    "coarsen_types": [0, 3, 6, 8, 10, 21, 22],
                    "interp_types": [
                        0,
                        2,
                        3,
                        4,
                        6,
                        7,
                        8,
                        12,
                        13,
                        14,
                        16,
                        17,
                        18,
                    ],
                    "agg_interp_types": [1, 2, 3, 4, 5, 6, 7, 8],
                },
            },
        )

    def test_n40_linucb_vs_lints_v2_is_paired_on_original_space(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[2]
            / "joint"
            / "solve_control"
            / "configs"
            / "n40_linucb_vs_lints_v2_joint4k.json"
        )
        args = high_level.config_to_args(
            json.loads(config_path.read_text(encoding="utf-8"))
        )
        specs = runner._validate_composable_protocol(args)

        self.assertEqual(args.grid_shape, "40,40,40")
        self.assertEqual(args.online_cases, 4000)
        self.assertEqual(
            tuple((spec.name, spec.setup_kind, spec.setup_space) for spec in specs),
            (
                ("linucb_original", "linucb", "original"),
                ("lints_v2_original", "lints", "original"),
            ),
        )
        self.assertEqual({spec.solve_kind for spec in specs}, {"default"})
        self.assertAlmostEqual(args.lin_ts_relative_sampling_scale, 0.15)
        self.assertAlmostEqual(args.lin_ts_loss_scale_prior, 0.1)

    def test_n40_recommended_candidate_oracle_config_resolves_four_branches(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[2]
            / "joint"
            / "solve_control"
            / "configs"
            / "n40_recommended_candidate_oracles_joint4k.json"
        )
        args = high_level.config_to_args(
            json.loads(config_path.read_text(encoding="utf-8"))
        )
        specs = runner._validate_composable_protocol(args)
        self.assertEqual(
            tuple(
                (spec.setup_kind, spec.candidate_sampling, spec.solve_kind)
                for spec in specs
            ),
            (
                ("linucb", "uniform512", "default"),
                ("linucb", "structured512", "default"),
                ("lints", "uniform512", "default"),
                ("lints", "structured512", "default"),
            ),
        )

    def test_n40_rblspi_config_uses_structured_recommended_space(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[2]
            / "joint"
            / "solve_control"
            / "configs"
            / "n40_recommended_lstdq_ucb_vs_rblspi_joint4k.json"
        )
        args = high_level.config_to_args(
            json.loads(config_path.read_text(encoding="utf-8"))
        )
        specs = runner._validate_composable_protocol(args)
        self.assertEqual(
            tuple(
                (
                    spec.setup_kind,
                    spec.setup_space,
                    spec.candidate_sampling,
                    spec.solve_kind,
                )
                for spec in specs
            ),
            (
                ("linucb", "recommended", "structured512", "recursive_lstdq_v2"),
                ("lints", "recommended", "structured512", "rblspi"),
            ),
        )
        self.assertAlmostEqual(args.rblspi_prior_precision, 1.0e4)
        self.assertAlmostEqual(args.rblspi_noise_precision, 1.0e6)
        self.assertAlmostEqual(args.rblspi_gram_ridge, 1.0e-6)

    def test_high_level_reproduction_script_has_valid_continuation_args(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "result"
            output_dir.mkdir()
            args = Mock(
                output_dir=output_dir,
                train_cases=3,
                warmup_cases=0,
                online_cases=3,
            )
            validation = {
                "stream": {
                    "sha256": "test-hash",
                    "problem": "scalar_anisotropic_diffusion",
                    "grid": [4, 4, 4],
                },
                "methods": [
                    {
                        "id": "default_setup",
                        "setup": "default",
                        "solve": "default",
                    }
                ],
            }

            high_level._write_reproduction_artifacts(
                config={"schema_version": 1, "output_dir": str(output_dir)},
                config_path=Path("config.json"),
                args=args,
                validation=validation,
                plot_summary=None,
            )

            script = (output_dir / "reproduce.sh").read_text(encoding="utf-8")
            self.assertIn('\n  --config "', script)
            self.assertIn('\n  --output-dir "$OUTPUT_DIR"', script)
            self.assertNotIn("\n+  --", script)


if __name__ == "__main__":
    unittest.main()
