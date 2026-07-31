from __future__ import annotations

import _project_paths  # noqa: F401

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import composable_joint_4k as composable
from joint_method_spec import ComposableMethodSpec
from setup.space import SetupConfigurationSpace
from solve.controllers import ppo
from solve.controllers.common import SolveStateSpec
from solve.registry import OnlineControllerBuildSpec


class ComposableJoint4KAssemblyTests(unittest.TestCase):
    def test_frozen_ppo_construction_is_owned_by_solve_family(self) -> None:
        args = SimpleNamespace(
            ppo_model=Path("/tmp/model.zip"),
            output_dir=Path("/tmp/output"),
            grid_n=80,
            c_min=1.0,
            c_max=1000.0,
            max_cycles=50,
            tol=1.0e-6,
            ppo_action_mode="continuous_absolute",
            ppo_w_center=1.5,
            ppo_w_scale=0.5,
            ppo_initial_observation_weight=1.0,
            ppo_force_default_first_action=True,
            ppo_default_first_weight=1.0,
        )
        built = Mock()
        self.assertIs(composable.FrozenPpoConfig, ppo.FrozenPpoConfig)

        with patch.object(
            composable,
            "build_frozen_ppo_runner",
            return_value=built,
        ) as build:
            result = composable.make_frozen_ppo_runner(args)

        self.assertIs(result, built)
        build.assert_called_once()
        config = build.call_args.args[0]
        self.assertIsInstance(config, ppo.FrozenPpoConfig)
        self.assertEqual(config.ppo_model, args.ppo_model)
        self.assertEqual(config.grid_n, 80)
        self.assertTrue(config.ppo_force_default_first_action)

    def test_named_aot_branches_preserve_per_method_schedule_contract(self) -> None:
        space = SetupConfigurationSpace(
            name="recommended",
            coarsen_types=(0, 3, 6, 8, 10),
            interp_types=(0, 2, 3, 6, 8, 17),
        )
        specs = (
            ComposableMethodSpec(
                name="linucb_uniform",
                setup_kind="linucb",
                solve_kind="default",
                setup_space=space.name,
                candidate_sampling="uniform512",
            ),
            ComposableMethodSpec(
                name="lints_structured",
                setup_kind="lints",
                solve_kind="default",
                setup_space=space.name,
                candidate_sampling="structured512",
                seed_offset=100_003,
            ),
        )
        study = composable.ComposableStudy(
            specs=specs,
            specs_by_name={spec.name: spec for spec in specs},
            methods=tuple(spec.name for spec in specs),
            family_by_method={spec.name: spec.family for spec in specs},
            setup_configuration_spaces={space.name: space},
            bandit_methods=tuple(spec.name for spec in specs),
        )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            branches = []
            for _ in specs:
                model = Mock()
                model.save_mutable_state = Mock()
                branches.append(
                    SimpleNamespace(
                        policy=SimpleNamespace(model=model),
                    )
                )
            args = SimpleNamespace(
                output_dir=output_dir,
                reuse_warmup=False,
                setup_candidate_mode="aot",
                bandit_seed=31,
                setup_action_space="full_cartesian",
                tol=1.0e-6,
                max_cycles=50,
                setup_param_resolution=20,
                online_cases=4000,
                aot_max_selections_per_case=3,
                aot_schedule_chunk_rounds=128,
                lin_ts_relative_sampling_scale=0.15,
                lin_ts_loss_scale_prior=0.1,
            )

            with patch.object(
                composable,
                "build_online_linucb_branch",
                side_effect=[(branch, Mock()) for branch in branches],
            ) as build:
                artifacts = composable.build_named_setup_branches(
                    args,
                    study=study,
                    warmup_instances=(),
                    stream_hash="stream-hash",
                )

            self.assertEqual(build.call_count, 2)
            for call, spec in zip(build.call_args_list, specs):
                self.assertEqual(
                    call.kwargs["seed"],
                    31 + spec.seed_offset,
                )
                self.assertEqual(
                    call.kwargs["learner_kind"],
                    spec.setup_kind,
                )
                self.assertEqual(
                    call.kwargs["candidate_sampling"],
                    spec.candidate_sampling,
                )
                self.assertEqual(
                    call.kwargs["candidate_schedule_rounds"],
                    12000,
                )
                expected_schedule_dir = (
                    output_dir
                    / "aot_candidate_schedules"
                    / space.name
                    / spec.candidate_sampling
                )
                if spec.seed_offset:
                    expected_schedule_dir /= (
                        f"seed_offset_{spec.seed_offset}"
                    )
                self.assertEqual(
                    call.kwargs["candidate_schedule_dir"],
                    expected_schedule_dir,
                )
            self.assertTrue(artifacts.aot_enabled)
            self.assertEqual(tuple(artifacts.branches), study.methods)
            self.assertEqual(
                artifacts.warmup_state,
                {
                    spec.name: str(
                        output_dir / "bandit_warmup_states" / f"{spec.name}.npz"
                    )
                    for spec in specs
                },
            )
            self.assertTrue(
                (output_dir / "warmup_trajectory.jsonl").exists()
            )
            self.assertTrue((output_dir / "warmup_progress.json").exists())

    def test_named_branch_uses_method_specific_setup_context_contract(self) -> None:
        space = SetupConfigurationSpace(
            name="recommended",
            coarsen_types=(0, 2, 3, 6, 8, 10),
            interp_types=(0, 2, 3, 6, 8, 17),
        )
        specs = (
            ComposableMethodSpec(
                name="context7",
                setup_kind="linucb",
                setup_space=space.name,
                setup_context="canonical_no_c_mean",
                candidate_sampling="structured512",
                solve_kind="default",
            ),
            ComposableMethodSpec(
                name="context8",
                setup_kind="linucb_v5",
                setup_space=space.name,
                candidate_sampling="structured512",
                solve_kind="default",
            ),
        )
        study = composable.ComposableStudy(
            specs=specs,
            specs_by_name={spec.name: spec for spec in specs},
            methods=tuple(spec.name for spec in specs),
            family_by_method={spec.name: spec.family for spec in specs},
            setup_configuration_spaces={space.name: space},
            bandit_methods=tuple(spec.name for spec in specs),
        )
        args = SimpleNamespace(
            output_dir=None,
            reuse_warmup=False,
            setup_candidate_mode="aot",
            bandit_seed=31,
            problem="scalar_anisotropic_diffusion_advection",
            setup_action_space="full_cartesian",
            tol=1.0e-6,
            max_cycles=50,
            setup_param_resolution=20,
            online_cases=4,
            aot_max_selections_per_case=3,
            aot_schedule_chunk_rounds=8,
            lin_ts_relative_sampling_scale=0.15,
            lin_ts_loss_scale_prior=0.1,
        )

        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = Path(directory)
            branches = [
                SimpleNamespace(
                    policy=SimpleNamespace(
                        model=Mock(
                            A_inv=object(),
                            actions=object(),
                        )
                    )
                )
                for _spec in specs
            ]
            for branch in branches:
                branch.policy.model.save_mutable_state = Mock()
            with patch.object(
                composable,
                "build_online_linucb_branch",
                side_effect=[(branch, Mock()) for branch in branches],
            ) as build:
                composable.build_named_setup_branches(
                    args,
                    study=study,
                    warmup_instances=(),
                    stream_hash="stream-hash",
                )

        calls = build.call_args_list
        self.assertEqual(calls[0].kwargs["context_dim"], 7)
        self.assertEqual(
            calls[0].kwargs["context_interaction_indices"],
            (1, 2, 3, 4, 5, 6),
        )
        self.assertEqual(calls[1].kwargs["context_dim"], 8)
        self.assertEqual(
            calls[1].kwargs["context_interaction_indices"],
            (1, 2, 3, 4, 5, 6, 7),
        )

    def test_named_warmup_depths_share_one_prefix_and_align_evaluation(self) -> None:
        space = SetupConfigurationSpace(
            name="recommended",
            coarsen_types=(8, 10),
            interp_types=(0, 6),
        )
        specs = tuple(
            ComposableMethodSpec(
                name=f"warmup_{depth}",
                setup_kind="linucb",
                solve_kind="default",
                setup_space=space.name,
                candidate_sampling="structured512",
                setup_warmup_cases=depth,
            )
            for depth in (2, 1, 0)
        )
        study = composable.ComposableStudy(
            specs=specs,
            specs_by_name={spec.name: spec for spec in specs},
            methods=tuple(spec.name for spec in specs),
            family_by_method={spec.name: spec.family for spec in specs},
            setup_configuration_spaces={space.name: space},
            bandit_methods=tuple(spec.name for spec in specs),
        )
        args = SimpleNamespace(
            output_dir=None,
            reuse_warmup=False,
            setup_candidate_mode="aot",
            bandit_seed=31,
            setup_action_space="full_cartesian",
            tol=1.0e-6,
            max_cycles=50,
            setup_param_resolution=20,
            online_cases=4,
            aot_max_selections_per_case=3,
            aot_schedule_chunk_rounds=8,
            lin_ts_relative_sampling_scale=0.15,
            lin_ts_loss_scale_prior=0.1,
            progress_every=10,
        )

        with tempfile.TemporaryDirectory() as directory:
            args.output_dir = Path(directory)
            models = [Mock() for _spec in specs]
            branches = [
                SimpleNamespace(
                    policy=SimpleNamespace(model=model),
                    parameter_space={},
                )
                for model in models
            ]
            instances = [
                (
                    {
                        "case": index,
                        "nx": 4,
                        "ny": 4,
                        "nz": 4,
                    },
                    [1.0, 0.1, 0.2, 0.3, 0.2, 0.0, 0.0, 0.0],
                )
                for index in range(2)
            ]
            with (
                patch.object(
                    composable,
                    "build_online_linucb_branch",
                    side_effect=[(branch, Mock()) for branch in branches],
                ) as build,
                patch.object(
                    composable,
                    "run_bandit_step_test_final",
                    return_value=({}, {}, {}, 0, 0.0),
                ) as step,
                patch.object(
                    composable,
                    "_policy_last_arm",
                    return_value=0,
                ),
                patch.object(
                    composable,
                    "_report_online_outcome",
                    return_value={},
                ),
                patch.object(
                    composable,
                    "_method_stream_summary",
                    side_effect=lambda rows: {"count": len(rows)},
                ),
            ):
                artifacts = composable.build_named_setup_branches(
                    args,
                    study=study,
                    warmup_instances=instances,
                    stream_hash="stream-hash",
                )

            self.assertEqual(step.call_count, 2)
            self.assertEqual(build.call_args.kwargs["candidate_schedule_rounds"], 18)
            for model in models:
                model.set_candidate_schedule_cursor.assert_called_once_with(6)
            self.assertIn("group_0_warmup_2.npz", str(models[0].load_mutable_state.call_args.args[0]))
            self.assertIn("group_0_warmup_1.npz", str(models[1].load_mutable_state.call_args.args[0]))
            self.assertIn("group_0_warmup_0.npz", str(models[2].load_mutable_state.call_args.args[0]))
            self.assertEqual(len(artifacts.warmup_rows), 2)
            self.assertEqual(
                artifacts.warmup_summary,
                {
                    "warmup_2": {"count": 2},
                    "warmup_1": {"count": 1},
                    "warmup_0": composable._empty_stream_summary(),
                },
            )

    def test_typed_solve_specs_bypass_compatibility_factories_and_keep_seeds(
        self,
    ) -> None:
        kinds_and_offsets = (
            ("recursive_mc", 0),
            ("recursive_lstdq_v1", 1009),
            ("recursive_lstdq_v2", 2018),
            ("recursive_lstdq_v3", 2018),
            ("rblspi", 5045),
            ("stagewise_lsvi", 2018),
            ("structured_model_based", 3027),
            ("recalibrated_lsvi", 4036),
        )
        specs = tuple(
            ComposableMethodSpec(
                name=kind,
                setup_kind="linucb",
                solve_kind=kind,
                seed_offset=(
                    100_003
                    if kind == "recursive_lstdq_v3"
                    else 0
                ),
            )
            for kind, _offset in kinds_and_offsets
        ) + (
            ComposableMethodSpec(
                name="native",
                setup_kind="linucb",
                solve_kind="default",
            ),
        )
        typed_specs = {
            kind: SimpleNamespace(kind=kind)
            for kind, _offset in kinds_and_offsets
        }
        args = SimpleNamespace(
            controller_seed=101,
            solve_controller_specs=typed_specs,
        )
        bundles = [Mock() for _kind, _offset in kinds_and_offsets]

        with (
            patch.object(
                composable,
                "_COMPATIBILITY_BUNDLE_FACTORIES",
                {},
            ),
            patch.object(
                composable,
                "make_setup_obs_encoder",
                side_effect=[Mock() for _ in kinds_and_offsets],
            ),
            patch.object(
                composable,
                "build_online_solve_controller",
                side_effect=bundles,
            ) as build,
        ):
            runtime = composable.build_composable_solve_runtime(
                args,
                specs=specs,
            )

        self.assertEqual(
            runtime.controller_bundles,
            {
                kind: bundle
                for (kind, _offset), bundle in zip(
                    kinds_and_offsets,
                    bundles,
                )
            },
        )
        self.assertIsNone(runtime.ppo_runner)
        self.assertEqual(build.call_count, len(kinds_and_offsets))
        for call, spec, (kind, offset) in zip(
            build.call_args_list,
            specs,
            kinds_and_offsets,
        ):
            self.assertIs(call.args[0], typed_specs[kind])
            self.assertEqual(
                call.kwargs["seed"],
                101 + offset + spec.seed_offset,
            )

    def test_typed_solve_runtime_rejects_missing_and_unknown_online_kinds(
        self,
    ) -> None:
        args = SimpleNamespace(
            controller_seed=101,
            solve_controller_specs={},
        )
        missing = ComposableMethodSpec(
            name="missing",
            setup_kind="linucb",
            solve_kind="recursive_lstdq_v2",
        )
        with self.assertRaisesRegex(
            ValueError,
            "Missing typed solve-controller spec",
        ):
            composable.build_composable_solve_runtime(
                args,
                specs=(missing,),
            )

        unknown = ComposableMethodSpec(
            name="unknown",
            setup_kind="linucb",
            solve_kind="future_controller",
        )
        with self.assertRaisesRegex(ValueError, "Unknown online solve kind"):
            composable.build_composable_solve_runtime(
                args,
                specs=(unknown,),
            )

    def test_typed_solve_runtime_applies_method_tolerance_to_encoder(self) -> None:
        typed_spec = OnlineControllerBuildSpec(
            kind="recursive_lstdq_v3",
            state=SolveStateSpec(
                tol=1.0e-6,
                max_cycles=50,
                c_max=1000.0,
            ),
            actions=Mock(),
            algorithm=Mock(),
            trace_lambda=0.8,
        )
        method = ComposableMethodSpec(
            name="strict",
            setup_kind="linucb_v5",
            solve_kind="recursive_lstdq_v3",
            solve_tolerance=1.0e-8,
            solve_context="legacy",
        )
        args = SimpleNamespace(
            controller_seed=101,
            solve_controller_specs={
                "recursive_lstdq_v3": typed_spec,
            },
        )
        bundle = Mock()

        with (
            patch.object(
                composable,
                "make_setup_obs_encoder",
                return_value=Mock(),
            ),
            patch.object(
                composable,
                "build_online_solve_controller",
                return_value=bundle,
            ) as build,
        ):
            runtime = composable.build_composable_solve_runtime(
                args,
                specs=(method,),
            )

        resolved_spec = build.call_args.args[0]
        self.assertIs(runtime.controller_bundles[method.name], bundle)
        self.assertIsNot(resolved_spec, typed_spec)
        self.assertEqual(resolved_spec.state.tol, 1.0e-8)
        self.assertEqual(
            resolved_spec.state.problem_context_mode,
            "legacy",
        )
        self.assertEqual(typed_spec.state.tol, 1.0e-6)
        self.assertEqual(
            typed_spec.state.problem_context_mode,
            "canonical",
        )

    def test_typed_solve_runtime_skips_non_controller_backends(self) -> None:
        specs = (
            ComposableMethodSpec(
                name="native",
                setup_kind="linucb",
                solve_kind="default",
            ),
            ComposableMethodSpec(
                name="fixed",
                setup_kind="linucb",
                solve_kind="fixed",
                fixed_weight=1.55,
            ),
            ComposableMethodSpec(
                name="ppo",
                setup_kind="linucb",
                solve_kind="ppo",
            ),
        )
        args = SimpleNamespace(
            controller_seed=101,
            solve_controller_specs={},
        )
        ppo_runner = Mock()

        with patch.object(
            composable,
            "make_frozen_ppo_runner",
            return_value=ppo_runner,
        ) as make_ppo:
            runtime = composable.build_composable_solve_runtime(
                args,
                specs=specs,
            )

        self.assertEqual(runtime.controller_bundles, {})
        self.assertIs(runtime.ppo_runner, ppo_runner)
        make_ppo.assert_called_once_with(args)


if __name__ == "__main__":
    unittest.main()
