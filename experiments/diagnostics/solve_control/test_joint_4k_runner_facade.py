from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import ast
from dataclasses import fields
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import experiments.joint.solve_control.joint_4k_runner as canonical
import experiments.joint.solve_control.run_joint_online_sarsa_4k as legacy


class Joint4KRunnerFacadeTests(unittest.TestCase):
    def test_core_entry_points_delegate_to_canonical_runner(self) -> None:
        self.assertIs(legacy.build_parser, canonical.build_parser)
        self.assertIs(legacy.run, canonical.run)
        self.assertIs(legacy.main, canonical.main)
        self.assertIs(
            legacy._make_recursive_lstdq_v2_controller,
            canonical._make_recursive_lstdq_v2_controller,
        )
        self.assertIs(legacy._window_result, canonical._window_result)

    def test_default_setup_wrapper_injects_patchable_legacy_reporter(self) -> None:
        reporter = Mock()
        args = SimpleNamespace(tol=1.0e-6, max_cycles=50)
        sentinel = {"ok": True}
        with (
            patch.object(legacy, "_report_online_outcome", reporter),
            patch.object(
                canonical,
                "_run_default_setup_method",
                return_value=sentinel,
            ) as delegate,
        ):
            result = legacy._run_default_setup_method(
                spec=Mock(),
                solver_fn=Mock(),
                args=args,
                mkw={"nx": 40, "ny": 40, "nz": 40},
                controller_methods=("method",),
            )

        self.assertIs(result, sentinel)
        self.assertIs(
            delegate.call_args.kwargs["report_online_outcome"],
            reporter,
        )

    def test_runner_plan_keywords_match_execution_dataclass(self) -> None:
        tree = ast.parse(inspect.getsource(canonical.run))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "OnlineComparisonPlan"
        ]
        self.assertEqual(len(calls), 1)
        keyword_names = {
            keyword.arg
            for keyword in calls[0].keywords
            if keyword.arg is not None
        }
        field_names = {
            field.name
            for field in fields(canonical.OnlineComparisonPlan)
        }
        self.assertEqual(keyword_names - field_names, set())
        self.assertIn("include_solve_screen_report", keyword_names)


if __name__ == "__main__":
    unittest.main()
