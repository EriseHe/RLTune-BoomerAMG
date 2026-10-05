"""Process configuration must be explicit and portable."""

import importlib
import os
import unittest
from unittest.mock import Mock, patch

from experiments import runtime


class ExperimentRuntimeTests(unittest.TestCase):
    def test_import_does_not_change_thread_environment(self):
        with patch.dict(os.environ, {key: "7" for key in runtime.THREAD_KEYS}):
            before = dict(os.environ)
            importlib.reload(runtime)
            self.assertEqual(dict(os.environ), before)

    def test_worker_environment_is_pinned_without_mutating_parent(self):
        source = {"OMP_NUM_THREADS": "7", "CUSTOM_RUN_SETTING": "preserved"}
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "9"}):
            result = runtime.single_thread_environment(source)
            self.assertEqual(source["OMP_NUM_THREADS"], "7")
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "9")
        self.assertEqual(result["CUSTOM_RUN_SETTING"], "preserved")
        self.assertTrue(all(result[key] == "1" for key in runtime.THREAD_KEYS))

    def test_cli_configuration_sets_every_thread_key(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime.configure_single_thread()
            self.assertTrue(all(os.environ[key] == "1" for key in runtime.THREAD_KEYS))

    def test_non_macos_sleep_protection_does_not_launch_a_process(self):
        with (
            patch.object(runtime.sys, "platform", "linux"),
            patch.object(runtime.subprocess, "Popen") as launch,
        ):
            self.assertIsNone(runtime.prevent_sleep())
            launch.assert_not_called()
        runtime.stop_sleep_prevention(None)

    def test_missing_sleep_helper_does_not_launch_a_process(self):
        with (
            patch.object(runtime.sys, "platform", "darwin"),
            patch.object(runtime.Path, "is_file", return_value=False),
            patch.object(runtime.subprocess, "Popen") as launch,
        ):
            self.assertIsNone(runtime.prevent_sleep())
            launch.assert_not_called()

    def test_sleep_helper_launch_failure_is_optional(self):
        with (
            patch.object(runtime.sys, "platform", "darwin"),
            patch.object(runtime.Path, "is_file", return_value=True),
            patch.object(runtime.os, "access", return_value=True),
            patch.object(
                runtime.subprocess, "Popen", side_effect=OSError("unavailable")
            ),
        ):
            self.assertIsNone(runtime.prevent_sleep())

    def test_sleep_helper_is_reaped(self):
        process = Mock()
        process.poll.return_value = None
        runtime.stop_sleep_prevention(process)
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5)

    def test_sleep_helper_uses_original_macos_command(self):
        with (
            patch.object(runtime.sys, "platform", "darwin"),
            patch.object(runtime.Path, "is_file", return_value=True),
            patch.object(runtime.os, "access", return_value=True),
            patch.object(runtime.subprocess, "Popen") as launch,
        ):
            self.assertIs(runtime.prevent_sleep(), launch.return_value)
            launch.assert_called_once_with(
                ["/usr/bin/caffeinate", "-is", "-w", str(os.getpid())]
            )


if __name__ == "__main__":
    unittest.main()
