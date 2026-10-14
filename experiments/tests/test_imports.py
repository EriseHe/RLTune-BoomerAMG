"""Regressions for canonical imports and solver-independent inspection."""

from pathlib import Path
import subprocess
import sys
import unittest


class ImportTests(unittest.TestCase):
    def run_fresh_python(self, program):
        result = subprocess.run(
            [sys.executable, "-c", program],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_experiment_import_does_not_need_search_path_bootstrap(self):
        self.run_fresh_python(
            "import sys\n"
            "from experiments.paper_final.online import run\n"
            "from experiments.paper_final.online import methods\n"
            "assert '_project_paths' not in sys.modules\n"
            "assert 'methods' not in sys.modules\n"
        )

    def test_configuration_imports_do_not_load_native_library(self):
        self.run_fresh_python(
            "from pathlib import Path\n"
            "original = Path.exists\n"
            "Path.exists = lambda p: False if p.name in "
            "{'libamg_runtime.dylib', 'libamg_runtime.so'} else original(p)\n"
            "import setup.registry\n"
            "import solve.registry\n"
            "from hypre.bindings import boomeramg, create_env\n"
            "assert boomeramg._lib is None\n"
            "try:\n"
            "    create_env()\n"
            "except RuntimeError as error:\n"
            "    assert 'Compiled library not found' in str(error)\n"
            "else:\n"
            "    raise AssertionError('Native execution must require the library')\n"
        )
