"""Check that Make preserves the loader-relative path on Linux."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class NativeBuildTests(unittest.TestCase):
    def test_linux_linker_receives_literal_origin(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            objects = temporary / "objects"
            objects.mkdir()
            for name in ("amg_runtime.o", "amg_cycle.o", "amg_rl_shared.o"):
                (objects / name).touch()
            capture = temporary / "arguments.json"
            compiler = temporary / "compiler.py"
            compiler.write_text(
                "import json, sys\nfrom pathlib import Path\n"
                f"Path({str(capture)!r}).write_text(json.dumps(sys.argv[1:]))\n"
            )
            subprocess.run(
                [
                    "make",
                    "-C",
                    str(root / "hypre/interfaces"),
                    "UNAME_S=Linux",
                    f"BUILD_DIR={objects}",
                    f"RUNTIME_TARGET={temporary / 'libamg_runtime.so'}",
                    f"MPICC={sys.executable} {compiler}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            arguments = json.loads(capture.read_text())
            self.assertIn("-Wl,-rpath,$ORIGIN/../install/lib", arguments)
            self.assertNotIn("Makefile", arguments)


if __name__ == "__main__":
    unittest.main()
