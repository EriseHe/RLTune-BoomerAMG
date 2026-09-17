"""Stage 4 entry point; reuse the existing formal-suite implementation."""
from experiments.diagnostics.solve_control import _project_paths  # noqa: F401
from analyze_paper_final import main

if __name__ == "__main__":
    main()
