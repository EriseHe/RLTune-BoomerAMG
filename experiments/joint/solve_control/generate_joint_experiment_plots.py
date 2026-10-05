from __future__ import annotations

"""Plot-only entry point for a completed joint AMG experiment.

This module reads existing result artifacts and writes visualization artifacts.
It never constructs learners, generates problem instances, or runs a solver.
"""

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from experiments.joint.solve_control.online_td_experiment_common import _json_ready, _write_json
from experiments.joint.solve_control.joint_experiment_plotting import generate_plots


def generate_experiment_plots(
    *,
    result_dir: Path,
    rolling_window: int = 100,
    analysis_stop: int | None = None,
) -> Dict[str, Any]:
    """Generate and index plots from one completed result directory."""

    output = generate_plots(
        result_dir=Path(result_dir),
        rolling_window=int(rolling_window),
        analysis_stop=analysis_stop,
    )
    _write_json(
        Path(result_dir) / "high_level_plot_summary.json",
        dict(output),
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots from an existing joint experiment result."
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--rolling-window", type=int, default=100)
    parser.add_argument("--analysis-stop", type=int, default=None)
    args = parser.parse_args()

    output = generate_experiment_plots(
        result_dir=args.result_dir,
        rolling_window=args.rolling_window,
        analysis_stop=args.analysis_stop,
    )
    print(
        json.dumps(
            _json_ready(
                {
                    "stage": "joint_experiment_plots_complete",
                    "result_dir": args.result_dir,
                    "plots": output,
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
