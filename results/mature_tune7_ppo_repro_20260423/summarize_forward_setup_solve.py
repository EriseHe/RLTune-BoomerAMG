from __future__ import annotations

import json
import sys
from pathlib import Path


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python summarize_forward_setup_solve.py <run_dir>", file=sys.stderr)
        return 2

    run_dir = Path(sys.argv[1]).resolve()
    files = sorted(run_dir.glob("forward_dual_seed*.json"))
    if not files:
        print(f"no forward_dual_seed*.json files found in {run_dir}", file=sys.stderr)
        return 1

    windows = {}
    for path in files:
        obj = json.loads(path.read_text())
        for window_name, window in obj["windows"].items():
            out = windows.setdefault(window_name, {})
            for method_name, method in window["methods"].items():
                bucket = out.setdefault(
                    method_name,
                    {"setup": [], "solve": [], "runtime": [], "iterations": [], "failed": []},
                )
                bucket["setup"].append(float(method["mean_setup_runtime"]))
                bucket["solve"].append(float(method["mean_solve_runtime"]))
                bucket["runtime"].append(float(method["mean_runtime"]))
                bucket["iterations"].append(float(method.get("mean_iterations", float("nan"))))
                bucket["failed"].append(float(method.get("failed_count", 0.0)))

    print(f"run_dir: {run_dir}")
    for window_name in sorted(windows):
        print()
        print(f"[{window_name}]")
        for method_name in sorted(windows[window_name]):
            bucket = windows[window_name][method_name]
            print(
                json.dumps(
                    {
                        "method": method_name,
                        "mean_setup_runtime": _mean(bucket["setup"]),
                        "mean_solve_runtime": _mean(bucket["solve"]),
                        "mean_runtime": _mean(bucket["runtime"]),
                        "mean_iterations": _mean(bucket["iterations"]),
                        "mean_failed_count": _mean(bucket["failed"]),
                    },
                    sort_keys=True,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
