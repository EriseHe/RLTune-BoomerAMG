"""Profile repeated HYPRE solves with scheduler/resource counters.

This diagnostic replays one exact ``(matrix, setup arm)`` pair from a joint
trajectory.  Native HYPRE timings are paired with process CPU time and
``getrusage`` deltas so wall-clock stalls can be separated from extra compute.
"""

from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np

from hypre.bindings import create_env
from experiments.joint.solve_control.online_td_experiment_common import _json_ready, _write_json


_RESOURCE_FIELDS = (
    "ru_utime",
    "ru_stime",
    "ru_minflt",
    "ru_majflt",
    "ru_nvcsw",
    "ru_nivcsw",
)

_PTHREAD = ctypes.CDLL("/usr/lib/system/libsystem_pthread.dylib")
_PTHREAD.pthread_cpu_number_np.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
_PTHREAD.pthread_cpu_number_np.restype = ctypes.c_int
_PTHREAD.qos_class_self.argtypes = []
_PTHREAD.qos_class_self.restype = ctypes.c_uint
_PTHREAD.pthread_set_qos_class_self_np.argtypes = [ctypes.c_uint, ctypes.c_int]
_PTHREAD.pthread_set_qos_class_self_np.restype = ctypes.c_int

_QOS_CLASSES = {
    "background": 0x09,
    "utility": 0x11,
    "user-initiated": 0x19,
    "user-interactive": 0x21,
}


class _RusageInfoV4(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        *(
            (name, ctypes.c_uint64)
            for name in (
                "ri_user_time",
                "ri_system_time",
                "ri_pkg_idle_wkups",
                "ri_interrupt_wkups",
                "ri_pageins",
                "ri_wired_size",
                "ri_resident_size",
                "ri_phys_footprint",
                "ri_proc_start_abstime",
                "ri_proc_exit_abstime",
                "ri_child_user_time",
                "ri_child_system_time",
                "ri_child_pkg_idle_wkups",
                "ri_child_interrupt_wkups",
                "ri_child_pageins",
                "ri_child_elapsed_abstime",
                "ri_diskio_bytesread",
                "ri_diskio_byteswritten",
                "ri_cpu_time_qos_default",
                "ri_cpu_time_qos_maintenance",
                "ri_cpu_time_qos_background",
                "ri_cpu_time_qos_utility",
                "ri_cpu_time_qos_legacy",
                "ri_cpu_time_qos_user_initiated",
                "ri_cpu_time_qos_user_interactive",
                "ri_billed_system_time",
                "ri_serviced_system_time",
                "ri_logical_writes",
                "ri_lifetime_max_phys_footprint",
                "ri_instructions",
                "ri_cycles",
                "ri_billed_energy",
                "ri_serviced_energy",
                "ri_interval_max_phys_footprint",
                "ri_runnable_time",
            )
        ),
    ]


_LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
_LIBPROC.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_LIBPROC.proc_pid_rusage.restype = ctypes.c_int
_RUSAGE_INFO_V4 = 4
_PROC_COUNTER_FIELDS = (
    "ri_user_time",
    "ri_system_time",
    "ri_pageins",
    "ri_cpu_time_qos_default",
    "ri_cpu_time_qos_background",
    "ri_cpu_time_qos_utility",
    "ri_cpu_time_qos_user_initiated",
    "ri_cpu_time_qos_user_interactive",
    "ri_instructions",
    "ri_cycles",
    "ri_billed_energy",
    "ri_serviced_energy",
    "ri_runnable_time",
)


def _cpu_number() -> int:
    cpu = ctypes.c_size_t()
    error = int(_PTHREAD.pthread_cpu_number_np(ctypes.byref(cpu)))
    return -error if error else int(cpu.value)


def _set_qos_class(name: str) -> None:
    if name == "unchanged":
        return
    error = int(
        _PTHREAD.pthread_set_qos_class_self_np(_QOS_CLASSES[name], 0)
    )
    if error:
        raise OSError(error, f"failed to set QoS class {name!r}")


def _proc_counters() -> dict[str, int]:
    usage = _RusageInfoV4()
    if _LIBPROC.proc_pid_rusage(
        os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(usage)
    ) != 0:
        error = ctypes.get_errno() or errno.EIO
        raise OSError(error, "proc_pid_rusage(RUSAGE_INFO_V4) failed")
    return {field: int(getattr(usage, field)) for field in _PROC_COUNTER_FIELDS}


def _usage() -> dict[str, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        field: float(getattr(usage, field))
        for field in _RESOURCE_FIELDS
    }


def _delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {key: float(after[key] - before[key]) for key in before}


def _trace_row(path: Path, index: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                return json.loads(line)
    raise IndexError(f"trace contains no row {index}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--row", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--settle-sec", type=float, default=0.0)
    parser.add_argument(
        "--qos-class",
        choices=("unchanged", *_QOS_CLASSES),
        default="unchanged",
    )
    args = parser.parse_args()
    if args.row < 0 or args.repeats <= 0 or args.settle_sec < 0.0:
        raise ValueError("row/settle-sec must be non-negative and repeats positive")

    _set_qos_class(args.qos_class)
    source = _trace_row(args.trace.resolve(), args.row)
    records: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        if args.settle_sec:
            time.sleep(args.settle_sec)
        load_before = os.getloadavg()
        create_usage_before = _usage()
        create_cpu_before = time.process_time()
        create_wall_before = time.perf_counter()
        env = create_env(**source["mkw"])
        create_wall_sec = time.perf_counter() - create_wall_before
        create_cpu_sec = time.process_time() - create_cpu_before
        create_usage = _delta(_usage(), create_usage_before)
        try:
            usage_before = _usage()
            proc_counters_before = _proc_counters()
            cpu_before = time.process_time()
            hardware_cpu_before = _cpu_number()
            wall_before = time.perf_counter()
            result = env.solve(
                params=source["params"],
                tol=float(args.tol),
                max_iter=int(args.max_iter),
            )
            outer_wall_sec = time.perf_counter() - wall_before
            hardware_cpu_after = _cpu_number()
            process_cpu_sec = time.process_time() - cpu_before
            proc_counter_delta = _delta(
                _proc_counters(), proc_counters_before
            )
            usage_delta = _delta(_usage(), usage_before)
        finally:
            env.close()
        record = {
            "repeat": repeat + 1,
            "load_average_before": list(load_before),
            "create_wall_sec": create_wall_sec,
            "create_process_cpu_sec": create_cpu_sec,
            "create_resource_delta": create_usage,
            "outer_wall_sec": outer_wall_sec,
            "process_cpu_sec": process_cpu_sec,
            "hardware_cpu_before": hardware_cpu_before,
            "hardware_cpu_after": hardware_cpu_after,
            "qos_class": int(_PTHREAD.qos_class_self()),
            "wall_minus_cpu_sec": outer_wall_sec - process_cpu_sec,
            "native_setup_sec": result.setup_runtime_sec,
            "native_solve_sec": result.solve_runtime_sec,
            "native_total_sec": result.runtime_sec,
            "outer_minus_native_sec": outer_wall_sec - result.runtime_sec,
            "resource_delta": usage_delta,
            "proc_counter_delta": proc_counter_delta,
            "iterations": result.iterations,
            "residual_norm": result.residual_norm,
        }
        records.append(record)
        print(json.dumps(_json_ready(record), separators=(",", ":")), flush=True)

    wall = np.asarray([row["outer_wall_sec"] for row in records])
    cpu = np.asarray([row["process_cpu_sec"] for row in records])
    native_solve = np.asarray([row["native_solve_sec"] for row in records])
    wall_minus_cpu = wall - cpu
    median = float(np.median(wall))
    mad = float(np.median(np.abs(wall - median)))
    threshold = median + max(0.050, 8.0 * 1.4826 * mad)
    summary = {
        "trace": str(args.trace.resolve()),
        "row": args.row,
        "arm_index": source.get("arm_index"),
        "params": source["params"],
        "mkw": source["mkw"],
        "repeats": args.repeats,
        "requested_qos_class": args.qos_class,
        "effective_qos_class": int(_PTHREAD.qos_class_self()),
        "outer_wall_median_sec": median,
        "outer_wall_min_sec": float(np.min(wall)),
        "outer_wall_max_sec": float(np.max(wall)),
        "process_cpu_median_sec": float(np.median(cpu)),
        "process_cpu_min_sec": float(np.min(cpu)),
        "process_cpu_max_sec": float(np.max(cpu)),
        "wall_minus_cpu_median_sec": float(np.median(wall_minus_cpu)),
        "wall_minus_cpu_max_sec": float(np.max(wall_minus_cpu)),
        "native_solve_median_sec": float(np.median(native_solve)),
        "native_solve_max_sec": float(np.max(native_solve)),
        "spike_threshold_sec": threshold,
        "spike_repeats": [
            row["repeat"]
            for row in records
            if row["outer_wall_sec"] > threshold
        ],
        "records": records,
    }
    _write_json(args.output.resolve(), summary)
    print(json.dumps(_json_ready({k: v for k, v in summary.items() if k != "records"}), indent=2), flush=True)


if __name__ == "__main__":
    main()
