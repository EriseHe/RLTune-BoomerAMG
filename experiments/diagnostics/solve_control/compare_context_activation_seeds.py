"""Compare completed context/activation replications without running PDE solves.

Reuse the trajectory summaries from the mechanism audit. Comparisons describe
the recorded adaptive paths; they are not causal activation effects or
confidence intervals over independently sampled training seeds.
"""
from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from experiments.diagnostics.solve_control.analyze_context_activation_mechanism import METHODS, summarize, trajectory_audit
from experiments.joint.solve_control.online_td_experiment_common import _json_ready, _write_json


DEFAULT = "default_setup_default_solve"
LABELS = {
    DEFAULT: "Default",
    "context8d_fixed": "8D fixed",
    "context3d_fixed": "3D fixed",
    "context4d_fixed": "4D fixed",
    "context3d_dynamic": "3D dynamic",
    "context4d_dynamic": "4D dynamic",
}
COMPONENTS = ("setup_runtime", "solve_runtime", "infer_runtime", "bandit_overhead_runtime")


def read_run(path: Path):
    config = json.loads((path / "experiment_config.json").read_text())
    result = json.loads((path / "result.json").read_text())
    manifest = json.loads((path / "stream_manifest.json").read_text())
    assert [m["id"] for m in config["methods"]] == list(LABELS)
    assert manifest["sha256"] == config["stream"]["expected_sha256"]
    assert result["protocol"]["stream"]["sha256"] == manifest["sha256"]
    assert json.loads((path / "progress.json").read_text())["completed_online_instances"] == 5000
    assert '"stage": "joint_experiment_complete"' in path.with_suffix(".log").read_text()

    rows = {}
    for method in LABELS:
        with (path / "trajectories" / f"{method}.jsonl").open() as handle:
            data = [json.loads(line) for line in handle]
        assert len(data) == 5000
        assert all(row["online_index"] == i for i, row in enumerate(data))
        for row in data:
            outcome = row["outcome"]
            total = float(outcome["end_to_end_runtime"])
            assert math.isfinite(total) and total >= 0
            assert math.isclose(total, sum(outcome[k] for k in COMPONENTS), abs_tol=1e-9)
        reported = result["windows"]["all_5000"]["methods"][method]
        assert math.isclose(sum(row["outcome"]["end_to_end_runtime"] for row in data),
                            reported["totals_sec"]["end_to_end_runtime"], abs_tol=1e-8)
        assert reported["unrecovered_failure_count"] == 0
        rows[method] = data

    inputs = [row["mkw"] for row in rows[DEFAULT]]
    assert all([row["mkw"] for row in data] == inputs for data in rows.values())
    audit = trajectory_audit({method: rows[method] for method in METHODS})
    recovery = {}
    for method, data in rows.items():
        recovery[method] = {}
        for name, window in (("all_5000", data), ("after_1000", data[1000:]),
                             ("last_1000", data[4000:])):
            summary = summarize(window)
            recovery[method][name] = {
                "cases": len(window),
                "primary_failures": summary["primary_failures"],
                "fallback_native_sec": sum(row["outcome"]["fallback_setup_runtime"]
                                           + row["outcome"]["fallback_solve_runtime"]
                                           for row in window),
                "failed_problem_e2e_sec": sum(row["outcome"]["end_to_end_runtime"]
                    for row in window if row["outcome"]["first_primary_status"] != "success"),
            }

    for dim in (3, 4):
        fixed, dynamic = (rows[f"context{dim}d_{start}"] for start in ("fixed", "dynamic"))
        pair = audit["pairs"][str(dim)]
        gate = result["rl_activation"][f"context{dim}d_dynamic"]
        assert pair["first_rl_case"] == gate["first_rl_case"]
        pair["common_primary_success"] = {}
        for name, start in (("after_1000", 1000), ("last_1000", 4000)):
            matched = [(f, d) for f, d in zip(fixed[start:], dynamic[start:])
                       if all(row["outcome"]["first_primary_status"] == "success"
                              for row in (f, d))]
            pair["common_primary_success"][name] = {
                "cases": len(matched),
                "dynamic_minus_fixed_sec": sum(d["outcome"]["end_to_end_runtime"]
                                                - f["outcome"]["end_to_end_runtime"]
                                                for f, d in matched),
            }

    return {
        "path": str(path.resolve()), "config": config, "manifest": manifest,
        "windows": result["windows"], "activation": result["rl_activation"],
        "trajectory": audit, "recovery": recovery,
        "inputs": inputs,
        "execution_ranks": {method: [row["execution_rank"] for row in data]
                            for method, data in rows.items()},
        "validation": {"cases_per_method": 5000, "methods": 6,
                       "input_pairing_verified": True, "cost_accounting_verified": True,
                       "completion_marker_verified": True, "unrecovered_failures": 0},
    }


def render_report(runs):
    lines = ["# 60³ context/activation：seed C 与 seed D 比较", "",
             "全部比较来自已完成的轨迹，不执行额外 PDE 求解。时间包含恢复、controller 和 bandit 开销。",
             "", "## 完整性与比较范围", "",
             "两轮各六分支、每分支 5000 题；逐题矩阵/RHS 描述和执行顺序一致，成本分项与 result.json 相符，未恢复失败均为 0。",
             "算法 seed offset 从 2000029 改为 3000047，其他算法及输入配置不变。",
             f"输入流 SHA-256：`{runs['D']['manifest']['sha256']}`。", "",
             "这是同一问题流上的两条算法随机性复现。墙钟时间反馈和机器状态仍能影响学习轨迹，不能把全部差异归因于 RNG，也不能把同一 seed 内 dynamic/fixed 的差距解释成纯启动时刻的因果效应。",
             "", "## 全 5000 题：主要比较", "",
             "累计时间单位为秒；降幅分别相对于各自运行中的 Default。",
             "", "| 方法 | C 累计时间 | C 降幅 | D 累计时间 | D 降幅 |",
             "|---|---:|---:|---:|---:|"]
    for method, label in LABELS.items():
        values = []
        for run in runs.values():
            methods = run["windows"]["all_5000"]["methods"]
            cost = methods[method]["totals_sec"]["end_to_end_runtime"]
            baseline = methods[DEFAULT]["totals_sec"]["end_to_end_runtime"]
            values.extend([f"{cost:.3f}", f"{100*(1-cost/baseline):.2f}%"])
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    lines.extend(["", "## 动态启动与总时间差", "",
                  "正值表示 dynamic 比对应 fixed 更慢。固定分支从第 1001 题启用 RL。分项差值单位为秒。",
                  "", "| Seed | Context | Dynamic 首题 | Setup 差 | Solve 差 | 两类开销差 | E2E 差 | 首次 setup 分岔题 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for seed, run in runs.items():
        windows = run["trajectory"]["windows"]
        for dim in (3, 4):
            fixed = windows[f"context{dim}d_fixed"]["all"]["totals_sec"]
            dynamic = windows[f"context{dim}d_dynamic"]["all"]["totals_sec"]
            delta = {key: dynamic[key]-fixed[key] for key in fixed}
            pair = run["trajectory"]["pairs"][str(dim)]
            lines.append(f"| {seed} | {dim}D | {pair['first_rl_case']} | "
                         f"{delta['setup_runtime']:+.3f} | {delta['solve_runtime']:+.3f} | "
                         f"{delta['infer_runtime']+delta['bandit_overhead_runtime']:+.3f} | "
                         f"{delta['end_to_end_runtime']:+.3f} | {pair['first_setup_divergence_case']} |")

    lines.extend(["", "## 启动前后成本分解", "",
                  "三个互斥区间的差值相加等于全程差值。区间定义由实际 gate 触发点确定，不代表反事实效果。",
                  "", "| Seed | Context | Dynamic 启动前 | Dynamic 已启用、fixed 未启用 | 第 1001–5000 题 |",
                  "|---|---|---:|---:|---:|"])
    for seed, run in runs.items():
        for dim in (3, 4):
            phases = run["trajectory"]["pairs"][str(dim)]["phases"][:3]
            cells = [f"{phase['dynamic_minus_fixed_sec']:+.3f} s（{phase['first']}–{phase['last']}）"
                     for phase in phases]
            lines.append(f"| {seed} | {dim}D | " + " | ".join(cells) + " |")

    lines.extend(["", "## Seed D 最后 1000 题", "", "每题平均耗时，单位 ms。",
                  "", "| 方法 | Setup | Solve | Controller + bandit | E2E | 对 Default 降幅 |",
                  "|---|---:|---:|---:|---:|---:|"])
    window = runs["D"]["windows"]["last_1000"]["methods"]
    baseline = window[DEFAULT]["means_sec"]["end_to_end_runtime"]
    for method, label in LABELS.items():
        m = window[method]["means_sec"]
        lines.append(f"| {label} | {1000*m['setup_runtime']:.3f} | {1000*m['native_solve_runtime']:.3f} | "
                     f"{1000*(m['controller_runtime']+m['setup_bandit_overhead']):.3f} | "
                     f"{1000*m['end_to_end_runtime']:.3f} | {100*(1-m['end_to_end_runtime']/baseline):.2f}% |")

    lines.extend(["", "## Seed D 失败与恢复成本", "",
                  "Fallback native 只包含默认恢复的 setup+solve，不包含此前失败尝试的时间。失败题 E2E 则包含该题所有已计入成本。两者不能相加，否则重复计数。",
                  "", "| 方法 | 全程首次失败 | 第 1001–5000 题首次失败 | 全程 fallback native (s) | 全程失败题 E2E (s) |",
                  "|---|---:|---:|---:|---:|"])
    for method, label in LABELS.items():
        a = runs["D"]["recovery"][method]["all_5000"]
        b = runs["D"]["recovery"][method]["after_1000"]
        lines.append(f"| {label} | {a['primary_failures']} | {b['primary_failures']} | "
                     f"{a['fallback_native_sec']:.3f} | {a['failed_problem_e2e_sec']:.3f} |")
    lines.extend(["", "仅取两分支都首次成功的共同问题，得到以下描述性核对；这是结果条件化的子集，不是另一个无偏性能估计。", "",
                  "| Context | 区间 | 共同首次成功题数 | Dynamic − fixed (s) |", "|---|---|---:|---:|"])
    for dim in (3, 4):
        for name, summary in runs["D"]["trajectory"]["pairs"][str(dim)]["common_primary_success"].items():
            lines.append(f"| {dim}D | {name} | {summary['cases']} | {summary['dynamic_minus_fixed_sec']:+.3f} |")

    lines.extend(["", "## 解释范围", "",
                  "4D dynamic 的全程劣势在两轮均出现；3D dynamic 的全程优势未在 seed D 保持。当前数据不支持 dynamic 稳定优于 fixed 的经验主张。",
                  "Seed D 的两个 dynamic 分支全程 solve 总时间更低，但 setup 成本增加更多；4D dynamic 的首次失败数也少于 fixed，依然总时间更高。可靠性指标因此不能替代完整成本目标。",
                  "这不反驳可靠性 mixture 的误触发定理，也不证明物理时间上限、全局最优性或严格局部最优。新的 seed D 没有执行冻结交叉求解，不能把 seed C 的具体联合适配诊断直接当成 seed D 已验证的机制。",
                  "Seed C 最初来自历史 8D 最好结果的选择；seed D 是预先选定的复现。这里不报告跨 seed 显著性或将两轮当成代表性总体均值。原 runner 的逐题 paired intervals 也不解释为跨训练 seed 的置信区间。",
                  "两轮 Default 耗时不同，降幅使用各轮 Default 归一化；不能仅以跨日绝对时间变化衡量算法改进。",
                  "论文应同时呈现两轮而非仅保留获胜 seed。当前最值得保留的理论区分是可靠性触发与累计时间目标的差异，以及 setup–solve 联合学习的成本变化。", "",
                  "## 数据来源", ""])
    for seed, run in runs.items():
        lines.append(f"- Seed {seed}: [{Path(run['path']).name}]({run['path']}/screen_report.md)")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = {"C": read_run(args.reference_run), "D": read_run(args.run)}
    assert runs["C"].pop("inputs") == runs["D"].pop("inputs")
    assert runs["C"].pop("execution_ranks") == runs["D"].pop("execution_ranks")
    normalized = copy.deepcopy(runs["D"]["config"])
    original = runs["C"]["config"]
    for key in ("name", "description", "output_dir"):
        normalized[key] = original[key]
    for new, old in zip(normalized["methods"], original["methods"]):
        if "seed_offset" in old:
            assert old["seed_offset"] == 2000029 and new["seed_offset"] == 3000047
            new["seed_offset"] = old["seed_offset"]
    assert normalized == original
    launch = json.loads(args.run.with_suffix(".launch.json").read_text())
    root = Path(launch["cwd"])
    changed = [name for name, expected in launch["source_sha256"].items()
               if not (root/name).is_file()
               or hashlib.sha256((root/name).read_bytes()).hexdigest() != expected]
    runs["D"]["validation"]["source_files_differing_from_launch"] = changed
    assert not changed, changed
    args.output.mkdir(parents=True, exist_ok=True)
    _write_json(args.output / "comparison.json", _json_ready(runs))
    (args.output / "report.md").write_text(render_report(runs), encoding="utf-8")
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
