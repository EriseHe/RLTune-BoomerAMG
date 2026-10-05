# SISC paper completion plan: checkpoint preparation, frozen evaluation, and theory

**2026-09-29 论文范围更新：** 按用户决定，论文保留原 Module 04 在线结果及原 Module 05 编号的 Run 05 matched-hierarchy 对照；从论文中完整移除基于新 Module 05 checkpoints 的新 Module 06 own-pair/crossed 实验。后者的已完成记录仅作历史存档，不再作为当前论文待整合或必跑证据。新增三张论文图来自刚完成的 Run 05／相应 minimax 数学：五方法节省、W1/per-instance/periodic/RL 四面板 action heatmap，以及 weighted-minimax 双面板图。

日期：2026-09-15。代码基线：online-bandit-rl，13a507d。论文基线：Overleaf/SISC_submission/local_revision_20260915。

最新模块安排：2026-09-28。**04 保留现有在线累计性能协议与已接受结果。05 改为 checkpoint 准备：一次共享 1000 题 W1 前缀，再分支进行 4000 题 solve-specific setup 训练；06 在 fresh inputs 上比较各自配套的冻结完整方法，并可附小型 crossed-hierarchy 机制比较。原 06–10 顺延为 07–11。**
本次按用户授权执行一个 diffusion 60³ 训练 replicate。Default setup hierarchy 的 100 个独立 development inputs × 41 weights 校准已完成，选定 w_dev=1.40。用户随后要求丢弃已启动的四分支 run，并额外加入历史 learned-hierarchy fixed-grid 最优常数 1.60：当前五分支从头开始，不复用旧训练状态。十项集成测试已通过；最新 native 检查与执行状态见 05 入口。训练胜负或 payback 不是 05 的通过条件。
具体协议见 [05 checkpoint preparation](../../experiments/archive/paper_development/05_online_policies/README.md)，冻结评价入口见 [06 policy](../../experiments/archive/paper_development/06_policy/README.md)。

05 五分支训练已于 2026-09-28 纽约时间 06:10 完成，总耗时 54.81 分钟：21,000 次实际 method–problem 执行，无 unrecovered failure；五份 setup checkpoint 与 LSTDQ controller 均已保存并通过最终核验。训练成本仅作诊断。用户随后授权执行的 06 也已于纽约时间 09:55 完成：100 个 fresh diffusion 60³ inputs × 3 次计时，五个完整冻结方法加 periodic/RL 两个 crossed pairing，共 2,100 trials，耗时 4.997 分钟，无 failure 或 recovery，冻结状态及原始数据独立核验均通过。仅一个训练 checkpoint set，不能解释为多训练 seed 证据。

历史状态（2026-09-16）：01、02 与单 seed 的 03 当时均已完成；04 当时尚未启动。下文早期执行记录按其日期理解，当前编号与新增任务以上述安排和第 3 节为准。
报告、图表和阅读顺序见[总入口](../../experiments/paper_final/README.md)。
当前显示名称为 LinUCB（diffusion 4D / advection 7D，均含截距）；维数是 context 配置。

**Module 04 remains unchanged.** No source code, completed configuration, result, checkpoint or historical path is modified by this documentation revision. No experiment is launched. The earlier full plan remains available in [revision 53f48f8](https://github.com/EriseHe/RLTune-BoomerAMG/blob/53f48f8ab16cc91f38aa7863c964147e132ca521/docs/theory/paper_completion_plan_20260915.md). The detailed original T1–T6 proof review and September 15–17 execution history remain in [revision 007bd195](https://github.com/EriseHe/RLTune-BoomerAMG/blob/007bd195b9aa976f6b640bd0de861669e3dcde7f/docs/theory/paper_completion_plan_20260915.md).

## 1. Current division of work

| Module | Role | Primary deliverable |
|---|---|---|
| 01–03 | Numerical checks and earlier development/activation diagnostics | Preserve existing records |
| **04** | Existing complete online experiment and cumulative-cost evidence | Leave completed data and protocol unchanged |
| **05** | **Matched-budget setup/controller checkpoint preparation** | Four frozen setup selectors and the corresponding trained RL controller |
| **06** | **Fresh-input evaluation of the frozen trained methods, with optional crossed-policy diagnostics** | Post-training runtime, cycles, accuracy/recovery, and hierarchy–controller compatibility |
| **07** | Theory integration and deterministic proof checks | Consolidated numerical mechanisms, estimator identities, and qualified performance statements |
| 08 | Stronger conventional baselines/problem breadth | Conditional on manuscript scope |
| 09 | Retrained feedback/cadence ablations | Only for stronger mechanism claims |
| 10 | Optional new learning algorithms | Not required for Modules 05–06 |
| 11 | Paper assembly and reproducibility checks | Final manuscript, tables and archived protocol |

Former Module 05 / physical `05_policy` is now logical Module 06. Former `06_theory` is logical Module 07; later logical numbers shift accordingly. Existing scripts, imports, result directories and archive names are not renamed.

## 2. Module 05: matched training to prepare frozen setup policies

### 2.1 Why this stage exists

The final Joint setup selector was trained under weight one for the first 1000 systems, then under an evolving LSTDQ controller for the next 4000. Evaluating other solve policies only on that selector's hierarchies is a valid conditional substitution test, but does not give those alternatives their own trained setup–solve pairing.

Module 05 supplies that missing pairing. Fairness here means a common starting setup history and an equal number of subsequent problem opportunities for adaptation to each designated solve method. It does not mean that all models converge to an optimum, have identical retained update counts, or receive equal numbers of seconds/cycles. Those are outcomes, not matching constraints.

### 2.2 Training protocol

For each prescribed training replicate, execute **one actual LinUCB + weight-one prefix on problems 1–1000**. Clone the complete setup-learning state at the boundary: regression/inverse state, retained observations/counts, selected and elite history, candidate cursor/generator state, random-number state, and protocol counters. Four independently timed prefixes with the same seed are not a substitute for a genuinely shared state when measured times enter learning.

Retain the shared observations in every branch. Do not reset the model only for prescribed alternatives. The continuation is:

| Branch | Problems 1–1000 | Problems 1001–5000 | Final artifacts |
|---|---|---|---|
| 01_numerics | P0 | 修复 LSTDQ inverse recovery，核对 commit/rollback；统一协议记录 | 65 项相关测试已通过；正常路径、异常路径与 batch 对齐 |
| 02_diagnostics | P0/P1 | 同 setup 的 fixed/schedule 开发筛查，加困难工况检查 | 已完成 1176 次 diffusion 与 336 次 advection 比较；另保留 58 条中止短训练记录；见完成报告 |
| 03_activation | P1 | 80³ advection 的六线路启动敏感性 | 单 seed 的六路各 5000 题已完成、审核并出图；开发诊断，未运行 s2/s3 |
| 04_online | P1 | 原正式在线比较及其组件分析 | 保留现有协议与已接受结果；新增比较不回写 04 |
| [05_online_policies](../../experiments/archive/paper_development/05_online_policies/README.md) | P1，checkpoint 准备 | 一次共享 1000 W1 前缀，再五条独立 4000 题 continuation；diffusion 60³ | W1、Default 校准 1.40、历史 learned-hierarchy 1.60、(2.85,1.10)、正常 RL；四分支旧 run 已丢弃 |
| [06_policy](../../experiments/archive/paper_development/06_policy/README.md) | P1，原 05 | 五个配套冻结完整方法与 periodic/RL 2×2 cross | 已完成 100 fresh inputs × 3 次，共 2,100 trials；periodic 完整成本最低，冻结与数据核验通过；历史记录保留 |
| 07_theory | P1，可与实验并行 | 整合 T1–T5 与 weighted-minimax schedule 依据，压缩 theory/appendix | 主张和证明范围一致；最终不超过 26 页 |
| 08_baselines | P2 | 强 native/polynomial baseline，补问题结构或收窄外推 | 不把网格大小当作不同结构 |
| 09_feedback | P2，取决于主张 | 匹配的反馈消融 | 相同预算重训有/无 feedback 模型，或降低因果主张 |
| 10_algorithms | P3 | 遗忘机制、schedule-policy bandit、SquareCB | 只有诊断明确支持且有重跑预算才启动 |
| 11_paper | P1，收尾 | 图表、摘要、讨论、局部引用与复现包 | 每个数字可追溯；原模板保留；提交文本黑色 |

目录总入口：[experiments/paper_final](../../experiments/paper_final/README.md)。新结果进入相应当前编号目录；历史证据保持原路径。原冻结评价的 `05_policy` 目录、`run_05_policy*` 入口及带 `module05` 的归档继续作为 06 的历史证据，不改写已审核的文件、hash 或 run 编号。通用 runner/helper 继续复用原模块。

02 的 diffusion 使用已有兼容 checkpoint，advection 预先指定使用 03 第一个 replicate 的 start_1000_final；06（原 05）使用最终训练 checkpoint。02 当时的结果未改变 03 的设计、算法和参数。后续版本必须独立记录，不混入已接受的 04。协议核对纳入 01 和每个实验的准备工作，不再占单独编号。

### 9 月 17 日的依赖说明，按 9 月 28 日编号更新

- **04 前完成 P0：**统一最终残差记录与验证，核对 stencil 的预期问题定义和困难输入，解决开发/正式输入重叠并冻结失败处理协议。计时与 cycle-cap 修正已完成。
- **04 保留原主实验：**Default / LinUCB / LinUCB–LSTDQ，两个 family 均 fixed-1000。没有因 baseline 建议自动增加第四方法或额外前置实验。
- **06（原 05）评价冻结方法：**当前 fresh-input own-pair 比较使用新 05 的配套 setup/controller snapshots；旧 matched-hierarchy 诊断可继续引用已有 04 checkpoints。历史较宽候选库及 transfer 建议以各完成记录的实际范围为准；当前 unanchored pair 采用已核验的 (2.85,1.10)。不反向改动 04。
- **新增 05 准备可比的冻结 checkpoint：**共享 1000 W1 前缀后，让五条 LinUCB 在各自规定的 solve policy 成本反馈下学习 4000 题，为 06 提供配套 selector。范围仅一个 diffusion 60³ replicate，不自动扩展原 04 的全部 18 组。schedule 使用离散 minimax 规则，不再通过 PDE test time 搜索 pair。
- **P1 表示论文证据的重要性，不意味着全部是 04 的前置任务。**P2 crossed-controller / 时间特征重训练仍由论文主张决定。

### 2.3 Selecting prescribed policies before training

Freeze w_dev using separate development inputs before training its setup branch. Tuning on Default hierarchies is permissible but not uniquely neutral and does not establish an optimum on later learned hierarchies. A prespecified mixed development panel is another permissible choice. Call it development-selected, not the global optimum of the coadapted system. Do not use Module 06 test hindsight to choose this coefficient.

Use (2.85,1.10) as the specified periodic pair. It minimizes, up to permutation, `max_{0<=lambda<=1} lambda[(1-a lambda)(1-b lambda)]^2` over `W={1,1.05,...,3}`. That discrete SPD smoothing-surrogate rationale does not imply optimal multilevel runtime or nonsymmetric-advection convergence. High-first phase is separately prescribed. Other already tested schedules can remain Module 06 diagnostics; do not add more training arms automatically.

A per-instance fixed-weight hindsight oracle is not a training branch. Scanning 41 weights on 5000 supplied hierarchies already requires 205000 solve trials before repeats and recovery; feeding their minima into setup training creates a different information-rich procedure. Retain the fixed-weight oracle as a smaller Module 06 diagnostic.

### 2.4 What to retain and how to judge completion

Save full trajectories, terminal outcomes, cost components, learning-state audits, final setup/controller checkpoints, and exact source/config/input identifiers. Run functional checks for clone independence, policy switching at problem 1001, schedule reset, recovery, and accounting before the prescribed training runs.

**The primary product is the final checkpoint set, not a winning training-time curve.** Periodic being faster during training does not disqualify the RL checkpoint or this design. Do not require an online victory or payback threshold before proceeding to Module 06. Training costs remain available for diagnostics and a transparent description of the budget; they need not become an additional standalone primary paper experiment.

Use the prescribed final training endpoint and retain every valid replicate. Test outcomes must not select replacement checkpoints. Final scope/seeds are fixed before training. Initial engineering checks and any development runs remain separate. No launch or final seed list is implied by this document.

## 3. Module 06: evaluate the trained frozen methods

### 3.1 Primary comparison: each solve method gets its own adapted setup selector

On each common fresh test input x, evaluate these four frozen pipelines:

- B_W1(x) followed by fixed weight one;
- B_Fixed(x) followed by fixed w_dev;
- B_Periodic(x) followed by prescribed (2.85,1.10);
- B_RL(x) followed by frozen pi_RL.

All setup regressions, statistics and inference conventions are frozen. Fix the same intended frozen-selector rule across models before evaluation; do not silently switch some models from LCB to mean-greedy. No test cost updates any setup model or solve controller. Prescribe candidate generation, source selection and model snapshots before evaluating outcomes; cache each input/source hierarchy choice for paired timing repetitions and crossed evaluations. Frozen RL can still respond to the evolving residual and measured cycle-time inputs, but it performs no parameter/statistic updates; random exploration is disabled according to the frozen-evaluation protocol.

The primary full-pipeline metric includes setup-selector inference, hierarchy construction, native solve/residual monitoring, solve-policy execution, and recovery. There is no online training-update cost during frozen evaluation, though necessary execution/bookkeeping is charged. Retain separate native and controller-inclusive breakdowns and successful outcomes/failures. Historical training expense is described in Module 05; do not claim Module 06 by itself establishes cumulative online payback. Module 04 remains the paper's existing evidence for that distinct claim.

This comparison removes the asymmetry of forcing every prescribed baseline onto Joint-selected hierarchies. It compares **post-training complete methods**, not relaxation policies on one fixed hierarchy. Different trained sources can legitimately select different hierarchies, and their construction costs cannot be omitted from this full-pipeline comparison.

The common weight-one prefix is a deliberately specified, matched training history. It does not guarantee that 4000 further problems suffice for every learner or that the resulting models are globally optimal. It also does not evaluate the fastest possible training protocol for every prescribed policy; no such claim is needed for this checkpoint comparison.

### 3.2 Optional crossed controls to explain compatibility

When attributing a difference specifically to solve control, test policies on the same hierarchy. The compact Periodic/RL crossed comparison is:

\[
2\text{ families}\times3\text{ grids}\times3\text{ replicates}
=18\text{ groups}.
\]

每组 5000 个共同 matrix/RHS 输入，三种方法：

1. Default setup + default solve；
2. LinUCB setup + weight 1；
3. LinUCB setup + LSTDQ，前 1000 题固定 solve，1001 开始 RL。

总计 270,000 次 method–problem 执行，恢复尝试另计。18 组是 90,000 个组内配对问题，不是 18 个独立 seed；相同 replicate tuple 跨网格/家族复用。

**必需产物：**

- 每个 seed、family、grid 的全程累计 solver cost、相对配对 Default 的 reduction。
- joint 相对 setup-only 的增量收益，而不只报告 joint 相对 Default。
- setup、native solve、controller、bandit、recovery 分解和累计曲线。
- 未恢复失败与最终准确性；不能丢弃失败样本后以幸存者均值声称成功完成更快。
- 一次性准备成本和持续 overhead 的区分。
- 三个 stream-level replicate 的结果、均值与 SD。不能把同一条自适应轨迹的 5000 题视为 5000 次独立训练重复。

**停止/处理规则：** 若发生协议认定的未恢复失败，按现有 runner 暂停并检查；记录该次失败与协议修订，不换掉困难 seed。若修复影响算法，重新冻结受影响的正式比较。

**解释：** 这估计完整自适应方法之间的差异。因为 hierarchy 路径会分化，它不能单独识别 RL 在同一 hierarchy 上的作用。

### E2：组件与学习过程分析——现有 Module 2，无需另跑 PDE

复用 E1 日志，重点展示两个 family 的 60³：

- 全 5000 和最后 1000 的同口径成本分解。
- primary failures、最终 failures、recovery 成本；recovery 是 setup/solve 的子项，不能二次加总。
- 累计节省曲线及描述性 break-even；一次暂时过零不等于以后始终回本。
- 学到的权重随 cycle 和 problem 的分布，而不只挑一条最漂亮的 high–low 轨迹。
- 预先约定代表性轨迹的选择方式；示意轨迹用于说明行为，不作为 optimality 证据。

这是分析任务，不应预算为第二轮完整在线实验。

### E2b / 05：共享前缀后训练 solve-specific setup checkpoints

**目的：** 为 06 产生训练机会可比、各自适配 solve policy 的冻结 setup selector；不另设“RL 必须在训练期回本”的实验目标。04 继续承担已有在线累计性能证据。

前 1000 个 diffusion 60³ 问题仅执行一次 LinUCB + w=1，然后克隆完整 setup-learning 状态；在相同的后续 4000 个输入上，各分支只用自己的实际成本更新。

| 分支 | 第 1001–5000 题 continuation | 第 5000 题冻结产物 |
|---|---|---|
| W1 | setup learning + w=1 | W1-adapted setup selector |
| Fixed 1.40 | setup learning + Default 校准 w_dev=1.40 | 1.40-adapted setup selector |
| Fixed 1.60 | setup learning + 历史 learned-hierarchy reference w=1.60 | 1.60-adapted setup selector |
| Periodic | setup learning + (2.85,1.10) | Periodic-adapted setup selector |
| RL | setup learning + 现有 online LSTDQ | RL-adapted setup selector 与 LSTDQ controller |

克隆包含 regression/failure 统计、candidate 统计与 cursor、RNG、selection history 和 previous-update 成本估计。分支后可选择不同 hierarchy，mutable state 互相独立。一次前缀加五条 continuation 共 21,000 次实际 method–problem 执行（恢复另计），每条分支逻辑训练预算均为 5000。不能把复用的 prefix 在实际总计算量里重复加五次。

**固定权重：** 按用户选择，仅在 Default setup hierarchy 上校准。100 个独立 development 输入 × 41 个 weights，每对一次计时、题内随机顺序。候选必须完成所有输入，按平均 inclusive continuation 成本最小选择，平局取小 weight。包括 solve/dispatch 与所有 recovery 成本；共同初始 setup 单列。此次已选定 w_dev=1.40（100 个输入平均 inclusive continuation 54.391 ms，所选权重无 recovery），冻结为 Default-setup development-selected fixed weight，不称为普遍或联合系统全局最优。不复用 learned-hierarchy mixed panel 来选该常数。

**额外固定参考：** 用户要求保留 1.40 并加入历史最优常数。对已接受六个 Joint checkpoint（含重训 seed 4）的 29,520 条 fixed-grid scan 记录重新计算，先在每题内平均重复计时，再汇总 native continuation 成本，单一常数最小值为 **1.60**；600 个 seed–case 组合的均值为 79.818 ms，1.65 为 82.621 ms。逐 seed 最优为 seed 1 的 1.55 和 seeds 2–6 的 1.60。该线路标为 historical learned-hierarchy fixed reference；历史 scan 使用旧研究 test inputs，本次作为事先指定参考，不能称为 Default 校准结果或对新的 06 test inputs 的无偏调参。06 使用新输入。

**数值与学习协议：** 沿用 04 的 setup search space、features/hyperparameters、tol=1e-6、cap=50、smoother profile、failure rollback 和 Default/W1 fallback。periodic 每次 primary attempt 从高权重开始；RL 在前缀中不学习，第 1001 题启动。串行 native 执行，每题随机化 continuation 顺序。

按用户追加询问，也重新核对了原 setup-only hierarchy 的 29,520 条 fixed scan；六个 checkpoint 汇总的单一常数最优同样是 1.60（80.229 ms；1.65 为 82.311 ms）。因此历史 Joint 与 setup-only 的常数选择由同一条 fixed-1.60 线路覆盖，不增加参数相同的重复线路，仍为五分支。

**验收：** 状态独立与克隆一致、正确 action/phase、正确恢复和计时、规定终点的五份 setup snapshot 与一份 controller snapshot、可恢复的 supplementary decision history、配置和数据/source hashes。成本与 first-1000/active-4000/final-1000 窗口只作记录和诊断；不根据训练胜负更改 seed、终点或 checkpoint。等训练机会不代表每个最终模型均达到 oracle optimum。

本次只执行一个指定训练 replicate，已先通过短功能检查；多 seed 或扩大问题族另行冻结协议。逐实例 fixed oracle 留在 06 的小型 matched-hierarchy 诊断，不作为 05 的训练分支。

细节见 [05 协议](../../experiments/archive/paper_development/05_online_policies/README.md)；数学依据见 [weighted minimax note](period_two_weighted_minimax_20260928.md)。

### E3 / 06：冻结完整方法比较与小型 hierarchy compatibility 诊断

**当前主要比较：** 在共同 fresh test inputs 上，比较 B_W1 + fixed 1、B_Fixed + fixed 1.40、B_FixedPrior + fixed 1.60、B_Periodic + periodic (2.85,1.10)、B_RL + frozen LSTDQ。各自使用 05 训练的配套 setup selector；所有 test observation 均不更新 setup 或 solve learner。

因 selector 可选择不同 hierarchy，主要指标包含 setup selection/construction、solve execution、controller computation 与 recovery。这是规定训练预算后的完整冻结方法性能，不是相同 hierarchy 上的单独 relaxation 效果。

**可附机制比较：** 只取 periodic-trained 与 RL-trained 两个 hierarchy 来源，分别评估 periodic 和 frozen RL，形成 2×2 crossed comparison。同一行的比较隔离该来源上的 solve-policy 差别；对角线是各自配套的方法。不自动扩展成 5×5。

**本次已锁定并完成的协议（2026-09-28）：** 100 个全新 diffusion 60³ inputs，每个 pairing 三次计时；五个 own pairing 共 1,500 trials，两个 cross 增加 600 trials。两题独立 native preflight 不计入结果。七项集成检查、28 次 preflight、2,100 次正式 trial 及独立汇总核验均通过。完整训练 history 保留；公共新 candidate schedule、按 input 固定的 tie RNG 与 cursor 重置使 hierarchy 选择不依赖执行顺序，每次计时仍执行 setup selection。setup statistics/history 与 solve controller 全程冻结并核验。恢复沿用旧 frozen-evaluation 规则：一个选定 hierarchy 的 primary，再按需 Default/W1 fallback，不根据 test outcome 重新选 hierarchy。串行执行、题内方法顺序及每次 repetition 的题顺序随机化；每题先平均全部三次，再等权汇总 100 题。paired bootstrap 仅表示给定 checkpoint 的 fresh-input 不确定性。协议与 hashes 在首次 test trial 前锁定，见 06 入口。

**本次结果：** 完整冻结方法平均成本（包含 selection、setup、solve、controller 和 recovery）为 W1 160.118 ms、fixed 1.40 132.523 ms、fixed 1.60 127.213 ms、periodic 119.988 ms、RL 124.671 ms。RL 相对 periodic 多耗 3.903%（paired input-bootstrap 95% CI [2.844%, 5.004%]），但相对 fixed 1.60 节省 1.998%（[0.970%, 2.996%]）。同 hierarchy 的 solve+controller+recovery 成本：periodic-trained 来源上 periodic/RL 为 61.998/83.387 ms，RL-trained 来源上为 60.880/62.731 ms。RL 在自身 hierarchy 来源上的 native-only 优势仅 0.321%，区间跨零；计入 controller 后 periodic 仍较快。该证据支持本次 checkpoint/distribution 下 prescribed schedule 的实践优势及 setup–controller compatibility，不是对所有 hierarchy 或训练 seed 的普遍排序，也不改变 04 的在线证据。详表及原始数据链接见 06 入口。

**2026-09-29 更正复测已完成：** 原 05 编号的 [Run 05](../../experiments/paper_final/05_policy/RUN05.md) 保留六份 Joint-trained setup/controller checkpoints、100 个输入、hierarchy 和 fixed-grid 选择，使用 (2.85,1.10) 作为唯一 periodic baseline，移除 (1,3) 与 (2.6,1)。丢弃未完成尝试后从零执行三次计时重复，共 8460 个不同 solve，全部成功且无需 recovery。RL 相对 stream-wide fixed / per-instance fixed 的 native 节省为 11.72% / 9.58%；相对 periodic 为 0.63% ± 0.55 pp，含 controller 后为 -3.65% ± 0.63 pp。所有五方法在每个 checkpoint/input 内共享 Joint-trained hierarchy。独立复算、完整 ZIP 解包验证、原八图系统与十张 seed 补图均已完成；论文 matched table 与相关数字使用该更正结果，04 与独立的新 05/06 数据不变。这补齐了当前 pair 的六 checkpoint 对照，不代表整个论文实验叙述与理论整合已全部完成。

**历史 matched-hierarchy 证据与方案：** 原 05 的完成记录与现有数据仍保留在 [06 当前入口](../../experiments/archive/paper_development/06_policy/README.md)。以下较宽的旧建议仅作历史参考，不因本次改号而成为新的必跑清单；当前 pair 锁定 (2.85,1.10)，不再自动搜索 pair/phase。

**历史 matched-hierarchy 设计建议：**

- 初始范围：60³ diffusion 和 60³ advection，各三个最终训练 replicate。80³ advection 用 E0/E4 检查，若机制明显不同则扩展一组对应测试。
- 从 setup-only 和 joint 两个最终 setup learner snapshot 各产生一组 hierarchy 选择；同一个测试问题，两来源都评估。
- 测试实例与训练、选权重/选 schedule 的 development 实例分开。
- 每个测试 hierarchy 上所有 solve policy 重启同一个 x0，保持矩阵、RHS、hierarchy、tol、cap 一致。
- 优先复用已有 frozen-cross 诊断、controller snapshot/restore 和 native solve wrapper；不要直接把历史训练问题的 replay 标为独立测试。

**必需比较对象：**

| 对象 | 选择方式 | 如何解释 |
|---|---|---|
| B_Periodic | Own pairing | Transferred RL pairing |
| B_RL | Transferred periodic pairing | Own pairing |

Generate each source hierarchy once per input and restart the identical initial state for its policy comparisons. Common initial setup cancels in within-row solve contrasts, while policy-induced recovery setup remains charged. Add W1/fixed sources and fixed-weight scans only where they address the intended claim; a full four-by-four evaluation is not automatically required.

The two own-pairing cells answer post-training pipeline performance. The within-row differences answer conditional controller performance. A changed ranking between sources indicates compatibility, not that either conditional comparison is invalid. No hierarchy source is universally neutral.

### 3.3 Fixed-reference and timing details

Per-case/global fixed references must be recomputed on their specified source hierarchies. Best-observed values from a finite, noisy scan are not exact continuous optima. Do not transplant a weight selected on B_RL and label it optimal on B_Periodic. Keep development-selected weights distinct from test-hindsight diagnostics.

Use the same matrices/RHS, tolerance, cap and recovery convention; reset all mutable native solve state. If rebuilding, verify numerical/structural consistency rather than calling a hash of parameters a full hierarchy-array hash. Randomize serial policy execution and use prescribed timing repetitions without training on those repetitions. Report uncertainty at appropriate problem and independently trained checkpoint levels; repetitions are not additional independent training seeds. No fastest-repeat selection or silent dropping of failed cases.

No dynamic oracle, new solve-bandit, or proof of RL necessity is required. A retrained observation ablation remains conditional on a specific residual-feedback claim, not on completing the current evaluation.

## 4. Module 07: concise theory integration

Use the [original detailed mathematical review](https://github.com/EriseHe/RLTune-BoomerAMG/blob/007bd195b9aa976f6b640bd0de861669e3dcde7f/docs/theory/paper_completion_plan_20260915.md) for complete T1–T6 arguments. Preserve:

1. Spectral-moment identities distinguishing fixed-cycle contraction from mixed-cycle interactions, with explicit admissibility and SPD/norm assumptions.
2. The three-dimensional exact-Galerkin construction and its seven-versus-two comparison, with the same initial residual and equal-cycle-cost qualifications; take the implementation cap strictly greater than seven if cap exhaustion has priority.
3. The separate two-dimensional residual-greedy counterexample, which concerns completion objectives, not learning convergence.
4. Complete retained-episode LSTDQ coercivity and recursive/batch identities, not all-prefix invertibility, accurate Q-values or calibrated confidence.
5. Conditional performance/payback accounting, not a proved regret theorem for the actual evolving joint learner.
6. Explicit polynomial-schedule optimization classes: anchored (2.6,1), continuous fourth-kind pair, and grid-constrained (2.85,1.10). Their smoothing objectives are not full multilevel runtime objectives.

Proceed with manuscript editing in parallel. Do not reopen cap, failure-target, dynamic-start, feature or learner design solely to make a training curve favorable.

## 5. Reproduction and paper checklist

- Module 04 is unchanged; Modules 05 and 06 have distinct training and evaluation roles.
- Clone the actual common setup prefix; all four branches inherit its records and complete state.
- Freeze endpoint models before accessing fresh Module 06 outcomes.
- Preserve all incurred training costs in logs, but make frozen test cost the new comparison's principal result.
- Include hierarchy construction when comparing own-pairing pipelines; use matched hierarchies for direct solve contrasts.
- Keep prescribed-policy selection, training budget and final evaluation inputs distinct.
- Match solver tolerances, caps, failure/recovery, features, cost scopes and model-selection conventions.
- Keep old files/archives and numbering traceable; no destructive renaming is needed.
- Report supported finite-domain results without asserting necessary RL, neutral hierarchies or globally optimal learned setups.

## 6. Existing entry points

- [Experiment index](../../experiments/paper_final/README.md)
- [Module 04 protocol](../../experiments/paper_final/04_online/README.md)
- [Shared-prefix/state restore machinery](../../experiments/joint/solve_control/joint_4k_execution.py)
- [Existing frozen cross-source diagnostic](../../experiments/diagnostics/solve_control/diagnose_context_activation_crossed.py)
- [Setup feedback](../../setup/learners/linucb/setup_reselection.py)
- [LSTDQ episode/recovery handling](../../solve/controllers/sarsa/online_td_lambda.py)
- [Original theory review](theory_review_and_revision_plan_20260915.md)

`setup_only` 替代含义含混的 default，回答 RL 是否值得加入；完全默认 AMG 保留在 04。主比较是相对 setup-only 的全 5000 累计记录成本（保留失败及恢复）；相对 start_1000 的差值是次要比较。

采用嵌套共享前缀：一条 setup-only reference 执行全程，在各边界克隆完整 setup 状态、候选 cursor/history、计数器与 RNG，再启动对应 RL 分支。分叉后各分支用自己的成本更新，hierarchy 可以分化。六分支执行适配、边界测试与完成审计均已通过；复制的是完整状态。

当前单 seed 的物理主尝试数为 5000+4250+4000+3750+3500+3000=23500，恢复尝试另计。六条逻辑曲线都完整计入自身 prefix 成本，总逻辑方法题数为 30000。原先额外准备的两个 seed 不运行。

记录全程成本、setup/solve/controller/recovery 分解、首尝试与未恢复失败、hierarchy 分化、cycle cap、time feature clipping、学习步数。不能用短 post-activation 片段裁决 start_2000 的全程 payback。

若 1000 附近无一致劣势，保留为工程选择；若 seed 排序不一致，应报告敏感性；若困难工况显示系统性问题，先区分 setup、编码、探索和历史数据原因。该研究估计启动规则引起的完整联合路径差异，不把全部差异归因于相同 hierarchy 上的 RL 效果。

[03 的具体协议](../../experiments/archive/paper_development/03_activation/README.md)与[完成报告](../../results/paper_final/03_activation/REPORT.md)。五条 RL 路径的记录累计成本下降 14.09%–22.53%，各路仍有 28–30 个未恢复输入；不能由此声称全成功完成时间或启动点最优。03 随后已供给 02 的预定 checkpoint；[02 完成报告](../../results/paper_final/02_diagnostics/REPORT.md)显示必须保留强 fixed/schedule 对照。

### E5：更强的数值基线与问题广度——P2，高价值

优先级高于换神经 RL 或全面调 SquareCB。

- 对 SPD diffusion，评估一个合适的 native/polynomial smoother baseline；本地 HYPRE 源码已有 Chebyshev 支持，但当前 Python wrapper/计时接口是否足够仍需确认。
- 公平记录 spectrum estimation、额外 smoothing work、setup 和参数选择成本；不能按“cycle 数相同”认定工作相同。
- 非对称 advection 不直接继承 SPD Chebyshev 或对称 two-grid 理论；使用适用的 reference 或明确范围。
- 加一类结构不同的开发/测试工况，例如空间变化系数、旋转各向异性或来源明确的应用矩阵；当前主 problem registry 只有两个空间常系数家族，不能假设此项已具备。
- 若时间不足以规范实现新问题类，准确收窄到所研究的 constant-coefficient family，并把外推限制写在摘要/讨论合适位置。

一个适当的强 numerical baseline 或结构不同的工况，通常比在同一分布上继续加很多 algorithm seeds 更有说服力。

### E6：若坚持 residual-feedback 的主张——有条件 P2

固定 schedule 对照是必要证据，但测试几条 schedule 仍不能证明反馈“必不可少”。

若论文明确声称 residual feedback 产生额外收益，增加训练预算匹配的消融：

- 相同 physics/setup/last-action/cycle 信息，移除 residual-derived feedback，重新训练；
- 若主张包括全部 solve feedback，再明确是否同时移除 cycle-time feedback，避免通过其它状态特征泄露相同信息；
- 相同 seed/input/training budget/测试 hierarchy；
- 比较完整控制器与消融后的独立测试完成成本。

不要只在已训练模型测试时把 residual feature 置零；那通常是在制造分布偏移，不能识别训练时利用该信息的价值。预算不允许时降低机制主张，不强行补一张无效消融图。

## 6. 算法建议：什么值得试，什么先不做

### A. 当前主方法先保留

LinUCB–LSTDQ 有已有收益证据，暂无依据认为换一个一般算法会显著改善结果。当前应先解决 baseline 与数值正确性问题。

### B. 如果证据指向历史数据失配，再试有针对性的遗忘

先诊断：activation 前后不同 age 的样本误差、当前 hierarchy 分布变化、老样本对动作排序的影响。TD residual 小本身不是 Q 正确的证据。

若做变体，分开控制：

1. 仅改变 setup continuation-cost 的记忆；保留稳定 setup-cost 知识。
2. 仅对 LSTDQ 的完整 episode 加正权重/遗忘。

不要第一次就同时改两个 learner、context、epsilon 与 activation，随后无法解释收益。

若采用指数遗忘并保持固定 ridge，应形成

\[
\mathsf A_{\rm new}
=\rho\mathsf A_{\rm old}+(1-\rho)\lambda_{\rm sol}I+M_e,
\qquad
b_{\rm new}=\rho b_{\rm old}+b_e.
\]

直接乘 rho 会同时衰减 ridge，不能沿用同一个固定下界。v3 的协方差权重、checkpoint、inverse 更新和成本也需相应处理；正权重保留 coercivity 不代表原 O(d²) 更新结构自动不变。

### C. Schedule-policy bandit：有解释力的后续替代

先用 E3 判断简单 library 是否有竞争力，再决定是否训练一个每系统选择整条 schedule 的 bandit。它按完整 completion cost 学习，仍能利用多周期效果，且可降低每 cycle 决策成本。

在 fixed/exogenous hierarchy 协议、有限 library、有界损失等条件下，可引用 [Tsallis-INF](https://jmlr.org/papers/v22/19-753.html) 等有限臂结果。比较对象是指定 library 的固定成员；不是任意 feedback policy，更不是另一条联合训练历史。cycle cap 不等于 wall-clock 损失有界。

### D. SquareCB：次于上述诊断

[SquareCB](https://proceedings.mlr.press/v119/foster20a.html) 的理论依赖 realizability/online regression 条件。它可以作为保持相同 features/candidates 的探索规则消融，但不会自动解决 endogenous drift、candidate coverage、recovery labels 或模型失配。

遗忘/重启的引用应使用考虑已知修正的版本，例如 [Faury et al. 的技术说明](https://proceedings.mlr.press/v132/faury21a.html)。不借一个漂亮的 regret 指数反向证明当前代码。

### E. Dynamic cost deployment：后续研究

对话的独立随机化 \(X_i=2(1-Z_i)C_i-2Z_iC_i\) 与给出的 time-uniform 下界，在固定条件均值、有界成本等假设下推导正确。但：

- 一次只运行一个策略减少单个样本的执行量，也失去了同问题配对的方差抵消；所需样本数可能增加，不能直接宣称总体验证更便宜。
- 正的 cost advantage 不等于已经收回训练/评估投入。
- 冻结策略验证不能直接保证随后两 learner 持续变化的收益。
- 测多次 candidate snapshot 要控制相应的选择/多重检验。
- 启动训练需要信息；部署已有策略的检验不能代替这一步。

[Off-Policy Confidence Sequences](https://proceedings.mlr.press/v139/karampatziakis21a.html) 是相关方法论，不是当前主程序的现成保证。建议独立维护这条研究线。

## 7. 论文结构、页数与引用

当前 PDF 已独立核实为 32 页。SISC 官方要求超过 26 个 SIAM-template 页面会退回，推荐长度约 20 页；核心贡献的证明不能移到非核心 supplementary 来支撑一个不自足的主文。[官方作者指南](https://epubs.siam.org/journal/sisc/instructions-for-authors)

因此，“加入新理论”的实际操作应是替换与合并：

1. Introduction：聚焦 online cumulative-cost coupling；说明与已有 cycle-level PPO、scheduled relaxation 的差别。
2. Method：一份准确的 setup/solve/recovery/cost/feature 定义，压缩版本历史。
3. Theory：一般数值机制 + 两个用途不同的 corollaries；episode-boundary LSTDQ well-posedness；紧凑的 conditional comparison。
4. Experiments：完整方法、成本分解、matched hierarchy/schedules、启动敏感性及必要 robustness。
5. Discussion：说明哪些反馈收益得到支持、哪些只展示机制；区分理论模型与 nonsymmetric 实验。
6. In-paper appendix：核心证明和确有必要的算法细节。把 activation 的独立研究、开发日志、大量重复图等留在复现材料或另稿。

建议以 24–25 页为内部目标，留排版余量，不改变教授模板的字体与边距挤页数。参考性预算：引言/相关工作 2–3，setup/solve 方法 5–6，理论 3–4，实验 5–6，讨论 1，核心证明 2–3，参考文献约 2；实际合计需受总页数约束。

新增引用只做有针对性的核验：

- 三维构造与一般 moment 分析：scheduled/polynomial relaxation、必要的 rank-one 先例，明确哪些是本文推导。
- coercivity：pathwise LSTD 先例与 gamma=1 的适用边界。
- 若主文不采用新 bandit/gate 算法，不为它们扩张大段相关工作。
- 避免“首次 high–low”“首次 cycle-level multigrid RL”“首次任何形式 joint tuning”等过宽表述。已有 [cycle-level PPO 工作](https://arxiv.org/html/2407.15872v1)；[Learning to Relax](https://proceedings.iclr.cc/paper_files/paper/2024/file/8fcc228e94aa7e4773a27c6c2d886243-Paper-Conference.pdf) 的特定数值到可学习性联系可作为参照，不是当前实现已获得的保证。
- 检查新增 cite key、作者/年份/版本、引用所支持的精确陈述；不重启全文 bibliography 重审工程。

SISC 接受以理论或有力启发支持、并有详细数值验证的方法贡献；完整 RL 收敛定理不是普遍前提。仍需证明清晰的算法进展、准确性、效率与稳健性。[官方编辑政策](https://epubs.siam.org/journal/sisc/editorial-policy)

## 8. 建议时间顺序与收尾标准

以官方 2026-09-30 截止为依据，下面是工作安排建议，不是尚未测量的计算时长承诺。[Special-section call](https://www.siam.org/publications/siam-news/articles/call-for-papers-sisc-iterative-and-multigrid-special-section/)

| 时段 | 建议工作 |
|---|---|
| 9/16–9/17 | inverse 修复/测试、协议与篇幅预算；matched-hierarchy runner；E0；理论整合开始 |
| 9/18–9/20 | 开发版 E3 与 E4，完成最终设计选择；冻结正式代码和输入；检查存储 |
| 9/20–9/24 | E1 顺序运行，E2 随完成结果生成；从最终 checkpoints 执行正式 E3；同步写核心理论 |
| 9/24–9/27 | 有预算时补 E5/E6；替换所有 placeholder；整合图表与结论 |
| 9/28–9/29 | 独立检查数字/证明/新增引用/页数/复现；完成最终稿与投稿材料 |
| 9/30 | 截止日，不把新算法开发或关键长实验留到这一天 |

同一计时机器上的 native workloads 顺序执行。理论写作、日志分析和排版可以与计算交错进行，但避免明显污染计时。先根据 E0 的实际耗时估算 E1/E3/E4 总预算；如果超出预算，优先削减 P3、过密 schedule 搜索及开发图表，不牺牲失败记录或挑选性删 seed。

### 投稿前必须打勾的清单

- [x] exceptional inverse path 有正确处理，相关测试通过。
- [ ] 最终实现、数学符号、特征维数、target、恢复、计时协议一致。
- [ ] 两 family/三 grid/三 prescribed replicates 的主要比较完成，或明确且有理由地修改论文范围。
- [ ] setup-only 与 joint 的完整累计比较，包含所有学习与恢复成本。
- [ ] fresh-instance matched-hierarchy fixed/schedule 对照完成。
- [ ] 1000 的工程选择有适当敏感性证据，不写成最优启动点。
- [ ] 核心新增证明经核对整合；没有把 toy/可表示性/可逆性写成学习性能保证。
- [ ] 所有报告数值可追溯，不混用 18/18/18、旧 context 与最终 18/18/9 配置。
- [ ] 最终 PDF 无 placeholder、不超过 26 页；核心贡献证明在论文内。
- [ ] novelty、实验范围、limitations 和新增引用匹配最终实际证据。

## 9. 本地依据与可复用入口

- [已准备的正式实验协议](../../experiments/joint/solve_control/PAPER_FINAL.md)
- [正式 suite runner](../../experiments/joint/solve_control/run_paper_final.py)
- [Module 1/2 分析](../../experiments/joint/solve_control/analyze_paper_final.py)
- [LSTDQ 均值与 inverse 更新](../../solve/controllers/recursive_lstdq/v1.py)
- [LSTDQ v3 episode statistic](../../solve/controllers/recursive_lstdq/v3.py)
- [episode、executed next action 与恢复](../../solve/controllers/sarsa/online_td_lambda.py)
- [state encoder](../../solve/controllers/common/state_encoder.py)
- [setup suffix-cost 更新](../../setup/learners/linucb/setup_reselection.py)
- [现有 frozen-cross 诊断](../../experiments/diagnostics/solve_control/diagnose_context_activation_crossed.py)
- [共享前缀执行与完整状态恢复](../../experiments/joint/solve_control/joint_4k_execution.py)
- [已有 clone helper](../../experiments/joint/solve_control/setup_aware_compare_common.py)
- [当前理论源文件](/Users/erisehe/Documents/GitHub/Overleaf/SISC_submission/local_revision_20260915/sections/theory_updated.tex)
- [当前证明 appendix](/Users/erisehe/Documents/GitHub/Overleaf/SISC_submission/local_revision_20260915/sections/appendix_updated.tex)
- [用户提供的新增证明](/Users/erisehe/Downloads/additional_theorems.tex)
- [用户提供的完整审阅](/Users/erisehe/Downloads/REVIEW.md)

本计划记录的是建议与已核验事实。未完成项仍是待办，不表示相应实验、修复或论文本体已经执行。
