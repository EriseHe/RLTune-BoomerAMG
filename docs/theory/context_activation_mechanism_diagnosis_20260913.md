# 4D 动态启动的机制诊断

在这次 60³、5000 题的实验中，4D dynamic 的端到端累计耗时比 4D 固定启动高 67.82 秒。现有证据支持的解释是：两个分支形成了不同的 setup–RL 联合学习路径，并学到了不同的、相互适配的 hierarchy 与 relaxation 策略。dynamic 对应的组合更慢，而且在已检查的有限交叉组合中，单独替换其中一个组成部分不能改善它。

这不是“提前启动导致性能下降”的因果证明。两条 4D 轨迹在第 6 题就出现 setup 分歧，当时两者均未启用 RL。现有实验同时改变了启动机制与后续学习路径，无法识别保持完全相同启动前历史时的纯启动时刻效应。

## 1. 数据范围和检查方式

主数据来自 [完整 paper_test 运行](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/results/joint/paper_test_n60_context_activation_restart_20260913_093636/result.json)。六个分支各有 5000 条逐题记录，问题顺序、矩阵参数与 RHS 种子一致。所有分支记录的 unrecovered failure 数均为零；学习分支发生的首次失败均经恢复处理。成本包含记录的 setup、native solve、恢复、bandit 和 controller 开销；它不是包括矩阵生成、候选表预计算和绘图在内的整个进程持续时间。

诊断包括：

1. 原始轨迹的成本、cycle、残差缩减、动作、setup 和探索历史分析。
2. 四个 3D/4D 学习分支在 1000、2000、3000、4000、5000 题的 20 个 LSTDQ checkpoint 数值检查。
3. 最终 4D controller 在相同记录状态上的决策比较，以及删除早期 estimating-equation 项的敏感性检查。
4. 最终 4D bandit 在共同候选 setup 上的预测与实际观察覆盖检查。
5. 一组新的、小规模冻结交叉诊断：50 个已有矩阵位置、500 次 native 求解。两组各 25 个位置事先由等距索引确定；分别使用第 2000 题和第 5000 题的 checkpoint，每题交叉两个 setup 来源、两个 controller 来源、LCB/mean 两种评分，并加入默认 weight=1。所有学习与随机探索均关闭，结束后确认 A、b、theta、更新计数与 RNG 状态不变。

交叉诊断的第 2000 题 checkpoint 使用原轨迹第 2001、2041、…、2961 题；最终 checkpoint 使用第 4001、4041、…、4961 题。后者是训练集内机制检查，不是独立泛化评测。两组检查均不能替代从相同历史分叉的启动实验。

可复核材料：

- [轨迹、checkpoint 与评分审计 JSON](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/results/diagnostics/paper_context_activation_mechanism_20260913/mechanism_audit.json)
- [冻结交叉诊断协议](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/results/diagnostics/paper_context_activation_mechanism_20260913/crossed_frozen/protocol.json)
- [冻结交叉诊断结果](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/results/diagnostics/paper_context_activation_mechanism_20260913/crossed_frozen/summary.json)
- [只读审计脚本](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/experiments/diagnostics/solve_control/analyze_context_activation_mechanism.py)
- [冻结求解诊断脚本](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/experiments/diagnostics/solve_control/diagnose_context_activation_crossed.py)

## 2. 67.82 秒差距从哪里来

这里的差值均为 dynamic 减去对应维度的固定启动，正数表示 dynamic 更慢。

| 4D 阶段 | 成本差 / s | 解释 |
|---|---:|---|
| 第 1–781 题 | +15.24 | 两者均未启用 RL，已有不同 setup 学习路径 |
| 第 782–1000 题 | −12.59 | 只有 dynamic 启用 RL，这段实际节省了时间 |
| 第 1001–5000 题 | +65.18 | 两者均启用 RL，持续性能差距是主要来源 |
| 全部 5000 题 | +67.82 | 三段之和 |

因此，不能把现象概括成“RL 一启动就表现很差”。4D dynamic 的前 219 个 RL 问题没有首次失败，且相对于另一条轨迹的同期固定 solve，成本低 12.59 秒。长期差距是在两条联合学习轨迹均已启用 RL 后持续积累的。

4D dynamic 的 gate 全程开销约 0.0378 秒。全程平均 native solve 时间较固定启动高 12.17 ms/题，controller 开销高 0.869 ms/题；gate 开销无法解释主要差距。

成功的受控求解在不同阶段有如下表现：

| 问题区间 | 固定启动：cycle/题 | dynamic：cycle/题 | 固定启动：ms/cycle | dynamic：ms/cycle |
|---|---:|---:|---:|---:|
| 1001–2000 | 11.742 | 14.666 | 7.436 | 7.387 |
| 2001–3000 | 11.537 | 14.501 | 7.488 | 7.331 |
| 3001–4000 | 11.490 | 14.071 | 7.495 | 7.315 |
| 4001–5000 | 11.451 | 12.807 | 7.497 | 7.389 |

cycle/题只统计成功的 primary controlled episodes。ms/cycle 使用记录的受控 cycle 时间，包括已记录的失败 primary cycle。最后一个区间两者均无首次失败，因而不存在该样本口径差异。

dynamic 的单次 cycle 并不更昂贵；它需要更多 cycle 才达到容差。最后 1000 题，每 cycle 的平均负对数残差比为 1.123，固定启动为 1.249。这是沿实际轨迹的残差缩减统计，不能直接解释为误差传播算子的谱半径。

## 3. 两个 learner 在 RL 前已分叉

4D 两分支第一次选择不同 setup 是第 6 题。第 5 题参数相同，但记录的端到端时间分别约为 0.7650、0.8106 秒。时间反馈进入 bandit 更新，之后不同动作、不同观测又进一步改变学习状态。3D 两分支的首次 setup 分叉发生在第 39 题。

同 seed 保证相同的伪随机起点和问题流，不保证以实测时间为标签的 learner 轨迹一致。实际候选集合也不是一张对所有分支始终相同的固定表：structured512 复用了基础候选表，但又结合各分支自身的 previous/elite anchors 和局部邻域。因此，历史分歧还会反馈到候选集合。[候选集合实现](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/setup/learners/linucb/SharedLinUCB_AMG_v4.py:750)

最后 1000 题的主导 setup 也明显不同：

- 固定启动：950/1000 题使用 coarsen_type=3、interp_type=8、trunc_factor=0、P_max_elmts=2 这一组四个参数；其余参数仍可能变化。
- dynamic：621/1000 题使用 coarsen_type=3、interp_type=8、trunc_factor=0.95、P_max_elmts=2；305/1000 题使用 coarsen_type=10、interp_type=6、trunc_factor=0、P_max_elmts=2。

第 1001–2000 题，dynamic 的后两类中，coarsen=10/interp=6 和 coarsen=6/interp=6 分别出现 397、366 次，平均 primary cycle 数约为 15.39、15.63。固定分支主要的 coarsen=3/interp=8/trunc=0/P_max=2 组合出现 919 次，平均约 11.66 cycle。由于选择与 PDE context 有关，这些组内均值本身不是改变 setup 的因果估计；下一节的交叉诊断用于进一步分离机制。

## 4. 冻结交叉诊断揭示相互适配

在每个选定矩阵上，分别使用两条原轨迹记录的 setup，并交叉运行两条冻结 controller。下面是无 epsilon 探索、保留原 beta=2 的 LCB 决策时，25 题的平均 cycle 数。所有单元均无 primary failure。

### 第 2000 题 checkpoint

| Setup 来源 | 固定分支 controller | dynamic 分支 controller | 默认 weight=1 |
|---|---:|---:|---:|
| 固定分支 | **11.28** | 13.36 | 18.20 |
| dynamic 分支 | 17.40 | **14.48** | 20.60 |

### 第 5000 题 checkpoint

| Setup 来源 | 固定分支 controller | dynamic 分支 controller | 默认 weight=1 |
|---|---:|---:|---:|
| 固定分支 | **11.16** | 13.76 | 18.36 |
| dynamic 分支 | 15.60 | **12.44** | 19.44 |

最终 checkpoint 对应的平均 setup+native solve+controller 时间为（冻结评估，没有 bandit 选择或学习更新）：

| Setup 来源 | 固定分支 controller | dynamic 分支 controller |
|---|---:|---:|
| 固定分支 | **189.87 ms** | 211.34 ms |
| dynamic 分支 | 228.32 ms | **200.71 ms** |

这些时间是新的冻结诊断测量，不能混入原 5000 题统计。核心解释优先依据 cycle 数，而不是不同运行时段的绝对毫秒值。

两张 cycle 表都显示：在所比较的两个 setup 来源与两个 controller 中，每个 controller 在其自身训练轨迹的 setup 来源上更有利。较慢的 dynamic 联合组合，单独替换 setup 或单独替换 controller 都更慢；同时替换两者，才到达更快的固定分支组合。

这支持“联合适配形成不同性能水平”的解释。它不证明整个参数空间中存在严格局部极小点、稳定均衡或吸收状态，也不证明第 782 题启动是进入该组合的必要原因。

### Uncertainty 的作用也依赖 setup 分布

将冻结策略的 beta 从 2 改为 0，关闭的是 LCB bonus；原始 theta、训练历史和随机探索设置均未改变。

| 最终 checkpoint 的组合 | LCB beta=2 | Mean beta=0 |
|---|---:|---:|
| 固定 setup + 固定 controller | 11.16 | 11.12 |
| dynamic setup + dynamic controller | 12.44 | 12.48 |
| 固定 setup + dynamic controller | 13.76 | 13.20 |
| dynamic setup + 固定 controller | 15.60 | 12.24 |

在自身 setup 分布上，去掉 bonus 几乎没有改善。主要变化出现在跨分支迁移，尤其是把固定 controller 放到 dynamic setup 上。因此，不能由此建议全局设 beta=0；证据指向的是跨 setup 分布时的预测与不确定性行为。

对 1187 个记录状态做相同输入的 frozen scoring，也看到两条 controller 有显著动作分歧。固定 controller 在固定来源状态上的平均 selected uncertainty 约 0.87 ms，在 dynamic 来源状态上约 8.26 ms；dynamic controller 的对应数值约为 0.92 ms（自身来源）与 9.90 ms（固定来源）。这些是 implemented uncertainty proxy 的数值，不是经本次诊断证明有效的置信区间。

## 5. Bandit 也形成了不同的经验覆盖和成本预期

用最终 bandit 对共同的 25 对 setup 候选评分：每对来自同一矩阵位置的固定分支选择与 dynamic 分支选择。仅对指定候选评分，不要求它们同时属于某一时刻实际生成的候选集合。

| 评分模型 | 对固定分支 setup 的平均预测成本 | 对 dynamic setup 的平均预测成本 | LCB 更偏向固定 setup 的比例 |
|---|---:|---:|---:|
| 固定分支 bandit | 183.74 ms | 457.89 ms | 100% |
| dynamic 分支 bandit | 496.73 ms | 196.65 ms | 0% |

dynamic bandit 对这 25 个固定来源 setup 的精确 arm 均没有实际观察记录；固定 bandit 对 25 个 dynamic 来源 setup 中的 20 个没有精确 arm 观察。context–action 特征允许泛化，故“未观察精确 arm”不等于数学上完全没有相关信息；但它明确显示了两条路径不同的直接经验覆盖。

预测的跨分支成本远高于冻结交叉诊断实际测得的成本量级。即使把各自 controller 的条件性考虑进去，这仍表明不能把这些 extrapolated predictions 当成可靠的跨组合性能证书。时间分布、持续更新的 solve policy、线性近似误差和样本覆盖都可能参与其中；本次不能唯一分解这些来源。

还需注意：dynamic bandit 偏向自身 setup，并不意味着只替换它的 setup 就会改善实际性能。交叉表显示，在当前 dynamic controller 固定的条件下，来自固定分支的 setup 的确可能更慢。问题涉及联合改善，而非单个 learner 是否选错了一个明显更好的条件动作。

## 6. 排除的解释与尚未排除的机制

**递归计算失稳不是当前证据指向的解释。** 20 个 checkpoint 的最大相对 inverse residual 约为 6.2×10⁻¹¹，最大 recursive theta 与直接 batch solve 的相对差约为 5.7×10⁻¹¹；均无 inverse rebuild，moment covariance 的最小特征值约为初始 floor 10⁻⁶。这里检查的是数值 estimating equation，不是 Q 函数逼近真实价值的误差。

**dynamic 更慢并不是因为它进行了更多随机探索。** 第 1001–2000 题，4D dynamic 的 epsilon 平均约为 0.194，固定分支约为 0.234；最后 1000 题约为 0.0496 与 0.0657。冻结并关闭随机探索后，性能差距仍然存在。

epsilon 按 cycle 更新次数衰减，而非按 problem 数衰减。dynamic 经历更多 cycle，会更快消耗探索进度。这可能强化路径差异，但目前只是由实现支持的潜在反馈机制，尚无控制实验量化其因果贡献。[epsilon 实现](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/solve/controllers/common/linear_lcb.py:242)

**“前 219 个 RL episode 污染了最终方程”也不是充分解释。** 从最终 dynamic 的 A、b 中删除第 1000 题 checkpoint 相对于 ridge 的增量，并同步移除早期 covariance 增量后，在 dynamic 来源的 629 个记录状态上，LCB 动作只改变约 15.1%。此操作不能撤销后续 4000 题的路径变化，也不是删除早期样本后重新训练的性能估计。它表明不能仅靠删除早期方程项就宣称解决了路径依赖。

**平均特征不是新增物理信息。** 4D 的 mean 是三个 log 系数的线性组合。它改变的是特征坐标、正则化与不确定性几何，不能单凭增加或删除它推导单调性能改善。当前固定启动下 4D 优于 3D、dynamic 下 3D 优于 4D，也不支持单一维度规则。

## 7. 数学上应怎样理解这次现象

令 a_t 表示 setup action，H_t=H(A_t,a_t) 表示构建出的 hierarchy，π_t 表示 solve 的行为策略。对于固定矩阵与相同的恢复约定，记

\[
J_t(a,\pi)
=\mathbb E[\text{setup、solve、恢复与规定的控制开销}\mid A_t,a,\pi].
\]

setup learner 实际面对的成本函数会随着 π_t 改变。solve learner 的数据分布也随着 setup 选择改变。在给定问题分布和合适的状态充分性假设下，可把其训练分布写为 d^{a_t,\pi_t}。两个更新相互改变对方的学习目标与样本覆盖。

当前 recursive LSTDQ 累计的是

\[
\mathsf M_n
=\lambda_{\rm ridge}I+
\sum_{j=1}^n z_j(\psi_j-\gamma\psi_{j+1}^{\rm executed})^\top,
\qquad
b_n=\sum_{j=1}^n z_j c_j,
\qquad
\widehat\theta_n=\mathsf M_n^{-1}b_n.
\]

这里 j 是按时间展平的 cycle 索引，z_j 为 episode 内 eligibility trace，c_j 为实现采用的 cycle 成本。后继 action 使用当时实际执行的 action；旧样本没有在每次策略变化后重新指定为当前目标 policy 的 action，也没有遗忘权重。[均值更新实现](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/solve/controllers/recursive_lstdq/v1.py:125)

所以，recursive=batch 的数值等价成立，不意味着该 batch 方程恰好是在评估当前一个固定 policy 的真实 Q，也不意味着增加样本会单调改善控制性能。经典 LSPI 的固定目标策略评估与近似 policy improvement 分析必须和这个具体的在线、变化行为策略实现区分。[Lagoudakis–Parr, 2003，§6、图5](https://www.jmlr.org/papers/volume4/lagoudakis03a/lagoudakis03a.pdf#page=17)

本次有限交叉表支持以下局部描述：

\[
J(a_F,\pi_F)<J(a_D,\pi_D),\qquad
J(a_D,\pi_D)<J(a_F,\pi_D),\qquad
J(a_D,\pi_D)<J(a_D,\pi_F).
\]

这里 J 指冻结诊断中所选矩阵上的经验平均，并非总体真值。a_F、a_D 指逐矩阵记录的 setup 选择，而不是所有矩阵共用同一个 hierarchy。这个结构说明：逐个替换一个组成部分可能看不到更好的联合组合。启动时刻影响的是后续整个联合学习路径，而不只是“获得多少个额外 RL 样本”。

近似 policy iteration 本身也没有无条件的单调改进性质；以性能改进下界为目标的受控更新，是另一类需要额外条件的算法设计。[Pirotta 等, 2013](https://proceedings.mlr.press/v28/pirotta13.html)

## 8. 为什么失败率不足以决定成本最优的启动时刻

**可识别性反例。** 考虑两个环境，在默认 relaxation 下每题都成功，成本也都恒为 1。它们在该默认策略产生的全部可见历史上完全相同。但一个候选 RL policy 在第一个环境的成本为 1/2，在第二个环境的成本为 2，也都成功。任何只读取默认运行失败历史的启动准则，在两个环境中给出相同的启动分布，却无法同时正确判断这个候选 policy 的成本优势。

因此，即使完全准确地知道默认策略失败概率为零，也不能推出 RL 的成本改进。这不是换一个更严格的失败率阈值就能解决的问题。

本例的 4D dynamic 在启动后的 4219 题中只有 3 次 primary failure，但 cycle 数和累计成本仍偏高。这与反例指出的信息缺口一致：是否能在 50-cycle cap 内收敛，与能否用 11 cycle 而非 15 cycle 收敛，是不同目标。

还要区分“开始收集 RL 数据”和“正式使用一个已训练的 RL policy”。在固定 action w₀ 下收集状态时，

\[
\psi(s,w_0)=b(w_0)\otimes x(s)
\]

的线性跨度维数至多为 dim(x)，而整个 state–action 空间的特征维数为 dim(b)dim(x)。在没有其他动作、模型关系或额外结构假设时，默认 action 数据不足以识别完整 action-value 参数。Ridge 使矩阵可逆，不等于为未尝试动作创造了证据。因此，如果未来想根据 RL 的置信性能来决定部署，必须说明之前如何获得足够的 RL 探索或验证数据。

## 9. 下一步理论的目标与边界

目前应先明确优化目标：

\[
\min_{\tau,\ \text{activation protocol}}
\mathbb E\!\left[\sum_{t=1}^{T} C_t\right],
\]

其中 C_t 包含学习、验证、恢复的实际成本，且 setup 与 solve 更新规则是 protocol 的一部分。不能把 τ 当成一个不改变后续训练路径的截断参数。

合理的后续理论需要区分三项：

1. **基本可运行性。** Reliability evidence 可作为进入后续阶段的条件；它自身仍需要明确的风险模型与误触发含义。
2. **训练和联合适配。** 明确 setup 分布是否冻结、缓慢变化或受某个变化预算约束；明确 controller 数据覆盖、策略更新和旧数据处理。当前例子要求把这一层写入问题定义。
3. **部署后的成本收益。** 在同一 setup 条件下或明确定义的联合 comparator 下，对候选 policy 的成本优势给出可信下界，并计入剩余预算和获取证据的成本。用来探索的 optimistic LCB 不可直接作为“部署必然更省时”的证书。

若一个优势下界只针对冻结的 setup 与 policy，就不能直接外推到两者都继续在线变化的后续 4000 题。需要固定评估阶段、控制更新幅度、定期重新验证，或者证明分布漂移带来的误差项。Safe Policy Improvement 文献提供的是这一类基于性能与覆盖的思路，不是可以直接套用到当前连续 context、gamma=1、联合自适应实现上的现成定理。[Laroche 等, 2019](https://proceedings.mlr.press/v97/laroche19a.html)

现阶段不宜宣称新的某个 failure threshold、固定等待下界或 epsilon 调整已经解决启动问题。上述诊断已经揭示了成本差距的具体机制，但没有识别只改变启动时间的因果效应。

如果论文最终要声称“该动态时刻优于第 1001 题启动”，下一项最有针对性的验证应是：在完全相同的启动前历史处分叉，复制 bandit、controller、candidate cursor、RNG 与成本记忆等全部状态，然后比较立即启动与延后启动。配合独立随机重复，才能把启动机制的收益与前期路径分歧分开。无需为了这个目标先重跑所有 context 组合，也不应将本次冻结诊断替代成正式性能排名。

## Sources

1. 完整 paper_test 原始运行，5000 题配对流：[result.json](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/results/joint/paper_test_n60_context_activation_restart_20260913_093636/result.json)、[screen_report.md](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/results/joint/paper_test_n60_context_activation_restart_20260913_093636/screen_report.md)、[启动记录](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/results/joint/paper_test_n60_context_activation_restart_20260913_093636/rl_activation.json)。
2. 本次派生诊断：[审计](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/results/diagnostics/paper_context_activation_mechanism_20260913/mechanism_audit.json)、[冻结交叉表](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/results/diagnostics/paper_context_activation_mechanism_20260913/crossed_frozen/summary.json)。
3. 实现：[LinUCB](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/setup/learners/linucb/SharedLinUCB_AMG_v4.py)、[setup cost 更新](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/setup/learners/linucb/setup_reselection.py)、[LSTDQ mean](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/solve/controllers/recursive_lstdq/v1.py)、[v3 covariance](/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/solve/controllers/recursive_lstdq/v3.py)。
4. Lagoudakis, M. G., and Parr, R. “Least-Squares Policy Iteration.” JMLR 4, 1107–1149, 2003. [论文](https://jmlr.org/papers/v4/lagoudakis03a.html)。
5. Pirotta, M., Restelli, M., Pecorino, A., and Calandriello, D. “Safe Policy Iteration.” ICML, PMLR 28(3), 307–315, 2013. [论文](https://proceedings.mlr.press/v28/pirotta13.html)。
6. Laroche, R., Trichelair, P., and Tachet des Combes, R. “Safe Policy Improvement with Baseline Bootstrapping.” ICML, PMLR 97, 3652–3661, 2019. [论文](https://proceedings.mlr.press/v97/laroche19a.html)。
