# Activation 判据与 AMG 总运行时间的性能平台

所讨论的性能是 setup、solve 和约定计入的开销之和，而非收敛因子或迭代次数本身。不同配置可以通过不同的时间分配获得相近的总耗时；这与数值传播算子是否相同是不同问题。

现有 sequential reliability criterion 可以保留，用于定义有明确误触发控制的训练启动协议。它不保证成本最优启动。对于 60³ 上观察到的 55–60% 时间下降平台，可以建立严谨的问题定义和条件性最优性证书；目前尚没有覆盖所有允许配置、且足够紧的时间下界来证明这个具体百分比是硬上限。

## 1. 现有 activation 定理仍然有效

### 1.1 证明没有被新实验推翻

在始终使用 reference solve policy、setup 继续学习的辅助过程中，定义 \(Y_t^{\rm ref}\) 为第 t 题首次求解在恢复前失败的指示量。令

\[
p_t^{\rm ref}=\mathbb E[Y_t^{\rm ref}\mid\mathcal F_{t-1}].
\]

实际运行在触发前与该过程一致。为叙述无限时域和检测延迟，辅助 reference 过程需在数学上定义到触发之后；这不要求实际执行反事实求解。

对 \(0<p_1<p_0<1\)，令

\[
\Lambda_t=(p_1/p_0)^{Y_t^{\rm ref}}
[(1-p_1)/(1-p_0)]^{1-Y_t^{\rm ref}}.
\]

当 \(p_t^{\rm ref}\ge p_0\) 时，

\[
\mathbb E[\Lambda_t\mid\mathcal F_{t-1}]
=p_t^{\rm ref}\frac{p_1}{p_0}
+(1-p_t^{\rm ref})\frac{1-p_1}{1-p_0}
\le1.
\tag{1}
\]

每个固定起点的乘积及其权重和为 1 的正混合因此都是非负 supermartingale。Ville 不等式给出
\(\Pr(\sup_t\mathcal M_t\ge1/\delta)\le\delta\)。这个证明允许 setup 自适应学习和观测依赖，不要求独立同分布。[Howard 等，2020](https://arxiv.org/abs/1808.03204)

权重 \(\omega_q=1/[q(q+1)]\) 的尾和为 \(1/(t+1)\)，所以所给常数内存递推正确。该尾项应保留。

对于固定的确定性整数 \(\nu\)，若 continued-reference 过程在 \(t\ge\nu\) 满足 \(p_t^{\rm ref}\le p_1\)，则 \(\log\Lambda_t\) 的条件均值至少为所给 KL 散度 I，取值区间长度为 D。条件 Hoeffding 界给出

\[
\Pr\left\{
\sum_{t=\nu}^{\nu+d-1}\log\Lambda_t
<dI-D\sqrt{\frac d2\log\frac1\eta}
\right\}\le\eta.
\tag{2}
\]

当证据下界达到 \(\log[1/(\delta\omega_\nu)]\)，起点为 \(\nu\) 的混合分量就足够触发。因此引用的检测延迟结论也成立。若同时满足变点前的坏风险假设，则

\[
\Pr\{\nu\le\tau_{\rm start}\le\nu+d-1\}
\ge1-\delta-\eta.
\tag{3}
\]

随机变点或看过数据后挑选的 \(\nu\) 需要额外处理，不能直接套用这个确定性 \(\nu\) 版本。

### 1.2 应保留的含义与应撤回的外推

定理控制的是：reference 过程始终处于 \(p_t^{\rm ref}\ge p_0\) 时，检验仍然触发的概率。它不直接控制新 RL policy 的失败概率，也不保证触发后的累计时间优于固定第 1001 题启动。

即使检验正确拒绝了“始终不可靠”，也不自动证明下一题 reference 风险低于 \(p_0\)。风险先下降、后反弹的过程不属于该原假设；未来风险的解释依赖持续改善等附加条件。

因此，“false activation”应限定为这个原假设下的 false trigger。\(p_0,p_1,\delta\) 和权重仍然是设计选择，只是其统计意义比任意分块明确。多个分支各用 \(\delta\)，也不等于所有分支合计只有 \(\delta\) 的误触发概率。

### 1.3 新结果支持继续使用，但不支持成本最优主张

完整 60³ 运行中，3D 和 4D dynamic 分别从第 660、782 题启用 RL。3D dynamic 相对对应固定启动分支累计节省约 42.78 秒；4D dynamic 多花约 67.82 秒。

4D 在第 782–1000 题实际节省约 12.59 秒；第 1001–5000 题才积累约 65.18 秒的劣势。两个 4D 分支在第 6 题已选择不同 setup，所以不能把全部差距归因于启动时刻。[机制诊断](context_activation_mechanism_diagnosis_20260913.md)

这些结果没有反驳可靠性定理，但显示“reference 运行可靠”和“立即进入某条长期联合学习路径最省时”是不同命题。

建议继续保留该规则，作为轻量、可复现的 reliability-based training-start protocol，同时保留 fixed-start ablation。现阶段不建议为了一个负例事后调阈值，也没有理由认为固定 1000 天然更有理论依据。

subsection 可定位为 “A reliability-based criterion for initiating solve-policy learning”。theorem 后可加入以下建议文字；这是拟议的新表述，不是文献引文：

> The stopping rule initiates solve-policy exploration after accumulating evidence against persistent unreliability of the setup learner under the reference solve policy. The theorem does not certify the reliability of the subsequently changing solve controller, nor does it establish wall-clock optimality over activation times.

此前提出的成本部署检验仍只是[条件性研究框架](cost_based_rl_activation_framework.md)。它需要额外性能证据并约束部署后的变化，取得证据也有成本；不能直接声称其已是更好的实际替代品。

## 2. 时间加速上限的对象

固定问题分布、矩阵规模、RHS 与初值、残差容差、恢复规则、硬件与并行设置、hypre 实现、允许的 setup 空间、允许的 solve-policy 类和时间统计口径。

用 z 表示一个完整可行方案。它可以是固定 setup 与 solve policy，也可以是根据 context 选择 setup 的规则配合 sequential solve policy；具体选择必须固定。要求这些方案完成相同的求解任务，不能把更快报错退出当成加速。

定义真实平均时间

\[
J(z)=S(z)+V(z),
\tag{4}
\]

其中 S 包含 setup 及分配给 setup 的选择/学习开销，V 包含 solve、controller 及相应恢复开销。所有计入目标的成本应完整归属，既不遗漏也不重复。

对有限预算在线算法，同样可以定义 S、V 为整个问题流的平均分项时间，但这时训练、探索与恢复都必须计入，不能与冻结部署的时间混用。

令 \(\mathcal Z\) 是所讨论的全部允许方案，

\[
J_\star=\inf_{z\in\mathcal Z}J(z),\qquad
R_\star=1-\frac{J_\star}{J_{\rm default}}.
\tag{5}
\]

这里 \(R_\star\) 是最大时间下降比例。55–60% 的下降对应最小成本约为默认的 40–45%，不是 1.55–1.60 倍加速。后者仅对应约 35.5–37.5% 的时间下降。

即使矩阵规模不变，改变容差、允许的 smoother、线程数、kernel 实现、是否计入 controller 开销或是否复用 hierarchy，都可能改变 \(J_\star\)。因此它至多是指定求解合同内的上限。

这次完整运行的最后 1000 题，8D fixed 的 native 时间约为 171.79 ms，含 controller/bandit 为 184.10 ms，默认为 371.00 ms；对应下降约为 53.7% 和 50.4%。全 5000 题含开销的累计降幅则为 41.33%。其他运行中的 55–60% 观察需要按相同口径比较。[原报告](../../results/joint/paper_test_n60_context_activation_restart_20260913_093636/screen_report.md)

## 3. 直接针对时间的 Pareto formulation

### 3.1 定义最优 solve 时间前沿

对 setup 平均时间预算 s，定义

\[
F(s)=\inf\{V(z):z\in\mathcal Z,\ S(z)\le s\}.
\tag{6}
\]

可行集合为空时取 \(F(s)=+\infty\)。F 是不增函数，因为增加预算不会缩小可行集合。这不要求任意两组参数的 setup 和 solve 时间都反向变化。

### 命题 1：最优总时间的前沿表示

若存在有限成本的可行方案，则

\[
\boxed{J_\star=\inf_{s\ge0}[s+F(s)].}
\tag{7}
\]

**证明。** 对任意 z，取 \(s=S(z)\)，则 \(F(s)\le V(z)\)，故式 (7) 右侧不大于 \(J_\star\)。反之，对任意 F(s) 有限的 s 和任意 \(\varepsilon>0\)，存在方案满足 \(S(z)\le s\)、\(V(z)\le F(s)+\varepsilon\)，所以 \(J_\star\le s+F(s)+\varepsilon\)。令 \(\varepsilon\downarrow0\)，再取 infimum。∎

这把 setup–solve tradeoff 转成了明确的总时间最小化问题。式 (7) 自身不是数值下界，也不说明 F 已被数据识别；它给出了需要研究的对象。

### 3.2 多个不同配置可以有相同的最优总时间

考虑以下纯示意时间，均除以同一默认总时间；不是实验数据：

| 配置 | Setup 部分 S | Solve 部分 V | 总时间 | 时间下降 |
|---|---:|---:|---:|---:|
| A | 0.16 | 0.26 | 0.42 | 58% |
| B | 0.21 | 0.21 | 0.42 | 58% |
| C | 0.27 | 0.15 | 0.42 | 58% |

这些配置不需要相同的 hierarchy、相同的 cycle 数或相同的收敛行为。总时间相同可以完全来自不同成本分项之间的补偿。

要进一步称其为全局最优，需要证明所有允许方案都满足 \(S+V\ge0.42\)。没有这一步，另一个未发现的配置仍可能只用 0.35 的总时间。

### 命题 2：非唯一时间最优性的一个充分条件

若存在 \(J_0>0\)，使全部可行预算满足

\[
s+F(s)\ge J_0,
\]

并有多个不同可行方案 \(z_j\) 达到

\[
S(z_j)=s_j,\qquad V(z_j)=F(s_j)=J_0-s_j,
\]

则 \(J_\star=J_0\)，且这些方案都是总时间 minimizer。

**证明。** 命题 1 给出 \(J_\star\ge J_0\)，任意一个达到等式的方案给出反向不等式。每个 \(z_j\) 的总时间均为 \(J_0\)。∎

若等式改成 \(0\le s_j+F(s_j)-J_0\le\zeta\)，并仍有全局下界 J₀，则这些方案距离最优总时间至多 \(\zeta\)。不必证明多个精确 minimizer，便可得到有实际意义的近等效性能集合。

### 3.3 “共同时间平台”的局部解释

如果用一个可微的局部连续模型近似前沿，且 s* 为内部极小点，则必要条件是

\[
1+F'(s_\star)=0.
\tag{8}
\]

其单位解释是：增加 1 ms 的 setup 恰好换取约 1 ms 的 solve 节省时，总时间的一阶边际改善消失。若一段前沿近似满足 \(F(s)\approx J_0-s\)，就会出现总时间近似恒定的多个成本分配。

实际参数含类别、离散网格和失败恢复分支，F 未必可微。式 (8) 是局部模型的解释，不能不经验证就当作真实前沿的光滑性定理。

另外，即使最优的时间分配 \((S,V)\) 唯一，也可以有多组不同参数映射到同一个时间分配。因此“参数不唯一”和“最优时间分配不唯一”也是不同现象。

## 4. 实验前沿为什么还不能证明硬上限

令 \(\mathcal Z_{\rm tested}\subseteq\mathcal Z\) 为已测试的方案。即使它们的真实均值已知，基于这些方案得到的前沿仍满足

\[
F_{\rm tested}(s)\ge F(s),
\qquad
J_{\rm best,tested}\ge J_\star.
\tag{9}
\]

因此，已达到的最大时间下降是整体可达到最大下降的**下界**，并非上界。很多方法都停在约 58%，说明 58% 已经可实现并且可能有平台；它本身不能证明 65% 不可实现。

观测带有计时噪声时，一次最小时间还可能过于乐观，需要先估计候选的可重复性能。不同配置若分别在不同难度的问题子集上被选择，直接比较其日志均值也可能不公平。

存在最优值与识别最优值也不同。在有限离散方案集合中，若每个方案有良定义的有限平均成本，则最小值存在；这通常容易证明，既没有确定这个值，也没有证明当前方法接近它。

## 5. 不枚举整个空间，也能怎样证明加速上限

### 定理 3：全局时间下界与候选共同构成最优性证书

假设能得到对所有可行预算都有效的前沿下界

\[
F(s)\ge g(s).
\]

令

\[
L=\inf_{s\ge0}[s+g(s)],
\]

并设某个可实现方案的真实平均总时间不超过 U。则

\[
\boxed{L\le J_\star\le U,}
\qquad
\boxed{1-U/J_{\rm default}\le R_\star\le1-L/J_{\rm default}.}
\tag{10}
\]

该候选的最优性 gap 至多为 U−L。若有多个不同配置都达到 U 附近，而 U−L 很小，就可以证明它们同时接近同一个最优总时间。

**证明。** 命题 1 和 \(F\ge g\) 给出下界；可实现候选给出上界。其余由代数和 \(J_{\rm default}>0\) 得到。∎

例如，若将来证明 \(L=0.42J_{\rm default}\)，则最大降幅不超过 58%；若还存在成本不超过 \(0.43J_{\rm default}\) 的候选，就把最大降幅夹在 57–58%。这些数字仅说明需要什么证书，不是已有结论。

也可以不经由 g，直接证明所有方案的总时间下界 L。核心是这个下界覆盖未测试配置，而不是再次搜索出若干接近的好配置。

### 真正困难的部分是 g 或 L 从哪里来

一条路线是分析在当前允许的 AMG 实现中不可避免的计算和数据移动，再结合硬件的有效吞吐上界。若必须执行至少 \(N_{\rm op}\) 次指定操作、跨某存储层移动至少 Q 字节，且对应吞吐上界为 \(P_{\max},B_{\max}\)，则

\[
T\ge\max\{N_{\rm op}/P_{\max},Q/B_{\max}\}.
\tag{11}
\]

这个方向与 Roofline 的计算/带宽性能限制有关，但要得到数学下界，需要真正有效的必要工作量和吞吐上界；一次 benchmark 的最大观测速度并不自动提供这个保证。缓存复用也会改变必须经过 DRAM 的流量。[Williams、Waterman 与 Patterson，2008 技术报告](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf)

另一条路线是建立覆盖允许配置的成本约束 \(V(z)\ge g(S(z))\)。AMG 的层次结构、必要工作量和达到容差所需的过程可以作为推导工具，但终点必须是以时间计量、覆盖 sequential solve policy 的下界。仅有一个收敛因子界不能直接承担这个结论。

如果某部分已计入目标的工作在所有允许配置下都不能消除，其时间至少为 \(T_{\rm irr}\)，则总加速的时间下降不超过 \(1-T_{\rm irr}/J_{\rm default}\)。但某个已测试配置的最小 setup 或 solve 时间，不能当作所有配置的 \(T_{\rm irr}\)。

更一般地，若有经证明有效的全局成本模型误差界，也可以据其构造 L，而不必枚举全部配置。当前 LinUCB 的预测及其实际探索系数还不提供这样的证书。有限样本上拟合得好，也不等于未测试区域已有有效下界。

这些路线都可在数学上研究，但给出有用的紧界很难。证明 AMG 在合适条件下具有 O(n) 复杂度，也不会自动确定 n=60³ 时 55–60% 的常数比例。

## 6. 文献对实际时间 tradeoff 的支持与限制

De Sterck、Yang 和 Heys 研究 parallel AMG 的复杂度增长，提出更稀疏的 coarsening，并比较内存和实际总执行时间。这直接支持：优化目标不能只看迭代次数，减少 hierarchy 工作量可能改善完整运行时间。[De Sterck 等，2006](https://www.osti.gov/servlets/purl/883821)

Falgout 与 Schroder 研究 non-Galerkin coarse grids，通过减少粗层 stencil 和通信改善 AMG 性能，并考察 operator complexity、收敛及 work per digit。这支持以实际工作和达到容差的代价分析 tradeoff，不提供当前硬件上统一的百分比上限。[Falgout 与 Schroder，2014](https://epubs.siam.org/doi/10.1137/130931539)

Gahvari 等直接建立 AMG solve cycle 的运行性能模型，并在多种大规模并行平台上与实际测量比较。它提供了从实现及机器特征研究时间的相关方法，但不是针对当前 setup 与 solve 联合参数空间的全局时间下界。[Gahvari 等，2011](https://research.ibm.com/publications/modeling-the-performance-of-an-algebraic-multigrid-cycle-on-hpc-platforms)

Xu 与 Zikatanov 的 optimal-coarse-space 理论，以及 Brannick 等的 optimal interpolation 分析，解决的是固定条件下的数值近似与收敛目标。它们可能成为时间下界推导的组成部分，但其结论本身不是最优 setup+solve 时间，更没有直接确定本实验的加速上限。[Xu 与 Zikatanov，2017](https://arxiv.org/abs/1611.01917)，[Brannick 等，2018](https://epubs.siam.org/doi/10.1137/17M1123456)

因此，现有文献为“存在复杂度、求解过程与运行时间的权衡”提供依据；仍需为指定问题族、策略类和机器建立足够具体的成本界。

## 7. 对观测平台的三个候选解释

| 解释 | 时间上的含义 | 尚需区分的证据 |
|---|---|---|
| 接近结构或硬件下界 | 剩余工作在当前合同下难以进一步减少或执行得更快 | 必要工作与有效吞吐界 |
| 多组配置的成本补偿 | setup、solve 和开销不同，总和相近 | 同一组问题上的分项时间与联合配置比较 |
| 共同学习或搜索限制 | 方法共享候选范围、特征、探索或函数类，尚未发现更快区域 | 已有候选的冻结性能、覆盖和受限搜索范围 |

多个 seed 或多个方法都落在相近区间，使“时间平台值得解释”更可信，但不能排除共有的搜索限制。当前配置还保留非零探索下限和 learner/controller 开销；在线尾段时间不等于去除探索后的最佳固定部署时间。[实验配置](../../experiments/joint/solve_control/configs/paper_test_n60_context_activation.json)

现有交叉诊断显示多个不同性能水平的联合组合，支持联合适配的作用，但还不能证明这些组合都接近同一个全局时间 floor。

## 8. 对论文范围和后续验证的建议

当前可以严谨提出并讨论以下主张：在测试合同下，多组不同 learned configurations 具有相近的总时间，提示存在一组性能相近的配置及可能的时间平台。若没有式 (10) 的证书，不应直接写成“所有允许方法的最大加速为约 60%”。

最有针对性的下一步是先用已有记录固定 native/含开销、尾段/全程的口径，并整理不同配置的 setup 与 solve 时间分配。若需要新的验证，应选少量结构不同的完整候选，冻结后在相同的未用于选候选的问题上比较总时间及分项。计时顺序与重复应预先规定，避免把时段噪声当成平台。

为了解释实际成本，下一次有针对性的测量可以增加每层 \(n_\ell\)、\(\operatorname{nnz}(A_\ell)\)、\(\operatorname{nnz}(P_\ell)\) 与阶段时间；这些结构量用于解释时间，不取代时间指标。在本次检查的 Python 记录接口中，未发现完整逐层稀疏结构统计。

这种小规模验证可以区分部分平台机制，仍不能独自证明 global ceiling。证明硬上限的关键工作是构造有效且足够紧的 L；没有必要先穷举整个 configuration space，但不能绕过这一数学缺口。

上述前沿恒等式和条件性证书是组织问题的工具，本身不应被包装成本文的核心新贡献。实质性的理论贡献需要进一步导出与 AMG 结构及实现有关的有效成本界，或证明具有明确范围的近最优时间差距。

## 9. 时间平台与 activation 的联系

即使多个方法最终趋于共同的每题时间 \(J_\infty\)，有限问题流仍要支付到达该水平前的训练、探索和恢复成本。

例如，若两种方法的期望每题时间为

\[
\mathbb E C_t^{(1)}=J_\infty+a/\sqrt t,
\qquad
\mathbb E C_t^{(2)}=J_\infty+b/\sqrt t,
\]

最终速度相同，累计时间差仍约为 \(2(a-b)\sqrt T\)。这是示例，说明共同的最终时间平台并不消除启动时机或学习效率的价值。

当前 4D 结果还不能假设其最终 \(J_\infty\) 必然与其他分支相同。可靠性触发、有限预算总耗时和可能的时间 floor 应分别定义，再通过共同成本目标连接。

## 文献

1. Howard, S. R., Ramdas, A., McAuliffe, J., and Sekhon, J. “Time-uniform Chernoff bounds via nonnegative supermartingales.” Probability Surveys 17 (2020), 257–317. [预印本](https://arxiv.org/abs/1808.03204)。顺序误触发控制的概率工具。
2. De Sterck, H., Yang, U. M., and Heys, J. J. “Reducing Complexity in Parallel Algebraic Multigrid Preconditioners.” SIAM Journal on Matrix Analysis and Applications 27(4) (2006), 1019–1039. [机构预印本](https://www.osti.gov/servlets/purl/883821)。复杂度、内存和执行时间。
3. Falgout, R. D., and Schroder, J. B. “Non-Galerkin Coarse Grids for Algebraic Multigrid.” SIAM Journal on Scientific Computing 36(3) (2014), C309–C334. [期刊页面](https://epubs.siam.org/doi/10.1137/130931539)，[机构预印本](https://www.osti.gov/servlets/purl/1237540)。稀疏性、通信与求解成本。
4. Williams, S., Waterman, A., and Patterson, D. “Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures.” UC Berkeley Technical Report UCB/EECS-2008-134 (2008). [全文](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf)。硬件性能模型。
5. Xu, J., and Zikatanov, L. “Algebraic multigrid methods.” Acta Numerica 26 (2017), 591–721. [全文](https://arxiv.org/pdf/1611.01917)。相关数值理论背景，不是当前 wall-clock ceiling 的证明。
6. Brannick, J., Cao, F., Kahl, K., Falgout, R. D., and Hu, X. “Optimal Interpolation and Compatible Relaxation in Classical Algebraic Multigrid.” SIAM Journal on Scientific Computing 40(3) (2018), A1473–A1493. [期刊页面](https://epubs.siam.org/doi/10.1137/17M1123456)，[全文预印本](https://arxiv.org/pdf/1703.10240)。相关最优 interpolation 背景。
7. Gahvari, H., Baker, A. H., Schulz, M., Yang, U. M., Jordan, K. E., and Gropp, W. “Modeling the Performance of an Algebraic Multigrid Cycle on HPC Platforms.” ICS 2011, 172–181. [作者机构页面](https://research.ibm.com/publications/modeling-the-performance-of-an-algebraic-multigrid-cycle-on-hpc-platforms)。AMG cycle 的运行性能模型和跨平台测量。
