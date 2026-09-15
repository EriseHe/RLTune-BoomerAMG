# 理论审阅：two-grid 强化、可观测性与学习保证的边界

2026-09-15。审阅对象：ChatGPT 对话 `6aa8e844-5584-83ea-a609-bacc227316b3` 的全部三轮对话，以及当前 SISC 修订稿。

对话附件中的 `theory_updated.tex`、`appendix_updated.tex`、`solve_updated.tex`、`experiments_updated.tex` 与当前 Overleaf 修订目录中的对应文件逐字一致。本笔记是理论评估和可供后续修改采用的推导；未替换论文主稿，未运行 PDE 实验。

## 1. 结论与代码依据

原稿的 calibrated regret、渐近最坏情形比较、条件性 policy-quality、recursive=batch 和 activation 证明，在写明的模型条件下没有发现推翻结论的错误。讨论提出的参数化 two-grid 计算和两步消去也成立。但是，“由此形成实际 RL 的完整理论闭环”过强；尤其要区分策略可表示性和 Q 函数可表示性。

实际实现的几个约束：

- `solve/controllers/common/state_encoder.py` 的输入是 PDE 描述量、setup 描述量、残差标量及其变化、cycle 位置、前一个 weight 和 cycle time；不含残差向量或误差方向。两个 learner 的共享 context 是含截距的 4/7 维。
- `solve/controllers/sarsa/online_td_lambda.py` 在 episode 开始时设置 `residual = previous_residual = initial_residual`、`cycle = 0`、`last_cycle_time = 0`，并使用同一个初始环境权重。首次编码不区分相同系统和 setup 上的不同 RHS 方向。
- `recursive_lstdq/v1.py` 使用实际选定并执行的 next action 构造 TD 目标；`v3.py` 的 episode moments 在各次拟合后的不同参数处计算。数据并不是同一个固定策略的独立评价样本。
- 实验使用 41 个 weight、9 个 RBF，gamma=1、trace lambda=0.8、beta=2、LCB 下截断为 0，并保留 0.03 的随机探索下限。
- `SharedLinUCB_AMG_v4.py::_structured_candidate_subset` 混合 anchors、balanced-global、local、Sobol，并去重；它不是 512 次独立同分布抽样。

已有 [C/D 比较](../../results/diagnostics/paper_context_activation_seed_comparison_20260914/report.md) 支持完整方法的成本改善，但不证明相同 hierarchy 上胜过 fixed/periodic oracle。[已有冻结交叉诊断](context_activation_mechanism_diagnosis_20260913.md) 支持 setup–controller 的相互适配，也没有识别 residual feedback 相对预设周期的独立贡献。这些事实用于限定理论对象，不作为数学证明的前提。

## 2. 参数化 two-grid 结果：可直接采用

保留原稿的 A、P 和 exact Galerkin correction C。在 energy-orthonormal 坐标下，令

\[
v(w)=\frac1{\sqrt2}(1-w/3,1-w)^\top,
\qquad \widetilde E(w)=v(w)v(w)^\top.
\]

为避免与 advection 向量及 action basis 混淆，用 h 表示第二个 relaxation weight。定义

\[
q(w)=\|v(w)\|^2
=\frac{5w^2-12w+9}{9}
=\frac15+\frac59\left(w-\frac65\right)^2,
\qquad c_h=v(h)^\top v(1)=\frac{3-h}{9}.
\]

因此对每个固定 w，

\[
\|E(w)^n\|_A=q(w)^n\ge5^{-n}.
\tag{1}
\]

对 F_h=E(h)E(1)，rank-one 乘法直接给出

\[
\rho(F_h)=c_h^2,
\qquad
\|F_h^n\|_A=|c_h|^{2n-1}\sqrt{\frac{2q(h)}9},\quad n\ge1.
\tag{2}
\]

式 (2) 同时给出了有限次乘积的范数，不能把它混同于 rho(F_h)^n。它补齐了原稿中连续 fixed comparator 的显式统一比较。

在相同 cycle cost 下，h 属于 (6/5,3) 时，周期 (1,h) 的渐近每 cycle 因子为 (3-h)/9<1/5。h=3 是有限步消去的单独情形，不应直接代入要求正谱半径的渐近命题。

此外，q(h)<1 当且仅当 0<h<12/5。因此 h 属于 (6/5,12/5) 时，两种 cycle 各自都收敛，周期仍严格优于全部固定权重。例如 h=2 时，q(1)=2/9、q(2)=5/9，而 sqrt(rho(F_2))=1/9。

### 2.1 有限容差比较及其量词

有

\[
E(3)E(1)=E(1)E(3)=0.
\tag{3}
\]

对相对 energy-error tolerance 0<epsilon<1/25，以“对全部初始误差保证达到容差”的 cycle 数定义 N，得到

\[
N_{(1,3)}^{\rm wc}(\epsilon)=2,
\qquad
\min_{w\in[1,3]}N_w^{\rm wc}(\epsilon)
=\left\lceil\frac{\log(1/\epsilon)}{\log5}\right\rceil>2.
\tag{4}
\]

证明：式 (3) 给出两步上界；第一步 E(1) 的范数为 2/9，所以最坏情形不能一步完成。式 (1) 给出固定权重下界，w=6/5 对每个 n 取等号。

式 (4) 的比较对象是 min_w sup_e N_w(e)，不是 sup_e min_w N_w(e)，也不是任意给定初始分布下的期望成本。不能把不同量词的 comparator 直接互换。零初始残差直接结束；某些非零初始方向也可被一个固定 action 一步消去。

两步消去在任意残差范数下仍成立；固定权重的两步最坏情形下界也可由残差传播矩阵的谱半径得到。不过式 (4) 的精确 ceiling 公式按 energy norm 陈述，不应未经转换就写成任意范数的精确次数。

若将 cycle 数换成 wall-clock cost，需要保留实际成本假设。例如在 epsilon<1/25 时，tau(1)+tau(3)+额外控制开销 < 3 min_w tau(w) 是一个充分条件。cycle cap 与 recovery charge 也必须使用共同定义，不能把提前失败算作提前完成。

### 2.2 必须降低对 exact annihilation 的新颖性解释

这个例子还满足

\[
S(3)S(1)=(I-3A)(I-A)=I-4A+3A^2=0,
\tag{5}
\]

因为 A 的两个特征值是 1/3 和 1。于是

\[
E(3)E(1)=S(3)C\,[S(3)S(1)]\,CS(1)=0.
\]

这说明两步消去已经由 smoother 的二次消去多项式推出，甚至不依赖这里 C 的特殊选择。它是有用的完整 cycle 构造，但不能被解读为 coarse correction 产生了一种此前不存在的消去原理。

这个限制有直接文献依据：Yang–Mittal 2014 的 §3 已展示按特征模态选择权重、有限步精确消去的小型例子，§4 研究预设 over/under-relaxation 周期。[原论文](https://engineering.jhu.edu/fsag/wp-content/uploads/2013/10/JCP_revised_WebPost1.pdf)。2017 年的后续工作将两个或三个 relaxed-Jacobi iterations 用作 multigrid smoother。[作者机构记录](https://pure.johnshopkins.edu/en/publications/efficient-relaxed-jacobi-smoothers-for-multigrid-on-parallel-comp/)。

因此建议同时保留参数区间、两个 individually contractive actions 的例子，以及有限步特例；不要只把 2.9 换成 3 就宣称理论创新显著提高。

## 3. 一般 PSD/product-overlap 引理：正确的辅助解释

若 A 是 SPD，D 是正定对角阵，S_w=I-wD^{-1}A，且 C 是 exact Galerkin coarse correction，则

\[
AS_w=S_w^\top A,\qquad C^\top A=AC,\qquad C^2=C.
\]

故对 E_w=S_wCS_w，

\[
\langle x,E_wx\rangle_A
=\langle S_wx,CS_wx\rangle_A
=\|CS_wx\|_A^2\ge0.
\]

设 B_w=A^{1/2}E_wA^{-1/2}，则 B_w 对称半正定，且

\[
\rho(E_bE_a)
=\lambda_{\max}(B_a^{1/2}B_bB_a^{1/2}),
\qquad
\sqrt{\rho(E_bE_a)}=\|B_b^{1/2}B_a^{1/2}\|_2.
\tag{6}
\]

证明：AB 与 BA 的非零谱一致，将乘积转换为半正定 Gram 矩阵即可；奇异情形同样成立。rank-one 情形右侧化为 |v(b)^T v(a)|。

式 (6) 是标准线性代数在此对象上的应用，适合作为几何解释引理。它本身没有推出严格优势，也没有分析学习速度。对 diffusion–advection 的非对称 A、不同 pre/post smoothers 或任意 BoomerAMG hierarchy，不可未经验证地套用这些 SPD/自伴条件。

## 4. 一个更贴近 sequential objective 的补充：即时收缩最优也可能更慢

这是本轮从现有构造直接推导的结果，不依赖新的实验假设；不声称其一般思想是首次提出。

**命题。** 在上述二维模型、相同 cycle cost 下，令

\[
z_\star=\sqrt5\,v(6/5)=\frac{(3,-1)^\top}{\sqrt{10}}.
\]

每一步精确最小化当前 energy-error norm 的策略

\[
w_k\in\arg\min_{w\in[1,3]}\|\widetilde E(w)z_k\|_2
\]

从 z_0=z_star 出发，将始终唯一选择 w=6/5，并满足 z_k=5^{-k}z_star。周期 (1,3) 则两步完成。因此对 epsilon<1/25，one-step-greedy control 严格慢于该两步序列。

**证明。** 对每个 w，

\[
v(w)^\top z_\star=1/\sqrt5,
\qquad
\|\widetilde E(w)z_\star\|_2^2=q(w)/5.
\]

q(w) 唯一最小于 6/5，并且 E_tilde(6/5)z_star=z_star/5。归纳即得所有后续选择；再用式 (3)。三个使用的 action 1、6/5、3 都在实际 41 点网格上。

例如 epsilon=10^-6 时，这个理想模型中 greedy 需要 9 cycles，周期需要 2 cycles；这是公式的精确后果，不是实际 60³ 的计时结果。

这个初始方向还可以消除有限成本比较中的量词缺口。对每个固定 w 和 n≥1，

\[
\|\widetilde E(w)^n z_\star\|_2
=q(w)^{n-1/2}/\sqrt5\ge5^{-n}.
\]

因此在这个同一个初始误差上，最优固定权重也需要 ceiling(log(1/epsilon)/log5) 个 cycles。选取初始分布为 z_star 处的点质量、单位 cycle cost、足够大的共同 cap，并取 reference 为先 1 后 3，即可给出明确的 finite-horizon expected-cost gap。例如 epsilon=10^-6、cap≥9 时该 gap 为 7。这个明确分布下的结论可作为 conditional policy-quality theorem 的一个数值模型实例，但不自动推广到实际 PDE 分布。

该命题直接说明“最大即时收缩”不等于“最小完成成本”，比单凭 rho(E(2.9))>1 来推测 myopic controller 不会选择大权重更严格。它支持考虑 future cost 的方法动机，也仍然允许一个预设两步 scheduler 达到优势；它不证明 LSTDQ 是唯一或最优的学习算法。这里的即时准则是 energy-error norm，不声称已证明实际 Euclidean-residual greedy baseline 具有相同轨迹。

## 5. 真正未闭合的环节：policy representability 不等于 Q representability

原稿的 endpoint-RBF 证明足以构造一个 greedy score 实现 1↔3，且式 (3) 的两个顺序都成立。因此可以对齐 witness 与 representability 的动作。

但是这只是在证明存在一个评分函数产生所需动作，不是在证明该评分函数近似 Q^{pi_ref}，更不说明 LSTDQ 的 estimating equations 会收敛到它。

### 5.1 观测编码造成的不可消除误差

对任意编码 h 和由它计算的估计 Q_hat(h(s),w)，若 h(s_A)=h(s_B)，则

\[
\sup_{s,w}|\widehat Q(h(s),w)-Q^{\pi_{\rm ref}}(s,w)|
\ge\frac12|Q^{\pi_{\rm ref}}(s_A,w)-Q^{\pi_{\rm ref}}(s_B,w)|.
\tag{7}
\]

证明：同一个预测值到两个不同真值的最大距离至少是两真值距离的一半。这对任意函数类都成立，不只是线性 RBF。

现在将二维模型的初始误差取为 energy 坐标轴 e_1、e_2，设置相同 system/context/setup、初始归一化残差、cycle、last_weight 和空历史。现有类型的编码对它们相同。单位 cycle cost、共同 cap 至少为 2、epsilon<2/9，以 1↔3 为 reference continuation，有

| 初始误差 | Q_ref(s,1) | Q_ref(s,3) |
|---|---:|---:|
| e_1 | 2 | 1 |
| e_2 | 1 | 2 |

理由是 E_tilde(1)=diag(2/9,0)，E_tilde(3)=diag(0,2)。一个 action 未结束时，reference 的相反 endpoint action 在下一步消去误差。

如果要求 uniform guarantee 的 relevant states 包含这两个初始状态，则由 (7)，uniform Q error 至少为 1/2 个 cycle cost。这是结构性信息损失，不会仅因训练样本增多而消失。它不是对真实 60³ Q error 的数值估计，也不排除在更窄的状态分布上得到更小的误差。

例如让初始分布对这两个状态均赋予正概率，再按讨论建议使用 K_max=50 和 2K_max epsilon_Q+selection_error 的界，其第一项至少为 50。对于上述 energy-norm toy、epsilon=10^-6，固定 w=6/5 对任意初始误差至多需要 9 个 cycles，reference 至少需要 1 个 cycle，因此这个相同初始分布下的 expected-cost gap 至多为 8；该充分条件无法成立。因而“finite-step witness + 实际 feature class + 现有 uniform-Q transfer bound”并未自动构成非空的实现保证。

这不推翻原稿的条件性命题，也不证明编码后的策略表现不好：一个不区分这两个初始方向的周期策略仍可在两步内解决二者。它说明“策略可以很好”和“完整状态 Q 可以被统一精确拟合”是不同要求。

### 5.2 更合适的误差组织方式

先保留 performance-difference identity。定义

\[
\delta_{\widehat\pi}^{\rm ref}(s)
=\mathbb E_{w\sim\widehat\pi(\cdot\mid s)}Q^{\pi_{\rm ref}}(s,w)
-V^{\pi_{\rm ref}}(s).
\]

在共同 finite horizon、terminal convention 下，

\[
J(\widehat\pi)-J(\pi_{\rm ref})
=\mathbb E_{\widehat\pi}\sum_k\delta_{\widehat\pi}^{\rm ref}(s_k).
\tag{8}
\]

它允许先分析沿实际访问状态的 action disadvantage，再把 uniform-Q bound 当作一个较强的充分条件。令 e(s,w)=Q_hat(h(s),w)-Q_ref(s,w)，并记 osc_w e=max_w e-min_w e。如果近似 greedy 的 selection deficit 为 delta_sel(s)，则

\[
\delta_{\widehat\pi}^{\rm ref}(s)
\le\operatorname{osc}_w e(s,w)+\delta_{\rm sel}(s).
\tag{9}
\]

证明：将 Q_ref=Q_hat-e 代入，使用 E_{pi_hat}Q_hat≤min_w Q_hat+delta_sel≤E_{pi_ref}Q_hat+delta_sel，剩余两个 e 的均值之差不超过其振幅。

式 (9) 消去了不影响 action ranking 的 state-only value offset。用 osc e≤2 epsilon_Q 即恢复原稿结果。它仍需要控制真实 action-value 差异，不是新的 LSTDQ 收敛保证。

实现的 optimistic clipping 和探索可单独列明：固定状态下令 q_w=Q_hat(h(s),w)，ell_w=max(q_w-beta u_w,0)，w_ell 为实际 tie-breaking 选中的 minimizer，探索概率为 epsilon_rand。均匀探索下的 selection deficit 恰为

\[
\delta_{\rm sel}(s)
=(1-\epsilon_{\rm rand})(q_{w_\ell}-\min_w q_w)
+\epsilon_{\rm rand}\left(\frac1{|\mathcal W|}\sum_wq_w-\min_wq_w\right).
\tag{10}
\]

它是估计评分内部的确定性量，不是统计 confidence coverage；未知的真实估计误差仍留在 (9)。controller overhead 若不在 Q 的成本中，还须独立计入总成本比较。

## 6. “反馈严格优于预设周期”不是可以直接强加的主线

讨论正确区分了 fixed、periodic、feedback。但是下列加强需要谨慎：

1. 两步周期只是一类 open-loop 控制。超过最优两周期可能来自更长的预设序列、problem-context-dependent scheduling 或不同搜索预算；它本身不单独证明 residual feedback 的贡献。
2. 对确定性模型和已知初始状态，任意确定性 feedback policy 的实际动作可预先展开成一条 open-loop 序列。对该初始状态，两者轨迹与成本相同。若允许 arbitrary initial-state-dependent open-loop oracle，则最优 feedback 不可能严格胜过它。
3. 若希望证明 feedback 的信息价值，必须明确 comparator 在何时知道哪些信息，并保证需要区分的状态在 controller 的实际 observation 中可区分。仅给两个 error directions 起名字并让 oracle 读取它们，不是在分析当前编码。
4. 一条 feedback superiority theorem 也分析的是控制策略类，不能仅因此称作 RL-specific：手工构造的反馈控制器可能有同样性质。学习方法还要解释数据来源、辨识和累计学习成本。
5. 一个在线 learner 不需要严格击败一个免计全部搜索成本的事后 oracle 才有价值。有效找到接近该 oracle 的策略、且付出较小累计成本，本身就是可研究的目标。

所以不建议把“必须证明 learned feedback > best periodic”替换成当前论文的证明目标。若后续研究这一方向，先定义公平的信息集合和有限预算，而不是从必须获胜的结论倒推例子。

## 7. Setup 与 payback 的建议：接受公式，限制解释

### 7.1 Candidate approximation

在相同 mu_t 和 C_t⊆U 下，定义

\[
\Delta_t^{\rm cand}=\min_{a\in\mathcal C_t}\mu_t(a)-\min_{a\in\mathcal U}\mu_t(a)\ge0.
\]

则 R_T^{full}=R_T^{cand}+sum_t Delta_t^{cand} 是精确恒等式，适合用几行加入现有 setup subsection。它明确了 candidate learner 与 candidate generator 的不同误差来源，不是整个 setup 空间的 regret 保证。

独立 proposals 下的 (1-p_epsilon)^m 是正确的抽样公式，但不可令 m=512 直接解释当前 structured512。更一般地，如果每次 proposal 在给定此前历史和未命中事件后，命中 near-optimal set 的条件概率至少为 p_j，则未命中概率至多 product_j(1-p_j)。当前 categorical cycling、分层 global sampling、Sobol 和 adaptive anchors 需要单独核对这些概率条件；目前没有对实际 near-optimal set 的质量下界。

原稿已经写明 candidate set 在当前 cost 之前已知，并对包含当前信息和所选 action 的条件分布施加 sub-Gaussian 假设。可以再明确 filtration 的 measurable 条件，但这不是原稿一个尚未处理的重大漏洞。

### 7.2 有限预算回本

令 B_T 为原 regret theorem 的显式上界，则在其 confidence event 上，

\[
\mathrm{Savings}_{\rm mean}(T)\ge G_T-\mathcal O_T-B_T.
\tag{11}
\]

若 G_T≥gT，O_T≤c_0+oT，且 (g-o)T>c_0+B_T，则获得该预算下的 modeled conditional-mean savings 正值。这比只写“充分大的 T”有用，适合替换现有 corollary 的主要陈述。

不过 (11) 不自动是可从日志计算的 certificate：G_T、realizability/noise 常数和未计入模型的 overhead 需要有依据；实际 realized savings 还需要噪声集中。当前 decaying-alpha joint learner 不因这个代入而继承 calibrated bound。

### 7.3 与实际联合学习更相关的成本变化

已有 [成本笔记](cost_based_rl_activation_framework.md) 给出

\[
\widehat\theta_t-\theta_t
=V_t^{-1}\left[\sum_{i<t}\phi_i\eta_i-\lambda\theta_t
+\sum_{i<t}\phi_i\phi_i^\top(\theta_i-\theta_t)\right].
\tag{12}
\]

这在时变线性成本模型下准确显示旧目标与当前目标的差异。设最后一项括号内的 V_t^{-1} 范数为 D_t，noise+ridge 的界为 beta_t，则预测误差受 (beta_t+D_t)sigma_t(a) 控制。将原 coverage-deficit 公式的 beta_t 换为 beta_t+D_t，可保留实际 alpha_t 的欠覆盖项。

这是有意义的误差记账，能够解释 fixed-downstream 保证为何不能直接搬到 joint phase。但在没有从更新机制进一步控制 D_t 时，它不是次线性 joint regret theorem。已有交叉数据支持研究这种耦合，不等于已经测出 D_t 或证明它小。

## 8. 建议的稿件修改次序

1. **直接加强现有 two-grid 命题**：参数区间、显式范数、有限容差比较，并精确注明 minimax comparator；不另堆多个“大定理”。
2. **加入短的 myopic-contraction 推论**：说明为何 completion cost 需要考虑后续动作；证明放 Appendix A。
3. **保留 representability 作为辅助结果**：与 1↔3 对齐，但不把它当 Q realizability。
4. **整理 policy-quality subsection**：以 (8) 为桥，说明 observation aliasing、(9) 的误差范围和实现的 (10)；保留 conditional 定位。
5. **简洁增强 setup accounting**：candidate gap、finite-budget payback；对持续联合学习明确模型变化项，避免伪装成已完成 joint regret 分析。
6. **PSD overlap 引理可放附录**，作为机制解释，不单独标榜新数值分析定理。
7. **保持 activation 在 Appendix B**；不重新扩展硬加速上限、全空间最优性或新增实验计划。

这会增加具体且可检查的数学内容，同时保留论文的 setup–solve 共同成本主线。它仍不同于从数值结构一直推出学习 regret 的完整保证。[Learning to Relax](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8fcc228e94aa7e4773a27c6c2d886243-Abstract-Conference.html) 的相应保证针对其明确的 SOR/SSOR 结构和 comparator；不能由一个二维 example 自动达到相同理论完成度。

## 核查记录

- 读取完整三轮对话和附件，核对上述四个 LaTeX 文件与当前稿一致。
- 逐项检查原稿 theorem/proof、state encoder、LSTDQ sampled-next-action update、clipped optimistic score、候选组成和已有机制报告。
- 用精确符号代数核对 q(w)、q(w)-1/5 的平方表示、S(3)S(1)、E(3)E(1)、pair trace/determinant、myopic 恒等式；证明逻辑如上，不以数值抽样代替证明。
- 本轮未运行 PDE 求解、未改 controller 或实验配置、未修改 Overleaf 主稿、未 commit/push。
