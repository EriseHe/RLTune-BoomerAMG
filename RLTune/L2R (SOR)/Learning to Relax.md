
# Learning to Relax: Setting Solver Parameters Across a Sequence of Linear System Instances 

Mikhail Khodak<br>CMU<br>khodak@cmu.edu

Edmond Chow<br>Georgia Tech.<br>echow@cc.gatech.edu

Maria-Florina Balcan<br>CMU<br>ninamf@cs.cmu.edu

Ameet Talwalkar<br>CMU<br>talwalkar@cmu.edu


#### Abstract

Solving a linear system $\mathbf{A x}=\mathbf{b}$ is a fundamental scientific computing primitive for which numerous solvers and preconditioners have been developed. These come with parameters whose optimal values depend on the system being solved and are often impossible or too expensive to identify; thus in practice sub-optimal heuristics are used. We consider the common setting in which many related linear systems need to be solved, e.g. during a single numerical simulation. In this scenario, can we sequentially choose parameters that attain a near-optimal overall number of iterations, without extra matrix computations? We answer in the affirmative for Successive Over-Relaxation (SOR), a standard solver whose parameter $\omega$ has a strong impact on its runtime. For this method, we prove that a bandit online learning algorithm-using only the number of iterations as feedback-can select parameters for a sequence of instances such that the overall cost approaches that of the best fixed $\omega$ as the sequence length increases. Furthermore, when given additional structural information, we show that a contextual bandit method asymptotically achieves the performance of the instance-optimal policy, which selects the best $\omega$ for each instance. Our work provides the first learning-theoretic treatment of high-precision linear system solvers and the first end-to-end guarantees for data-driven scientific computing, demonstrating theoretically the potential to speed up numerical methods using well-understood learning algorithms.


## 1 Introduction

The bottleneck subroutine in many scientific computations is a solver returning an approximate solution to a linear system. For example, simulating a partial differential equation (PDE) often involves solving sequences of high-dimensional systems to very high precision (Thomas, 1999). A vast array of solvers and preconditioners have thus been developed, many of which have tunable parameters that significantly affect runtime (Greenbaum, 1997; Hackbusch, 2016). There is a long literature analyzing these algorithms, and indeed for some problems we have a strong understanding of the optimal parameters for a given matrix. However, computing them can be more costly than solving the original system, leading to an assortment of heuristics for setting good parameters (Ehrlich, 1981; Golub \& Ye, 1999).
We provide an alternative to such heuristics by taking advantage of the fact that we often sequentially solve many linear systems. In addition to numerical simulation, this occurs in graphics computations such as mean-curvature flow (Kazhdan et al., 2012), nonlinear system solvers (Marquardt, 1963), and beyond. A natural approach is to treat these instances as data to be passed to a machine learning (ML) algorithm; in particular the framework of online learning (Cesa-Bianchi \& Lugosi, 2006) provides a language to reason about such sequential learning problems. For example, if we otherwise would solve a sequence of linear systems $\left(\mathbf{A}_{1}, \mathbf{b}_{1}\right), \ldots,\left(\mathbf{A}_{T}, \mathbf{b}_{T}\right)$ using a given solver with a fixed parameter, can we use ML to do as well as the best choice of that parameter, i.e. can we minimize regret? Or, if the matrices are all diagonal shifts of single matrix $\mathbf{A}$, can we learn the functional relationship between the shift $c_{t}$ and the optimal solver parameter for $\mathbf{A}_{t}=\mathbf{A}+c_{t} \mathbf{I}_{n}$, i.e. can we predict using context?
We investigate these questions for the Successive Over-Relaxation (SOR) solver, a generalization of Gauss-Seidel whose relaxation parameter $\omega \in(0,2)$ dramatically affects the number of iterations (c.f. Figure 1, noting the log-scale). SOR and its symmetric variant are well-studied and often used as preconditioners for Krylov methods such as conjugate gradient (CG), as bases for semi-iterative schemes, and as multigrid smoothers. Analogous to some past setups in data-driven algorithms (Balcan et al., 2018; Khodak et al., 2022), we sequentially set the parameter $\omega_{t}$ for SOR to use when solving each lin-
ear system ( $\mathbf{A}_{t}, \mathbf{b}_{t}$ ). Unlike past theoretical studies of related methods (Gupta \& Roughgarden, 2017; Bartlett et al., 2022; Balcan et al., 2022), we aim to provide end-to-end guarantees-covering the full pipeline from data-intake to efficient learning to execution-while minimizing dependence on the dimension ( $n$ can be $10^{5}$ or higher) and precision ( $1 / \varepsilon$ can be $10^{8}$ or higher). We emphasize that we do not seek to immediately improve the empirical state of the art, and also that existing research on saving computation when solving sequences of linear systems (recycling Krylov subspaces, reusing preconditioners, etc.) is complementary to our own, i.e. it can be used in addition to the ideas presented here.

### 1.1 Core contributions

We study two distinct theoretical settings, corresponding to views on the problem from two different approaches to data-driven algorithms. In the first we have a deterministic sequence of instances and study the spectral radius of the iteration matrix, the main quantity of interest in classical analysis of SOR (Young, 1971). We show how to convert its asymptotic guarantee into a surrogate loss that upper bounds the number of iterations via a quality measure of the chosen parameter, in the style of algorithms with predictions (Mitzenmacher \& Vassilvitskii, 2021). The bound holds under a nearasymptotic condition implying that convergence occurs near the asymptotic regime, i.e. when the spectral radius of the iteration matrix governs the convergence. We verify the assumption and show that one can learn the surrogate losses using only bandit feedback from the original costs; notably, despite being non-Lipschitz, we take advantage of the losses' unimodal structure to match the optimal $\tilde{\mathcal{O}}\left(T^{2 / 3}\right)$ regret for Lipschitz bandits (Kleinberg, 2004). Our bound also depends only logarithmically on the precision and not at all on the dimension. Furthermore, we extend to the diagonally shifted setting described before, showing that an efficient, albeit pessimistic, contextual bandit (CB) method has $\tilde{\mathcal{O}}\left(T^{3 / 4}\right)$ regret w.r.t. the instance-optimal policy that always picks the best $\omega_{t}$. Finally, we show a similar analysis of learning a relaxation parameter for the more popular (symmetric SOR-preconditioned) CG method.
Our second setting is semi-stochastic, with target vectors $\mathbf{b}_{t}$ drawn i.i.d. from a (radially truncated) Gaussian. This is a reasonable simplification, as convergence usually depends more strongly on $\mathbf{A}_{t}$, on which we make no extra assumptions. We show that the expected cost of running a symmetric variant of SOR (SSOR) is $\mathcal{O}(\sqrt{n})$ polylog $\left(\frac{n}{\varepsilon}\right)$-Lipschitz w.r.t. $\omega$, so we can (a) compete with the optimal number of iterations-rather than with the best upper bound-and (b) analyze more practical, regression-based CB algorithms (Foster \& Rakhlin, 2020; Simchi-Levi \& Xu, 2021). We then show $\tilde{\mathcal{O}}\left(\sqrt[3]{T^{2} \sqrt{n}}\right)$ regret when comparing to the single best $\omega$ and $\tilde{\mathcal{O}}\left(T^{9 / 11} \sqrt{n}\right)$ regret w.r.t. the instance-optimal policy in the diagonally shifted setting using a novel, Chebyshev regression-based CB algorithm. While the results do depend on the dimension $n$, the dependence is much weaker than that of past work on data-driven tuning of a related regression problem (Balcan et al., 2022).
Remark 1.1. Likely the most popular algorithms for linear systems are Krylov subspace methods such as CG. While an eventual aim of our line of work is to understand how to tune (many) parameters of (preconditioned) CG and other algorithms, SOR is a well-studied method and serves as a meaningful starting point. In fact, we show that our near-asymptotic analysis extends directly, and in the semi-stochastic setting there is a natural path to (e.g.) SSOR-preconditioned CG, as it can be viewed as computing polynomials of iteration matrices where SSOR just takes powers. Lastly, apart from its use as a preconditioner and smoother, SOR is still sometimes preferred for direct use as well (Fried \& Metzler, 1978; Van Vleck \& Dwyer, 1985; King et al., 1987; Woźnicki, 1993; 2001).

### 1.2 Technical and theoretical contributions

By studying a scientific computing problem through the lens of data-driven algorithms and online learning, we also make the following contributions to the latter two fields:

1. Ours is the first head-to-head comparison of two leading theoretical approaches to data-driven algorithms applied to the same problem. While the algorithms with predictions approach in Section 2 takes better advantage of the scientific computing literature to obtain (arguably) more interpretable and dimension-independent bounds, data-driven algorithm design (Balcan, 2021) competes directly with the quantity of interest in Section 3 and enables guarantees for modern CB algorithms.
2. For algorithms with predictions, our near-asymptotic approach may be extendable to other iterative solvers, as we demonstrate with CG. We also show that such performance bounds on a (partiallyobservable) cost are learnable even when the bounds themselves are too expensive to compute.
3. In data-driven algorithm design, we take the novel theoretical approach of proving continuity of the expectation of a discrete cost, rather than showing dispersion of its discontinuities (Balcan et al., 2018) or bounding predicate complexity (Bartlett et al., 2022).
```
Algorithm 1: Successive over-relaxation (SOR) with a relative convergence condition.
Input: $\mathbf{A} \in \mathbb{R}^{n \times n}, \mathbf{b} \in \mathbb{R}^{n}$, parameter $\omega \in(0,2)$, initial vector $\mathbf{x} \in \mathbb{R}^{n}$, tolerance $\varepsilon>0$
$\mathbf{D}+\mathbf{L}+\mathbf{L}^{T} \leftarrow \mathbf{A} / / \mathbf{D}$ is diagonal, $\mathbf{L}$ is strictly lower triangular
$\mathbf{W}_{\omega} \leftarrow \mathbf{D} / \omega+\mathbf{L} \quad / /$ compute the third normal form
$\mathbf{r}_{0} \leftarrow \mathbf{b}-\mathbf{A} \mathbf{x} \quad / /$ compute initial residual
for $k=0, \ldots$ do
    if $\left\|\mathbf{r}_{k}\right\|_{2} \leq \varepsilon\left\|\mathbf{r}_{0}\right\|_{2}$ then
        return $k$ // return iteration count (for use in learning)
    $\mathbf{x}=\mathbf{x}+\mathbf{W}_{\omega}^{-1} \mathbf{r}_{k} / /$ update vector after solving triangular system
    $\mathbf{r}_{k+1} \leftarrow \mathbf{b}-\mathbf{A} \mathbf{x} \quad / /$ compute the next residual
```

4. We introduce the idea of using CB to set instance-adaptive algorithmic parameters; while (linear) instance-adaptivity was also shown via convexity by Khodak et al. (2022), we go further by taking advantage of multi-instance structure to asymptotically do as well as the instance-optimal policy.
5. We show that standard discretization-based bandit algorithms are optimal for sequences of adversarially chosen semi-Lipschitz losses that generalize regular Lipschitz functions (c.f. Appendix B).
6. We introduce a new CB method that combines SquareCB (Foster \& Rakhlin, 2020) with Chebyshev polynomial regression to get sublinear regret on Lipschitz losses (c.f. Appendix C).

### 1.3 Related work and comparisons

We discuss the existing literature on solving sequences of linear systems (Parks et al., 2006; Tebbens \& Tůma, 2007; Elbouyahyaoui et al., 2021), work integrating ML with scientific computing to amortize cost (Amos, 2023; Arisaka \& Li, 2023), and past theoretical studies of data-driven algorithms (Gupta \& Roughgarden, 2017; Balcan et al., 2022) in Appendix A. For the latter we include a detailed comparison of the generalization implications of our work with the GJ framework (Bartlett et al., 2022). Lastly, we address the baseline of approximating the spectral radius of the Jacobi iteration matrix.

## 2 Asymptotic analysis of learning the relaxation parameter

We start this section by going over the problem setup and the SOR solver. Then we consider the asymptotic analysis of the method to derive a reasonable performance upper bound to target as a surrogate loss for the true cost function. Finally, we prove and analyze online learning guarantees.

### 2.1 Setup

At each step $t=1, \ldots, T$ of (say) a numerical simulation we get a linear system instance, defined by a matrix-vector pair $\left(\mathbf{A}_{t}, \mathbf{b}_{t}\right) \in \mathbb{R}^{n \times n} \times \mathbb{R}^{n}$, and are asked for a vector $\mathbf{x} \in \mathbb{R}^{n}$ such that the norm of its residual or defect $\mathbf{r}=\mathbf{b}_{t}-\mathbf{A}_{t} \mathbf{x}$ is small. For now we define "small" in a relative sense, specifically $\left\|\mathbf{A}_{t} \mathbf{x}-\mathbf{b}_{t}\right\|_{2} \leq \varepsilon\left\|\mathbf{b}_{t}\right\|_{2}$ for some tolerance $\varepsilon \in(0,1)$; note that when using an iterative method initialized at $\mathbf{x}=\mathbf{0}_{n}$ this corresponds to reducing the residual by a factor $1 / \varepsilon$, which we call the precision. In applications it can be quite high, and so we will show results whose dependence on it is at worst logarithmic. To make the analysis tractable, we make two assumptions (for now) about the matrices A: they are symmetric positive-definite and consistently-ordered (c.f. Hackbusch (2016, Definition 4.23)). We emphasize that, while not necessary for convergence, both are standard in the analysis of SOR (Young, 1971); see Hackbusch (2016, Criterion 4.24) for multiple settings where they holds.
To find a suitable $\mathbf{x}$ for each instance in the sequence we apply Algorithm 1 (SOR), which at a high-level works by multiplying the current residual $\mathbf{r}$ by the inverse of a matrix $\mathbf{W}_{\omega}$-derived from the diagonal $\mathbf{D}$ and lower-triangular component $\mathbf{L}$ of $\mathbf{A}$-and then adding the result to the current iterate $\mathbf{x}$. Note that multiplication by $\mathbf{W}_{\omega}^{-1}$ is efficient because $\mathbf{W}_{\omega}$ is triangular. We will measure the cost of this algorithm by the number of iterations it takes to reach convergence, which we denote by $\operatorname{SOR}(\mathbf{A}, \mathbf{b}, \omega)$, or $\operatorname{SOR}_{t}(\omega)$ for short when it is run on the instance $\left(\mathbf{A}_{t}, \mathbf{b}_{t}\right)$. For simplicity, we will assume that the algorithm is always initialized at $\mathbf{x}=\mathbf{0}_{n}$, and so the first residual is just $\mathbf{b}$.
Having specified the computational setting, we now turn to the learning objective, which is to sequentially set the parameters $\omega_{1}, \ldots, \omega_{T}$ so as to minimize the total number of iterations:

$$
\begin{equation*}
\sum_{t=1}^{T} \operatorname{SOR}_{t}\left(\omega_{t}\right)=\sum_{t=1}^{T} \operatorname{SOR}\left(\mathbf{A}_{t}, \mathbf{b}_{t}, \omega_{t}\right) \tag{1}
\end{equation*}
$$

To set $\omega_{t}$ at some time $t>1$, we allow the learning algorithm access to the costs $\operatorname{SOR}_{s}\left(\omega_{s}\right)$ incurred at the previous steps $s=1, \ldots, t-1$; in the literature on online learning this is referred to as the bandit or partial feedback setting, to distinguish from the (easier, but unreasonable for us) full information case where we have access to the cost function $\mathrm{SOR}_{s}$ at every $\omega$ in its domain.
Selecting the optimal $\omega_{t}$ using no information about $\mathbf{A}_{t}$ is impossible, so we must use a comparator to obtain an achievable measure of performance. In online learning this is done by comparing the total cost incurred (1) to the counterfactual cost had we used a single, best-in-hindsight $\omega$ at every timestep $t$. We take the minimum over some domain $\Omega \subset(0,2)$, as SOR diverges outside it. While in some settings we will compete with every $\omega \in(0,2)$, we will often algorithmically use $\left[1, \omega_{\text {max }}\right]$ for some $\omega_{\max }<2$. The upper limit ensures a bound on the number of iterations-required by bandit algorithms-and the lower limit excludes $\omega<1$, which is rarely used because theoretical convergence of vanilla SOR is worse there for realistic problems, e.g. those satisfying our assumptions.
This comparison-based approach for measuring performance is standard in online learning and effectively assumes a good $\omega \in \Omega$ that does well-enough on all problems; in Figure 1 (center-left) we show that this is sometimes the case. However, the center-right plot in the same figure shows we might do better by using additional knowledge about the instance; in online learning this is termed a context and there has been extensive development of contextual bandit algorithms that do as well as the best fixed policy mapping contexts to predictions. We will study an example of this in the diagonally shifted setting, in which $\mathbf{A}_{t}=\mathbf{A}+c_{t} \mathbf{I}_{n}$ for scalars $c_{t} \in \mathbb{R}$; while mathematically simple, this structure arises in natural settings, e.g. solving the heat equation with temporally variable diffusivity, and is well-motivated by other applications (Frommer \& Glässner, 1998; Bellavia et al., 2011; Baumann \& van Gijzen, 2015; Anzt et al., 2016; Wang et al., 2019). Furthermore, the same learning algorithms can also be extended to make use of other context information, e.g. rough spectral estimates.

### 2.2 Establishing a surrogate upper bound

Our first goal is to solve $T$ linear systems almost as fast as if we had used the best fixed $\omega \in \Omega$. In online learning, this corresponds to minimizing regret, which for cost functions $\ell_{t}: \Omega \mapsto \mathbb{R}$ is defined as

$$
\begin{equation*}
\operatorname{Regret}_{\Omega}\left(\left\{\ell_{t}\right\}_{t=1}^{T}\right)=\sum_{t=1}^{T} \ell_{t}\left(\omega_{t}\right)-\min _{\omega \in \Omega} \sum_{t=1}^{T} \ell_{t}(\omega) \tag{2}
\end{equation*}
$$

In particular, since we can upper-bound the objective (1) by $\operatorname{Regret}_{\Omega}\left(\left\{\operatorname{SOR}_{t}\right\}_{t=1}^{T}\right)$ plus the optimal cost $\min _{\omega \in \Omega} \sum_{t=1}^{T} \operatorname{SOR}_{t}(\omega)$, if we show that regret is sublinear in $T$ then the leading-order term in the upper bound corresponds to the cost incurred by the optimal fixed $\omega$.
Many algorithms attaining sublinear regret under different conditions on the losses $\ell_{t}$ have been developed (Cesa-Bianchi \& Lugosi, 2006; Bubeck \& Cesa-Bianchi, 2012). However, few handle losses with discontinuities-i.e. most algorithmic costs-and those that do (necessarily) need additional conditions on their locations (Balcan et al., 2018; 2020). At the same time, numerical analysis often deals more directly with continuous asymptotic surrogates for cost, such as convergence rates. Taking inspiration from this, and from the algorithms with predictions idea of deriving surrogate loss functions for algorithmic costs (Khodak et al., 2022), in this section we instead focus on finding upper bounds $U_{t}$ on $\mathrm{SOR}_{t}$ that are both (a) learnable and (b) reasonably tight in-practice. We can then aim for overall performance nearly as good as the optimal $\omega \in \Omega$ as measured by these upper bounds:

$$
\begin{equation*}
\sum_{t=1}^{T} \operatorname{SOR}_{t}\left(\omega_{t}\right) \leq \sum_{t=1}^{T} U_{t}\left(\omega_{t}\right)=\operatorname{Regret}_{\Omega}\left(\left\{U_{t}\right\}_{t=1}^{T}\right)+\min _{\omega \in \Omega} \sum_{t=1}^{T} U_{t}(\omega)=o(T)+\min _{\omega \in \Omega} \sum_{t=1}^{T} U_{t}(\omega) \tag{3}
\end{equation*}
$$

A natural approach to get a bound $U_{t}$ is via the defect reduction matrix $\mathbf{C}_{\omega}=\mathbf{I}_{n}-\mathbf{A}(\mathbf{D} / \omega+\mathbf{L})^{-1}$, so named because the residual at iteration $k$ is equal to $\mathbf{C}_{\omega}^{k} \mathbf{b}$ and $\mathbf{b}$ is the first residual. Under our assumptions on $\mathbf{A}$, Young (1971) shows that the spectral radius $\rho\left(\mathbf{C}_{\omega}\right)$ of $\mathbf{C}_{\omega}$ is a (nontrivial to compute) piecewise function of $\omega$ with a unique minimum in $[1,2)$. Since we have error $\left\|\mathbf{C}_{\omega}^{k} \mathbf{b}\right\|_{2} /\|\mathbf{b}\|_{2} \leq \left\|\mathbf{C}_{\omega}^{k}\right\|_{2}$ at iteration $k, \rho\left(\mathbf{C}_{\omega}\right)=\lim _{k \rightarrow \infty} \sqrt[k]{\left\|\mathbf{C}_{\omega}^{k}\right\|_{2}}$ asymptotically bounds how much the error is reduced at each step. It is thus often called the asymptotic convergence rate and the number of iterations is said to be roughly bounded by $\frac{-\log \varepsilon}{-\log \rho\left(\mathbf{C}_{\omega}\right)}$ (e.g. Hackbusch (2016, Equation 2.31b)). However, while it is tempting to use this as our upper bound $U$, in fact it may not upper bound the number of iterations at all, since $\mathbf{C}_{\omega}$ is not normal and so in-practice the iteration often goes through a transient phase where the residual norm first increases before decreasing (Trefethen \& Embree, 2005, Figure 25.6).
Thus we must either take a different approach or make some assumptions. Note that one can in-fact show an $\omega$-dependent, finite-time convergence bound for SOR via the energy norm (Hackbusch,

![](https://cdn.mathpix.com/cropped/2025_11_24_6122a0dad929fc569471g-05.jpg?height=263&width=1367&top_left_y=271&top_left_x=382)
Figure 1: Left: comparison of different cost estimates. Center-left: mean performance of different parameters across forty instances of form $\mathbf{A}+\frac{12 c-3}{20} \mathbf{I}_{n}$, where $c \sim \operatorname{Beta}(2,6)$. Center-right: the same but for $c \sim \operatorname{Beta}(1 / 2,3 / 2)$, which is relatively higher-variance. In both cases the dashed line indicates instance-optimal performance, the matrix $\mathbf{A}$ is a discrete Laplacian of a $100 \times 100$ square domain, and the targets $\mathbf{b}$ are truncated Gaussians. Right: asymptocity as measured by the difference between the spectral norm at iteration $k$ and the spectral radius, together with its upper bound $\tau\left(1-\rho\left(\mathbf{C}_{\omega}\right)\right)$.

2016, Corollary 3.45), but this can give rather loose upper bounds on the number of iterations (c.f. Figure 1 (left)). Instead, we make the following assumption, which roughly states that convergence always occurs near the asymptotic regime, where nearness is measured by a parameter $\tau \in(0,1)$ :
Assumption 2.1. There exists $\tau \in(0,1)$ s.t. $\forall \omega \in \Omega$ the matrix $\mathbf{C}_{\omega}=\mathbf{I}_{n}-\mathbf{A}(\mathbf{D} / \omega+\mathbf{L})^{-1}$ satisfies $\left\|\mathbf{C}_{\omega}^{k}\right\|_{2} \leq\left(\rho\left(\mathbf{C}_{\omega}\right)+\tau\left(1-\rho\left(\mathbf{C}_{\omega}\right)\right)\right)^{k}$ at $k=\min _{\left\|\mathbf{C}_{\omega}^{i+1} \mathbf{b}\right\|_{2}<\varepsilon\|\mathbf{b}\|_{2}} i$.
This effectively assumes an upper bound $\rho\left(\mathbf{C}_{\omega}\right)+\tau\left(1-\rho\left(\mathbf{C}_{\omega}\right)\right)$ on the empirically observed convergence rate, which gives us a measure of the quality of each parameter $\omega$ for the given instance $(\mathbf{A}, \mathbf{b})$. Note that the specific form of the surrogate convergence rate was chosen both because it is convenient mathematically-it is a convex combination of 1 and the asymptotic rate $\rho\left(\mathbf{C}_{\omega}\right)$-and because empirically we found the degree of "asymptocity" as measured by $\left\|\mathbf{C}_{\omega}^{k}\right\|_{2}^{1 / k}-\rho\left(\mathbf{C}_{\omega}\right)$ for $k$ right before convergence to vary reasonably similarly to a fraction of $1-\rho\left(\mathbf{C}_{\omega}\right)$ (c.f. Figure 1 (right)). This makes intuitive sense, as the parameters $\omega$ for which convergence is fastest have the least time to reach the asymptotic regime. Finally, note that since $\lim _{k \rightarrow \infty}\left\|\mathbf{C}_{\omega}^{k}\right\|_{2}^{1 / k}=\rho\left(\mathbf{C}_{\omega}\right)$, for every $\gamma>0$ there always exists $k^{\prime}$ s.t. $\left\|\mathbf{C}_{\omega}^{k}\right\|_{2} \leq\left(\rho\left(\mathbf{C}_{\omega}\right)+\gamma\right)^{k} \forall k \geq k^{\prime}$; therefore, since $1-\rho\left(\mathbf{C}_{\omega}\right)>0$, we view Assumption 2.1 not as a restriction on $\mathbf{C}_{\omega}$ (and thus on $\mathbf{A}$ ), but rather as an an assumption on $\varepsilon$ and $\mathbf{b}$. Specifically, the former should be small enough that $\mathbf{C}_{\omega}^{i}$ reaches that asymptotic regime for some $i$ before the criterion $\left\|\mathbf{C}_{\omega}^{k} \mathbf{b}\right\|_{2} \leq \varepsilon\|\mathbf{b}\|_{2}$ is met; for similar reasons, the latter should not happen to be an eigenvector corresponding to a tiny eigenvalue of $\mathbf{C}_{\omega}$ (c.f. Figure 2 (left)).
Having established this surrogate of the spectral radius, we can use it to obtain a reasonably tight upper bound $U$ on the cost (c.f. Figure 1 (left)). Crucially for learning, we can also establish the following properties via the functional form of $\rho\left(\mathbf{C}_{\omega}\right)$ derived by Young (1971):
Lemma 2.1. Define $U(\omega)=1+\frac{-\log \varepsilon}{-\log \left(\rho\left(\mathbf{C}_{\omega}\right)+\tau\left(1-\rho\left(\mathbf{C}_{\omega}\right)\right)\right)}$, $\alpha=\tau+(1-\tau) \max \left\{\beta^{2}\right.$, $\left.\omega_{\text {max }}-1\right\}$, and $\omega^{*}=1+\beta^{2} /\left(1+\sqrt{1-\beta^{2}}\right)^{2}$, where $\beta=\rho\left(\mathbf{I}_{n}-\mathbf{D}^{-1} \mathbf{A}\right)$. Then the following holds:

1. $U$ bounds the number of iterations and is itself bounded: $\operatorname{SOR}(\mathbf{A}, \mathbf{b}, \omega)<U(\omega) \leq 1+\frac{-\log \varepsilon}{-\log \alpha}$ 2. $U$ is decreasing towards $\omega^{*}$, and $\frac{-(1-\tau) \log \varepsilon}{\alpha \log ^{2} \alpha}$-Lipschitz on $\omega \geq \omega^{*}$ if $\tau \geq \frac{1}{e^{2}}$ or $\beta^{2} \geq \frac{4}{e^{2}}\left(1-\frac{1}{e^{2}}\right)$

Lemma 2.1 introduces a quantity $\alpha=\tau+(1-\tau) \max \left\{\beta^{2}, \omega_{\text {max }}-1\right\}$ that appears in the upper bounds on $U(\omega)$ and in its Lipschitz constant. This quantity will in some sense measure the difficulty of learning: if $\alpha$ is close to 1 for many of the instances under consideration then learning will be harder. Crucially, all quantities in the result are spectral and do not depend on the dimensionality of the matrix.

### 2.3 Performing as well as the best fixed $\omega$

Having shown these properties of $U$, we now show that it is learnable via Tsallis-INF (Abernethy et al., 2015; Zimmert \& Seldin, 2021), a bandit algorithm which at each instance $t$ samples $\omega_{t}$ from a discrete probability distribution over a grid of $d$ relaxation parameters, runs SOR with $\omega_{t}$ on the linear system ( $\mathbf{A}_{t}, \mathbf{b}_{t}$ ), and uses the number of iterations required $\operatorname{SOR}_{t}\left(\omega_{t}\right)$ as feedback to update the probability distribution over the grid. The scheme is described in full in Algorithm 2. Note that it is a relative of the simpler and more familiar Exp3 algorithm (Auer et al., 2002), but has a slightly better dependence on the grid size $d$. In Theorem 2.1, we bound the cost of using the parameters $\omega_{t}$ suggested by Tsallis-INF by the total cost of using the best fixed parameter $\omega \in \Omega$ at all iterations-as measured by the surrogate bounds $U_{t}$-plus a term that increases sublinearly in $T$ and a term that decreases in the size of the grid.

```
Algorithm 2: Online tuning of a linear system solver using Tsallis-INF. The probabilities can be
computed using Newton's method (e.g. Zimmert \& Seldin (2021, Algorithm 2)).
Input: solver SOLVE: $\mathbb{R}^{n \times n} \times \mathbb{R}^{n} \times \Omega \mapsto \mathbb{Z}_{>0}$, instance sequence $\left\{\left(\mathbf{A}_{t}, \mathbf{b}_{t}\right)\right\}_{t=1}^{T} \subset \mathbb{R}^{n \times n} \times \mathbb{R}^{n}$,
        normalization $K>0$, parameter grid $\mathbf{g} \in \Omega^{d}$, step-size $\eta>0$
$\mathbf{k} \leftarrow \mathbf{0}_{d} \quad / /$ initialize vector of cumulative costs
for $t=1, \ldots, T$ do
    $\mathbf{p} \leftarrow \arg \min _{\mathbf{p} \in \triangle_{d}}\langle\mathbf{k}, \mathbf{p}\rangle-\frac{4 K}{\eta} \sum_{i=1}^{d} \sqrt{\mathbf{p}_{[i]}} \quad / /$ compute probabilities
    sample $i_{t} \in[d]$ w.p. $\mathbf{p}_{\left[i_{t}\right]}$ and set $\omega_{t}=\mathbf{g}_{\left[i_{t}\right]}$ // sample action from grid
    $\mathbf{k}_{\left[i_{t}\right]} \leftarrow \mathbf{k}_{\left[i_{t}\right]}+\left(\operatorname{SOLVE}\left(\mathbf{A}_{t}, \mathbf{b}_{t}, \omega_{t}\right)-1\right) / \mathbf{p}_{\left[i_{t}\right]} / /$ run solver and update cost
```

Theorem 2.1. Define $\alpha_{t}=\tau_{t}+\left(1-\tau_{t}\right) \max \left\{\beta_{t}^{2}, \omega_{\text {max }}-1\right\}$, where $\beta_{t}=\rho\left(\mathbf{I}_{n}-\mathbf{D}_{t}^{-1} \mathbf{A}_{t}\right)$ and $\tau_{t}$ is the minimal $\tau$ satisfying Assumption 2.1 and the second part of Lemma 2.1. If we run Algorithm 2 using SOR initialized at $\mathbf{x}=\mathbf{0}_{n}$ as the solver, $\mathbf{g}_{[i]}=1+\left(\omega_{\max }-1\right) \frac{i}{d}$ as the parameter grid, normalization $K \geq \frac{-\log \varepsilon}{-\log \alpha_{\max }}$ for $\alpha_{\max }=\max _{t} \alpha_{t}$, and step-size $\eta=1 / \sqrt{T}$ then the expected number of iterations is bounded as

$$
\begin{gather*}
\mathbb{E} \sum_{t=1}^{T} S O R_{t}\left(\omega_{t}\right) \leq 2 K \sqrt{2 d T}+\sum_{t=1}^{T} \frac{-\log \varepsilon}{d \log ^{2} \alpha_{t}}+\min _{\omega \in\left(0, \omega_{\max }\right]} \sum_{t=1}^{T} U_{t}(\omega)  \tag{4}\\
\text { Using } \omega_{\max }=1+\max _{t}\left(\frac{\beta_{t}}{1+\sqrt{1-\beta_{t}^{2}}}\right)^{2}, K=\frac{-\log \varepsilon}{-\log \alpha_{\max }}, \text { and } d=\sqrt[3]{\frac{T}{2} \bar{\gamma}^{2} \log ^{2} \alpha_{\max }}, \text { for } \bar{\gamma}=\frac{1}{T} \sum_{t=1}^{T} \frac{1}{\log ^{2} \alpha_{t}}, \text { yields } \\
\mathbb{E} \sum_{t=1}^{T} S O R_{t}\left(\omega_{t}\right) \leq 3 \log \frac{1}{\varepsilon} \sqrt[3]{\frac{2 \bar{\gamma} T^{2}}{\log ^{2} \alpha_{\max }}}+\min _{\omega \in(0,2)} \sum_{t=1}^{T} U_{t}(\omega) \leq 3 \log \frac{1}{\varepsilon} \sqrt[3]{\frac{2 T^{2}}{\log ^{4} \alpha_{\max }}}+\min _{\omega \in(0,2)} \sum_{t=1}^{T} U_{t}(\omega) \tag{5}
\end{gather*}
$$

Thus asymptotically (as $T \rightarrow \infty$ ) the average cost on each instance is that of the best fixed $\omega \in(0,2)$, as measured by the surrogate loss functions $U_{t}(\omega)$. The result clearly shows that the difficulty of the learning problem can be measured by how close the values of $\alpha_{t}$ are to one. As a quantitative example, for the somewhat "easy" case of $\tau_{t} \leq 0.2$ and $\beta_{t} \leq 0.9$, the first term is $<T \log \frac{1}{\varepsilon}$-i.e. we take at most $\log \frac{1}{\varepsilon}$ excess iterations on average-after around 73 K instances.
The proof of Theorem 2.1 (c.f. Section E) takes advantage of the fact that the upper bounds $U_{t}$ are always decreasing wherever they are not locally Lipschitz; thus for any $\omega \in\left(0, \omega_{\text {max }}\right]$ the next highest grid value in $\mathbf{g}$ will either be better or $\mathcal{O}(1 / d)$ worse. This allows us to obtain the same $\mathcal{O}\left(T^{2 / 3}\right)$ rate as the optimal Lipschitz-bandit regret (Kleinberg, 2004), despite $U_{t}$ being only semi-Lipschitz. One important note is that setting $\omega_{\text {max }}, K$, and $d$ to obtain this rate involves knowing bounds on spectral properties of the instances. The optimal $\omega_{\text {max }}$ requires a bound on $\max _{t} \beta_{t}$ akin to that used by solvers like Chebyshev semi-iteration; assuming this and a reasonable sense of how many iterations are typically required is enough to estimate $\alpha_{\text {max }}$ and then set $d=\sqrt[3]{\frac{T / 2}{\log ^{2} \alpha_{\text {max }}}}$, yielding the right-hand bound in (5). Lastly, we note that Tsallis-INF adds quite little computational overhead: it has a per-instance update cost of $\mathcal{O}(d)$, which for $d=\mathcal{O}(\sqrt[3]{T})$ is likely to be negligible in practice.

### 2.4 THE DIAGONALLY SHIFTED SETTING

The previous analysis is useful when a fixed $\omega$ is good for most instances $\left(\mathbf{A}_{t}, \mathbf{b}_{t}\right)$. A non-fixed comparator can have much stronger performance (c.f. the dashed lines in Figure 1 (center)), so in this section we study how to use additional, known structure in the form of diagonal shifts: at all $t \in[T]$, $\mathbf{A}_{t}=\mathbf{A}+c_{t} \mathbf{I}_{n}$ for some fixed $\mathbf{A}$ and scalar $c_{t}$. It is easy to see that selecting instance-dependent $\omega_{t}$ using the value of the shift is exactly the contextual bandit setting (Beygelzimer et al., 2011), in which the comparator is a fixed policy $f: \mathbb{R} \mapsto \Omega$ that maps the given scalars to parameters for them. Here the regret is defined by $\operatorname{Regret}_{f}\left(\left\{\ell_{t}\right\}_{t=1}^{T}\right)=\sum_{t=1}^{T} \ell_{t}\left(\omega_{t}\right)-\sum_{t=1}^{T} \ell_{t}\left(f\left(c_{t}\right)\right)$. Notably, if $f$ is the optimal mapping from $c_{t}$ to $\omega$ then sublinear regret implies doing nearly optimally at every instance. In our case, the policy $\omega^{*}$ minimizing $U_{t}$ is a well-defined function of $\mathbf{A}_{t}$ (c.f. Lemma 2.1) and thus of $c_{t}$ (Young, 1971); in fact, we can show that the policy is Lipschitz w.r.t. $c_{t}$ (c.f. Lemma E.1). This allows us to use a very simple algorithm-discretizing the space of offsets $c_{t}$ into $m$ intervals and running Tsallis-INF separately on each-to obtain $\mathcal{O}\left(T^{3 / 4}\right)$ regret w.r.t. the instance-optimal policy $\omega^{*}$ :

![](https://cdn.mathpix.com/cropped/2025_11_24_6122a0dad929fc569471g-07.jpg?height=289&width=1364&top_left_y=262&top_left_x=374)
Figure 2: Left: solver cost for $\mathbf{b}$ drawn from a truncated Gaussian v.s. $\mathbf{b}$ a small eigenvector of $\mathbf{C}_{1.4}$. Center-left: cost to solve 5 K diagonally shifted systems $\mathbf{A}_{t}=\mathbf{A}+\frac{12 c_{t}-3}{20} \mathbf{I}_{n}$ for $c_{t} \sim \operatorname{Beta}(2,6)$. Center-right: total SSOR-preconditioned CG iterations taken while solving the 2D heat equation with a time-varying diffusion coefficient (used as context) on different grids, as a function of the linear system dimension. Right: (smoothed) parameters chosen at each timestep of one such simulation, overlaid on a contour plot of the cost of solving the system at step $t$ with parameter $\omega$ (c.f. Appendix G ).

Theorem 2.2 (c.f. Theorem E.1). Suppose all offsets $c_{t}$ lie in $\left[c_{\min }, c_{\min }+C\right]$ for some $c_{\text {min }}>-\lambda_{\text {min }}(\mathbf{A})$, and define $L=\frac{1+\beta_{\text {max }}}{\beta_{\text {max }} \sqrt{1-\beta_{\text {max }}^{2}}}\left(\frac{\lambda_{\text {min }}(\mathbf{D})+c_{\text {min }}+1}{\lambda_{\text {min }}(\mathbf{D})+c_{\text {min }}}\right)^{2}$ for $\beta_{\text {max }}$ as in Theorem 2.1. Then there is a discretization of this interval s.t. running Algorithm 2 separately on each sequence of contexts in each bin with appropriate parameters results in expected cost

$$
\begin{equation*}
\mathbb{E} \sum_{t=1}^{T} \operatorname{SOR}_{t}\left(\omega_{t}\right) \leq \sqrt[4]{\frac{54 C^{3} L^{3} T}{\log ^{2} \alpha_{\max }}}+\frac{4 \log \frac{1}{\varepsilon}}{\log \frac{1}{\alpha_{\max }}} \sqrt[4]{24 C L T^{3}}+\sum_{t=1}^{T} U_{t}\left(\omega^{*}\left(c_{t}\right)\right) \tag{6}
\end{equation*}
$$

Observe that, in addition to $\alpha_{t}$, the difficulty of this learning problem also depends on the maximum spectral radius $\beta_{\text {max }}$ of the Jacobi matrices $\mathbf{I}_{n}-\mathbf{D}^{-1}\left(c_{t}\right) \mathbf{A}\left(c_{t}\right)$ via the Lipschitz constant $L$ of $\omega^{*}$.

### 2.5 Tuning preconditioned CONJugate GRADIENT

CG is perhaps the most-used solver for positive definite systems; while it can be run without tuning, in practice significant acceleration can be realized via a good preconditioner such as (symmetric) SOR. The effect of $\omega$ on CG performance can be somewhat distinct from that of regular SOR, requiring a separate analysis. We use the condition number analysis of Axelsson (1994, Theorem 7.17) to obtain an upper bound $U^{\mathrm{CG}}(\omega)$ on the number of iterations required $\mathrm{CG}(\mathbf{A}, \mathbf{b}, \omega)$ to solve a system. While the resulting bounds match the shape of the true performance less exactly than the SOR bounds (c.f. Figure 4), they still provide a somewhat reasonable surrogate. After showing that these functions are also semi-Lipschitz (c.f. Lemma E.2), we can bound the cost of tuning CG using Tsallis-INF:
Theorem 2.3. Set $\mu_{t}=\rho\left(\mathbf{D}_{t} \mathbf{A}_{t}^{-1}\right), \mu_{\text {max }}=\max _{t} \mu_{t}, \overline{\sqrt{\mu}}=\frac{1}{T} \sum_{t=1}^{T} \sqrt{\mu}{ }_{t}$, and $\kappa_{\text {max }}=\max _{t} \kappa\left(\mathbf{A}_{t}\right)$. If $\min _{t} \mu_{t}-1$ is a positive constant then for Algorithm 2 using preconditioned $C G$ as the solver there exists a parameter grid $\mathbf{g} \in\left[2 \sqrt{2}+2, \omega_{\max }\right]^{d}$ and normalization $K>0$ such that

$$
\begin{equation*}
\mathbb{E} \sum_{t=1}^{T} C G_{t}\left(\omega_{t}\right)=\mathcal{O}\left(\sqrt[3]{\frac{\log ^{2} \frac{\sqrt{\kappa_{\text {max }}}}{\varepsilon}}{\log ^{2} \frac{\sqrt{\mu_{\text {max }}}-1}{\sqrt{\mu_{\text {max }}}+1}}} \sqrt{\mu} T^{2}\right)+\min _{\omega \in(0,2)} \sum_{t=1}^{T} U_{t}(\omega) \tag{7}
\end{equation*}
$$

Observe that the rate in $T$ remains the same as for SOR, but the difficulty of learning now scales mainly with the spectral radii of the matrices $\mathbf{D}_{t} \mathbf{A}_{t}^{-1}$.

## 3 A stochastic analysis of symmetric SOR

Assumption 2.1 in the previous section effectively encodes the idea that convergence will not be too quick for a typical target vector $\mathbf{b}$, e.g. it will not be a low-eigenvalue eigenvector of $\mathbf{C}_{\omega}$ for some otherwise suboptimal $\omega$ (e.g. Figure 2 (left)). Another way of staying in a "typical regime" is randomness, which is what we assume in this section. Specifically, we assume that $\mathbf{b}_{t}=m_{t} \mathbf{u}_{t} \forall t \in[T]$, where $\mathbf{u}_{t} \in \mathbb{R}^{n}$ is uniform on the unit sphere and $m_{t}^{2}$ is a $\chi^{2}$ random variable with $n$ degrees of freedom truncated to $[0, n]$. Since the standard $n$-dimensional Gaussian is exactly the case of untruncated $m_{t}^{2}$, b can be described as coming from a radially truncated normal distribution. Note also that the exact choice of truncation was done for convenience; any finite bound $\geq n$ yields similar results.
We also make two other changes: (1) we study symmetric SOR (SSOR) and (2) we use an absolute convergence criterion, i.e. $\left\|\mathbf{r}_{k}\right\|_{2} \leq \varepsilon$, not $\left\|\mathbf{r}_{k}\right\|_{2} \leq \varepsilon\left\|\mathbf{r}_{0}\right\|_{2}$. Symmetric SOR (c.f. Algorithm 8) is very
similar to the original, except the linear system being solved at every step is now symmetric: $\breve{\mathbf{W}}_{\omega}= \frac{\omega}{2-\omega} \mathbf{W}_{\omega} \mathbf{D}^{-1} \mathbf{W}_{\omega}^{T}$. Note that the defect reduction matrix $\breve{\mathbf{C}}_{\omega}=\mathbf{I}_{n}-\mathbf{A} \breve{\mathbf{W}}_{\omega}^{-1}$ is still not normal, but it is (non-orthogonally) similar to a symmetric matrix, $\mathbf{A}^{-1 / 2} \breve{\mathbf{C}}_{\omega} \mathbf{A}^{1 / 2}$. SSOR is twice as expensive per-iteration, but often converges in fewer steps, and is commonly used as a base method because of its spectral properties (e.g. by the Chebyshev semi-iteration, c.f. Hackbusch (2016, Section 8.4.1)).

### 3.1 Regularity of the expected cost function

We can then show that the expected cost $\mathbb{E}_{\mathbf{b}} \operatorname{SSOR}(\mathbf{A}, \mathbf{b}, \omega)$ is Lipschitz w.r.t. $\omega$ (c.f. Corollary F.1). Our main idea is the observation that, whenever the error $\left\|\breve{\mathbf{C}}_{\omega}^{k} \mathbf{b}\right\|_{2}$ falls below the tolerance $\varepsilon$, randomness should ensure that it does not fall so close to the threshold that the error $\left\|\breve{\mathbf{C}}_{\omega^{\prime}}^{k} \mathbf{b}\right\|_{2}$ of a nearby $\omega^{\prime}$ is not also below $\varepsilon$. Although clearly related to dispersion (Balcan et al., 2018), here we study the behavior of a continuous function around a threshold, rather than the locations of the costs' discontinuities. Our approach has two ingredients, the first being Lipschitzness of the error $\left\|\breve{\mathbf{C}}_{\omega}^{k} \mathbf{b}\right\|_{2}$ at each iteration $k$ w.r.t. $\omega$, which ensures $\left\|\breve{\mathbf{C}}_{\omega^{\prime}}^{k} \mathbf{b}\right\|_{2} \in\left(\varepsilon, \varepsilon+\mathcal{O}\left(\left|\omega-\omega^{\prime}\right|\right)\right]$ if $\left\|\breve{\mathbf{C}}_{\omega}^{k} \mathbf{b}\right\|_{2} \leq \varepsilon<\left\|\breve{\mathbf{C}}_{\omega^{\prime}}^{k} \mathbf{b}\right\|_{2}$. The second ingredient is anti-concentration, specifically that the probability that $\left\|\breve{\mathbf{C}}_{\omega}^{k} \mathbf{b}\right\|_{2}$ lands in $(\varepsilon, \varepsilon+\mathcal{O}(\mid \omega- \left.\left.\omega^{\prime} \mid\right)\right]$ is $\mathcal{O}\left(\left|\omega-\omega^{\prime}\right|\right)$. While intuitive, both steps are made difficult by powering: for high $k$ the random variable $\left\|\breve{\mathbf{C}}_{\omega}^{k} \mathbf{b}\right\|_{2}$ is highly concentrated because $\rho\left(\breve{\mathbf{C}}_{\omega}\right) \ll 1$; in fact its measure over the interval is $\mathcal{O}\left(\left|\omega-\omega^{\prime}\right| / \rho\left(\breve{\mathbf{C}}_{\omega}\right)^{k}\right)$. To cancel this, the Lipschitz constant of $\left\|\breve{\mathbf{C}}_{\omega}^{k} \mathbf{b}\right\|_{2}$ must scale with $\rho\left(\breve{\mathbf{C}}_{\omega}\right)^{k}$, which we can show because switching to SSOR makes $\breve{\mathbf{C}}_{\omega}^{k}$ is similar to a normal matrix. The other algorithmic modification we make-using absolute rather than relative tolerance-is so that $\left\|\breve{\mathbf{C}}_{\omega}^{k} \mathbf{b}\right\|_{2}^{2}$ is (roughly) a sum of i.i.d. $\chi^{2}$ random variables; note that the square of relative tolerance criterion $\left\|\breve{\mathbf{C}}_{\omega}^{k} \mathbf{b}\right\|_{2}^{2} /\|\mathbf{b}\|_{2}^{2}$ does not admit such a result. At the same time, absolute tolerance does not imply an a.s. bound on the number of iterations if $\|\mathbf{b}\|_{2}$ is unbounded, which is why we truncate its distribution.

Lipschitzness follows because $\left|\mathbb{E}_{\mathbf{b}} \operatorname{SSOR}(\omega)-\mathbb{E}_{\mathbf{b}} \operatorname{SSOR}\left(\omega^{\prime}\right)\right|$ can be bounded using Jensen's inequality by the probability that $\omega$ and $\omega^{\prime}$ have different costs $k \neq l$, which is at most the probability that $\left\|\breve{\mathbf{C}}_{\omega}^{k} \mathbf{b}\right\|_{2}$ or $\left\|\breve{\mathbf{C}}_{\omega^{\prime}}^{l} \mathbf{b}\right\|_{2}$ land in an interval of length $\mathcal{O}\left(\left|\omega-\omega^{\prime}\right|\right)$. Note that the Lipschitz bound includes an $\tilde{\mathcal{O}}(\sqrt{n})$ factor, which results from $\breve{\mathbf{C}}_{\omega}^{k}$ having stable rank $\ll n$ due to powering. Regularity of $\mathbb{E}_{\mathbf{b}}$ SSOR leads directly to regret guarantee for the same algorithm as before, Tsallis-INF:
Theorem 3.1. Define $\kappa_{\max }=\max _{t} \kappa\left(\mathbf{A}_{t}\right)$ to be the largest condition number and $\beta_{\min }= \min _{t} \rho\left(\mathbf{I}_{n}-\mathbf{D}_{t}^{-1} \mathbf{A}_{t}\right)$. Then there exists $K=\Omega\left(\log \frac{n}{\varepsilon}\right)$ s.t. running Algorithm 2 with SSOR has regret

$$
\begin{equation*}
\mathbb{E} \sum_{t=1}^{T} \operatorname{SSOR}_{t}\left(\omega_{t}\right)-\min _{\omega \in\left[1, \omega_{\max }\right]} \sum_{t=1}^{T} \operatorname{SSOR}_{t}(\omega) \leq 2 K \sqrt{2 d T}+\frac{32 K^{4} T}{\beta_{\min }^{4} d} \sqrt{\frac{2 n \kappa_{\max }}{\pi}} \tag{8}
\end{equation*}
$$

Setting $d=\Theta\left(K^{2} \sqrt[3]{n T}\right)$ yields a regret bound of $\mathcal{O}\left(\log ^{2} \frac{n}{\varepsilon} \sqrt[3]{T^{2} \sqrt{n}}\right)$. Note that, while this shows convergence to the true optimal parameter, the constants in the regret term are much worse, not just due to the dependence on $n$ but also in the powers of the number of iterations. Thus this result can be viewed as a proof of the asymptotic ( $T \rightarrow \infty$ ) correctness of Tsallis-INF for tuning SSOR.

### 3.2 CHEBYSHEV REGRESSION FOR DIAGONAL SHIFTS

For the shifted setting, we can use the same approach to prove that $\mathbb{E}_{\mathbf{b}} \operatorname{SSOR}\left(\mathbf{A}+c \mathbf{I}_{n}, \mathbf{b}, \omega\right)$ is Lipschitz w.r.t. the diagonal offset $c$ (c.f. Corollary F.2); for $n=O(1)$ this implies regret $\tilde{\mathcal{O}}\left(T^{3 / 4} \sqrt{n}\right)$ for the same discretization-based algorithm as in Section 2.4. While optimal for Lipschitz functions, the method does not readily adapt to nice data, leading to various smoothed comparators (Krishnamurthy et al., 2019; Majzoubi et al., 2020; Zhu \& Mineiro, 2022); however, as we wish to compete with the true optimal policy, we stay in the original setting and instead highlight how this section's semi-stochastic analysis allows us to study a very different class of bandit algorithms.
In particular, since we are now working directly with the cost function rather than an upper bound, we are able to utilize a more practical regression-oracle algorithm, SquareCB (Foster \& Rakhlin, 2020). It assumes a class of regressors $h:\left[c_{\text {min }}, c_{\text {min }}+C\right] \times[d] \mapsto[0,1]$ with at least one function that perfectly predicts the expected performance $\mathbb{E}_{\mathbf{b}} \operatorname{SSOR}\left(\mathbf{A}+c \mathbf{I}_{n}, \mathbf{b}, \mathbf{g}_{[i]}\right)$ of each action $\mathbf{g}_{[i]}$ given the context $c$; a small amount of model misspecification is allowed. If there exists an online algorithm that can obtain low regret w.r.t. this function class, then SquareCB can obtain low regret w.r.t. any policy.

```
Algorithm 3: ChebCB: SquareCB with a follow-the-leader oracle and polynomial regressor class.
Input: solver SOLVE: $\mathbb{R}^{n \times n} \times \mathbb{R}^{n} \times \Omega \mapsto \mathbb{Z}_{>0}$, instance sequence $\left\{\left(\mathbf{A}_{t}, \mathbf{b}_{t}\right)\right\}_{t=1}^{T} \subset \mathbb{R}^{n \times n} \times \mathbb{R}^{n}$,
        context sequence $\left\{c_{t}\right\}_{t=1}^{T} \subset\left[c_{\text {min }}, c_{\text {min }}+C\right]$, learning rate $\eta>0$, parameter grid $\mathbf{g} \in \Omega^{d}$,
        Chebyshev polynomial features $\mathbf{f}:\left[c_{\text {min }}, c_{\text {min }}+C\right] \mapsto \mathbb{R}^{m+1}$, normalizations $K, L, N>0$
for $t=1, \ldots, T$ do
    $\theta_{i} \leftarrow \underset{\left|\theta_{[0]}\right| \leq \frac{1}{N},\left|\theta_{[j]}\right| \leq \frac{2 C L}{K N j}}{\arg \min } \sum_{\substack{s=1 \\ i_{s}=i}}^{t-1}\left(\left\langle\theta, \mathbf{f}\left(c_{s}\right)\right\rangle-\frac{k_{s}}{K N}\right)^{2} \forall i \in[d] \quad$ // update models
    $\mathbf{s}_{[i]} \leftarrow\left\langle\theta_{i}, \mathbf{f}\left(c_{t}\right)\right\rangle \forall i \in[d] \quad / /$ compute model predictions
    $i^{*} \leftarrow \arg \min _{i \in[d]} \mathbf{S}_{[i]}$
    $\mathbf{p}_{[i]} \leftarrow \frac{1}{d+\eta\left(\mathbf{s}_{[i]}-\mathbf{s}_{\left[i^{*}\right]}\right)} \forall i \neq i^{*} \quad / /$ compute probability of each action
    $\mathbf{p}_{\left[i^{*}\right]} \leftarrow 1-\sum_{i \neq i^{*}} \mathbf{p}_{[i]}$
    sample $i_{t} \in[d]$ w.p. $\mathbf{p}_{\left[i_{t}\right]}$ and set $\omega_{t}=\mathbf{g}_{\left[i_{t}\right]}$ // sample action
    $k_{t} \leftarrow \operatorname{SOLVE}\left(\mathbf{A}_{t}, \mathbf{b}_{t}, \omega_{t}\right)-1 \quad / /$ run solver and update cost
```

To apply it we must specify a suitable class of regressors, bound its approximation error, and specify an algorithm attaining low regret over this class. Since $m$ terms of the Chebyshev series suffice to approximate a Lipschitz function with error $\tilde{\mathcal{O}}(1 / m)$, we use Chebyshev polynomials in $c$ with learned coefficients-i.e. models $\langle\theta, \mathbf{f}(c)\rangle=\sum_{j=0}^{m} \theta_{[j]} P_{j}(c)$, where $P_{j}$ is the $j$ th Chebyshev polynomialas our regressors for each action. To keep predictions bounded, we add constraints $\left|\theta_{[j]}\right|=\mathcal{O}(1 / j)$, which we can do without losing approximation power due to the decay of Chebyshev series coefficients. This allows us to show $\mathcal{O}(d m \log T)$ regret for Follow-The-Leader via Hazan et al. (2007, Theorem 5) and then apply Foster \& Rakhlin (2020, Theorem 5) to obtain the following guarantee:
Theorem 3.2 (Corollary of Theorem C.4). Suppose $c_{\min }>-\lambda_{\min }(\mathbf{A})$. Then Algorithm 3 with appropriate parameters has regret w.r.t. any policy $f:\left[c_{\min }, c_{\min }+C\right] \mapsto \Omega$ of

$$
\begin{equation*}
\mathbb{E} \sum_{t=1}^{T} \operatorname{SSOR}_{t}\left(\omega_{t}\right)-\sum_{t=1}^{T} \operatorname{SSOR}_{t}\left(f\left(c_{t}\right)\right) \leq \tilde{\mathcal{O}}\left(d \sqrt{m n T}+\frac{T \sqrt{d n}}{m}+\frac{T \sqrt{n}}{d}\right) \tag{9}
\end{equation*}
$$

Setting $d=\Theta\left(T^{2 / 11}\right)$ and $m=\Theta\left(T^{3 / 11}\right)$ yields $\tilde{\mathcal{O}}\left(T^{9 / 11} \sqrt{n}\right)$ regret, so we asymptotically attain instance-optimal performance, albeit at a rather slow rate. The rate in $n$ is also worse than e.g. our semi-stochastic result for comparing to a fixed $\omega$ (c.f. Theorem 3.1), although to obtain this the latter algorithm uses $d=\mathcal{O}(\sqrt[3]{n})$ grid points, making its overhead nontrivial. We compare ChebCB to the Section 2.4 algorithm based on Tsallis-INF (among other methods), and find that, despite the former's worse guarantees, it seems able to converge to an instance-optimal policy much faster than the latter.

## 4 Conclusion and limitations

We have shown that bandit algorithms provably learn to parameterize SOR, an iterative linear system solver, and do as well asymptotically as the best fixed $\omega$ in terms of either (a) a near-asymptotic measure of cost or (b) expected cost. We further show that a modern contextual bandit method attains near-instance-optimal performance. Both procedures require only the iteration count as feedback and have limited computational overhead settings, making them practical to deploy. Furthermore, the theoretical ideas in this work-especially the use of contextual bandits for taking advantage of instance structure and Section 3.1's conversion of anti-concentrated Lipschitz criteria to Lipschitz expected costs-have the strong potential to be applicable to other domains of data-driven algorithm design.
At the same time, only the near-asymptotic results yield reasonable bound on the instances needed to attain good performance, with the rest having large spectral and dimension-dependent factors; the latter is the most obvious area for improvement. Furthermore, the near-asymptotic upper bounds are somewhat loose for sub-optimal $\omega$ and for preconditioned CG, and as discussed in Section 2.4 do not seem amenable to regression-based CB. Beyond this, a natural direction is to attain semi-stochastic results for non-stationary solvers like preconditioned CG, or either type of result for the many other algorithms in scientific computing. Practically speaking, work on multiple parameters-e.g. the spectral bounds used for Chebyshev semi-iteration, or multiple relaxation parameters for Block-SOR-would likely be most useful. A final direction is to design online learning algorithms that exploit properties of the losses beyond Lipschitzness, or CB algorithms that take better advantage of such functions.

## Acknowledgments

We thank Akshay Krishnamurthy and Ainesh Bakshi for helpful feedback. This work was supported in part by National Science Foundation grants IIS-1705121, IIS-1838017, IIS-1901403, IIS-2046613, IIS-2112471, and OAC-2203821, the Defense Advanced Research Projects Agency under cooperative agreement HR00112020003, a TCS Presidential Fellowship, and funding from Meta, Morgan Stanley, Amazon, Google, and Jane Street. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of any of these funding agencies.

## References

Jacob Abernethy, Chansoo Lee, and Ambuj Tewari. Fighting bandits with a new kind of smoothness. In Advances in Neural Information Processing Systems, 2015.

Brandon Amos. Tutorial on amortized optimization. Foundations and Trends in Machine Learning, 16(5):592-732, 2023.

Hartwig Anzt, Edmond Chow, Jens Saak, and Jack Dongarra. Updating incomplete factorization preconditioners for model order reduction. Numerical Algorithms, 73:611-630, 2016.

Sohei Arisaka and Qianxiao Li. Principled acceleration of iterative numerical methods using machine learning. In Proceedings of the 40th International Conference on Machine Learning, 2023.

Peter Auer, Nicolò Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. The nonstochastic multiarmed bandit problem. SIAM Journal of Computing, 32:48-77, 2002.

Owe Axelsson. Iterative Solution Methods. Cambridge University Press, 1994.
Maria-Florina Balcan. Data-driven algorithm design. In Tim Roughgarden (ed.), Beyond the WorstCase Analysis of Algorithms. Cambridge University Press, Cambridge, UK, 2021.

Maria-Florina Balcan, Travis Dick, and Ellen Vitercik. Dispersion for data-driven algorithm design, online learning, and private optimization. In 59th Annual Symposium on Foundations of Computer Science, 2018.

Maria-Florina Balcan, Travis Dick, and Wesley Pegden. Semi-bandit optimization in the dispersed setting. In Proceedings of the Conference on Uncertainty in Artificial Intelligence, 2020.

Maria-Florina Balcan, Dan DeBlasio, Travis Dick, Carl Kingsford, Tuomas Sandholm, and Ellen Vitercik. How much data is sufficient to learn high-performing algorithms? Generalization guarantees for data-driven algorithm design. In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing, 2021.

Maria-Florina Balcan, Mikhail Khodak, Dravyansh Sharma, and Ameet Talwalkar. Provably tuning the ElasticNet across instances. In Advances in Neural Information Processing Systems, 2022.

Peter Bartlett, Piotr Indyk, and Tal Wagner. Generalization bounds for data-driven numerical linear algebra. In Proceedings of the 35th Annual Conference on Learning Theory, 2022.

Manuel Baumann and Martin B. van Gijzen. Nested Krylov methods for shifted linear systems. SIAM Journal on Scientific Computing, 37:S90-S112, 2015.

Stefania Bellavia, Valentina De Simone, Daniela di Serafina, and Benedetta Morini. Efficient preconditioner updates for shifted linear systems. SIAM Journal on Scientific Computing, 33: 1785-1809, 2011.

Alina Beygelzimer, John Langford, Lihong Li, Lev Reyzin, and Robert E. Schapire. Contextual bandit algorithms with supervised learning guarantees. In Proceedings of the 14th International Conference on Artificial Intelligence and Statistics, 2011.

Sébastien Bubeck and Nicolò Cesa-Bianchi. Regret analysis of stochastic and nonstochastic multiarmed bandit problems. Foundations and Trends in Machine Learning, 5(1):1-122, 2012.

Nicolò Cesa-Bianchi and Gábor Lugosi. Prediction, Learning, and Games. Cambridge University Press, 2006.

Justin Y. Chen, Sandeep Silwal, Ali Vakilian, and Fred Zhang. Faster fundamental graph algorithms via learned predictions. In Proceedings of the 40th International Conference on Machine Learning, 2022.

Xinyi Chen and Elad Hazan. A nonstochastic control approach to optimization. arXiv, 2023.
Elliott Ward Cheney. Introduction to Approximation Theory. Chelsea Publishing Company, 1982.
Giulia Denevi, Carlo Ciliberto, Riccardo Grazzi, and Massimiliano Pontil. Learning-to-learn stochastic gradient descent with biased regularization. In Proceedings of the 36th International Conference on Machine Learning, 2019.

Michael Dinitz, Sungjin Im, Thomas Lavastida, Benjamin Moseley, and Sergei Vassilvitskii. Faster matchings via learned duals. In Advances in Neural Information Processing Systems, 2021.

Paul Dütting, Guru Guruganesh, Jon Schneider, and Joshua R. Wang. Optimal no-regret learning for one-sided lipschitz functions. In Proceedings of the 40th International Conference on Machine Learning, 2023.

Louis W. Ehrlich. An ad hoc SOR method. Journal of Computational Physics, 44:31-45, 1981.
Lakhdar Elbouyahyaoui, Mohammed Heyouni, Azita Tajaddini, and Farid Saberi-Movahed. On restarted and deflated block FOM and GMRES methods for sequences of shifted linear systems. Numerical Algorithms, 87:1257-1299, 2021.

Dylan J. Foster and Alexander Rakhlin. Beyond UCB: Optimal and efficient contextual bandits with regression oracles. In Proceedings of the 37th International Conference on Machine Learning, 2020.

Isaac Fried and Jim Metzler. SOR vs. conjugate gradients in a finite element discretization. International Journal for Numerical Methods in Engineering, 12:1329-1332, 1978.

Andreas Frommer and Uew Glässner. Restarted GMRES for shifted linear systems. SIAM Journal on Scientific Computing, 19:15-26, 1998.

Gene H. Golub and Qiang Ye. Inexact preconditioned conjugate gradient method with inner-outer iteration. SIAM Journal on Scientific Computing, 21:1305-1320, 1999.

Anne Greenbaum. Iterative Methods for Solving Linear Systems. Society for Industrial and Applied Mathematics, 1997.

Rishi Gupta and Timothy Roughgarden. A PAC approach to application-specific algorithm selection. SIAM Journal on Computing, 46(3):992-1017, 2017.

Wolfgang Hackbusch. Iterative Solution of Large Sparse Systems of Equations. Springer International Publishing, 2016.

Elad Hazan, Amit Agarwal, and Satyen Kale. Logarithmic regret algorithms for online convex optimization. Machine Learning, 69:169-192, 2007.

George Em Karniadakis, Ioannis G. Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-informed machine learning. Nature Reviews Physics, 3:422-440, 2021.

Michael Kazhdan, Jake Solomon, and Mirela Ben-Chen. Can mean-curvature flow be modified to be non-singular? In Eurographics Symposium on Geometry Processing 2012, 2012.

Mikhail Khodak, Maria-Florina Balcan, and Ameet Talwalkar. Adaptive gradient-based meta-learning methods. In Advances in Neural Information Processing Systems, 2019.

Mikhail Khodak, Maria-Florina Balcan, Ameet Talwalkar, and Sergei Vassilvitskii. Learning predictions for algorithms with predictions. In Advances in Neural Information Processing Systems, 2022.

John B. King, Samim Anghaie, and Henry M. Domanus. Comparative performance of the conjugate gradient and SOR methods for computational thermal hydraulics. In Proceedings of the Joint Meeting of the American Nuclear Society and the Atomic Industrial Forum, 1987.

Robert Kleinberg. Nearly tight bounds for the continuum-armed bandit problem. In Advances in Neural Information Processing Systems, 2004.

Akshay Krishnamurthy, John Langord, Alexandrs Slivkins, and Chicheng Zhang. Contextual bandits with continuous actions: Smoothing, zooming, and adapting. In Proceedings of the 32nd Conference on Learning Theory, 2019.

John Lafferty, Han Liu, and Larry Wasserman. Statistical machine learning. https://www.stat.cmu.edu/ larry/=sml/Concentration.pdf, 2010.

Sören Laue, Matthias Mitterreiter, and Joachim Giesen. Computing higher order derivatives of matrix and tensor expressions. In Advances in Neural Information Processing Systems, 2018.

Randall J. LeVeque. Finite Difference Methods for Ordinary and Partial Differential Equations. SIAM, 2007.

Yichen Li, Peter Yichen Chen, Tao Du, and Wojciech Matusik. Learning preconditioners for conjugate gradient PDE solvers. In Proceedings of the 40th International Conference on Machine Learning, 2023.

Zongyi Li, Nikola Borislavov Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. In Proceedings of the 9th International Conference on Learning Representations, 2021.

Tyler Lu, Dávid Pál, and Martin Pál. Contextual multi-armed bandits. In Proceedings of the 13th International Conference on Artificial Intelligence and Statistics, 2010.

Ilay Luz, Meirav Galun, Haggai Maron, Ronen Basri, and Irad Yavneh. Learning algebraic multigrid using graph neural networks. In Proceedings of the 37th International Conference on Machine Learning, 2020.

Maryam Majzoubi, Chicheng Zhang, Rajan Chari, Akshay Krishnamurthy, John Langford, and Alexandrs Slivkins. Efficient contextual bandits with continuous actions. In Advances in Neural Information Processing Systems, 2020.
Donald Marquardt. An algorithm for least-squares estimation of nonlinear parameters. SIAM Journal on Applied Mathematics, 11(2):431-441, 1963.

Tanya Marwah, Zachary C. Lipton, and Andrej Risteski. Parametric complexity bounds for approximating PDEs with neural networks. In Advances in Neural Information Processing Systems, 2021.

Michael Mitzenmacher and Sergei Vassilvitskii. Algorithms with predictions. In Tim Roughgarden (ed.), Beyond the Worst-Case Analysis of Algorithms. Cambridge University Press, Cambridge, UK, 2021.

Cameron Musco and Christopher Musco. Randomized block Krylov methods for stronger and faster approximate singular value decomposition. In Advances in Neural Information Processing Systems, 2015.

Michael L. Parks, Eric de Sturler, Greg Mackey, Duane D. Johnson, and Spandan Maiti. Recycling Krylov subspaces for sequences of linear systems. SIAM Journal on Scientific Computing, 28(5): 1651-1674, 2006.

Shinsaku Sakaue and Taihei Oki. Discrete-convex-analysis-based framework for warm-starting algorithms with predictions. In Advances in Neural Information Processing Systems, 2022.

Rajiv Sambharya, Georgina Hall, Brandon Amos, and Bartolomeo Stellato. End-to-end learning to warm-start for real-time quadratic optimization. In Proceedings of the 5th Annual Conference on Learning for Dynamics and Control, 2023.

Nikunj Saunshi, Yi Zhang, Mikhail Khodak, and Sanjeev Arora. A sample complexity separation between non-convex and convex meta-learning. In Proceedings of the 37th International Conference on Machine Learning, 2020.

David Simchi-Levi and Yunzong Xu. Bypassing the monster: A faster and simpler optimal algorithm for contextual bandits under realizability. Mathematics of Operations Research, 47, 2021.

Ali Taghibakhshi, Scott MacLachlan, Luke Olson, and Matthew West. Optimization-based algebraic multigrid coarsening using reinforcement learning. In Advances in Neural Information Processing Systems, 2021.

Jurjen D. Tebbens and Miroslav Tůma. Efficient preconditioning of sequences of nonsymmetric linear systems. SIAM Journal on Scientific Computing, 29:1918-1941, 2007.

James William Thomas. Numerical Partial Differential Equations. Springer Science+Business Media, 1999.

Lloyd N. Trefethen. Is Gauss quadrature better than Clenshaw-Curtis? SIAM Review, 50(1):67-87, 2008.

Lloyd N. Trefethen and Mark Embree. Spectra and Pseudospectra: The Behavior of Nonnormal Matrices and Operators. Princeton University Press, 2005.
L. Dale Van Vleck and D. J. Dwyer. Successive overrelaxation, block iteration, and method of conjugate gradients for solving equations for multiple trait evaluation of sires. Jorunal of Dairy Science, 68:760-767, 1985.

Rui-Rui Wang, Qiang Niu, Xiao-Bin Tang, and Xiang Wang. Solving shifted linear systems with restarted GMRES augmented with error approximations. Computers \& Mathematics with Applications, 78:1910-1918, 2019.

Zbigniew I. Woźnicki. On numerical analysis of conjugate gradient method. Japan Journal of Industrial and Applied Mathematics, 10:487-519, 1993.

Zbigniew I. Woźnicki. On performance of SOR method for solving nonsymmetric linear systems. Journal of Computational and Applied Mathematics, 137:145-176, 2001.

David M. Young. Iterative Solution of Large Linear Systems. Academic Press, 1971.
Yinglun Zhu and Paul Mineiro. Contextual bandits with smooth regret: Efficient learning in continuous action spaces. In Proceedings of the 39th International Conference on Machine Learning, 2022.

Julian Zimmert and Yevgeny Seldin. Tsallis-INF: An optimal algorithm for stochastic and adversarial bandits. Journal of Machine Learning Research, 22:1-49, 2021.

## A Related work and comparisons

Our analysis falls mainly into the framework of data-driven algorithm design, which has a long history (Gupta \& Roughgarden, 2017; Balcan, 2021). Closely related is the study by Gupta \& Roughgarden (2017) of the sample complexity of learning the step-size of gradient descent, which can also be used to solve linear systems. While their sample complexity guarantee is logarithmic in the precision $1 / \varepsilon$, directly applying their Lipschitz-like analysis in a bandit setting yields regret with a polynomial dependence; note that a typical setting of $\varepsilon$ is $10^{-8}$. Mathematically, their analysis relies crucially on the iteration reducing error at every step, which is well-known not to be the case for SOR (e.g. Trefethen \& Embree (2005, Figure 25.6)). Data-driven numerical linear algebra was studied most explicitly by Bartlett et al. (2022), who provided sample complexity framework applicable to many algorithms; their focus is on the offline setting where an algorithm is learned from a batch of samples. While they do not consider linear systems directly, in Appendix A. 1 we do compare to the guarantee their framework implies for SOR; we obtain similar sample complexity with an efficient learning procedure, at the cost of a strong distributional assumption on the target vector. Note that generalization guarantees have been shown for convex quadratic programming-which subsumes linear systemsby Sambharya et al. (2023); they focus on learning-to-initialize, which we do not consider because for high precisions the initialization quality usually does not have a strong impact on cost. Note that all of the above work also does not provide end-to-end guarantees, only e.g. sample complexity bounds.

Online learning guarantees were shown for the related problem of tuning regularized regression by Balcan et al. (2022), albeit in the easier full information setting and with the target of reducing error rather than computation. Their approach relies on the dispersion technique (Balcan et al., 2018), which often involves showing that discontinuities in the cost are defined by bounded-degree polynomials (Balcan et al., 2020). While possibly applicable in our setting, we suspect using it would lead to unacceptably high dependence on the dimension and precision, as the power of the polynomials defining our decision boundaries is $\mathcal{O}\left(n^{-\log \varepsilon}\right)$. Lastly, we believe our work is notable within this field as a first example of using contextual bandits, and in doing so competing with the provably instance-optimal policy.
Iterative (discrete) optimization has been studied in the related area of learning-augmented algorithms (a.k.a. algorithms with predictions) (Dinitz et al., 2021; Chen et al., 2022; Sakaue \& Oki, 2022), which shows data-dependent performance guarantees as a function of (learned) predictions (Mitzenmacher \& Vassilvitskii, 2021); these can then be used as surrogate losses for learning (Khodak et al., 2022). Our construction of an upper bound under asymptotic convergence is inspired by this, although unlike previous work we do not assume access to the bound directly because it depends on hard-to-compute spectral properties. Algorithms with predictions often involve initializing a computation with a prediction of its outcome, e.g. a vector near the solution $\mathbf{A}^{-1} \mathbf{b}$; we do not consider this because the runtime of SOR and other solvers depends fairly weakly on the distance to the initialization.

A last theoretical area is that of gradient-based meta-learning, which studies how to initialize and tune other parameters of gradient descent and related methods (Khodak et al., 2019; Denevi et al., 2019; Saunshi et al., 2020; Chen \& Hazan, 2023). This field focuses on learning-theoretic notions of cost such as regret or statistical risk. Furthermore, their guarantees are usually on the error after a fixed number of gradient steps rather than the number of iterations required to converge; targeting the former can be highly suboptimal in scientific computing applications (Arisaka \& Li, 2023). This latter work, which connects meta-learning and data-driven scientific computing, analyzes specific case studies for accelerating numerical solvers, whereas we focus on a general learning guarantee.

Empirically, there are many learned solvers (Luz et al., 2020; Taghibakhshi et al., 2021; Li et al., 2023) and even full simulation replacements (Karniadakis et al., 2021; Li et al., 2021); to our knowledge, theoretical studies of the latter have focused on expressivity (Marwah et al., 2021). Amortizing the cost on future simulations (Amos, 2023), these approaches use offline computation to train models that integrate directly with solvers or avoid solving linear systems altogether. In contrast, the methods we propose are online and lightweight, both computationally and in terms of implementation; unlike many deep learning approaches, the additional computation scales slowly with dimension and needs only black-box access to existing solvers. As a result, our methods can be viewed as reasonable baselines, and we discuss an indirect comparison with the CG-preconditioner-learning approach of Li et al. (2023) in Appendix G. Finally, note that improving the performance of linear solvers across a sequence of related instances has seen a lot of study in the scientific computing literature (Parks et al., 2006; Tebbens \& Tůma, 2007; Elbouyahyaoui et al., 2021). To our knowledge, this work does not give explicit guarantees on the number of iterations, and so a direct theoretical comparison is challenging.

## A. 1 Sample complexity and comparison with the Goldberg-Jerrum framework

While not the focus of our work, we briefly note the generalization implications of our semi-stochastic analysis. Suppose for any $\alpha>0$ we have $T=\tilde{\mathcal{O}}\left(\frac{1}{\alpha^{2}} \operatorname{poly} \log \frac{n}{\delta}\right)$ i.i.d. samples from a distribution $\mathcal{D}$ over matrices $\mathbf{A}_{t}$ satisfying the assumptions in Section 2.1 and truncated Gaussian targets $\mathbf{b}_{t}$. Then empirical risk minimization $\hat{\omega}=\arg \min _{\hat{\omega} \in \mathbf{g}} \sum_{t=1}^{T} \operatorname{SSOR}\left(\mathbf{A}_{t}, \mathbf{b}_{t}, \omega\right)$ over a uniform grid $\mathbf{g} \in\left[1, \omega_{\max }\right]^{d}$ of size $d=\tilde{\mathcal{O}}(\sqrt{n T})$ will be $\alpha$-suboptimal w.p. $\geq 1-\delta$ :
Corollary A.1. Let $\mathcal{D}$ be a distribution over matrix-vector pairs $(\mathbf{A}, \mathbf{b}) \in \mathbb{R}^{n \times n} \times \mathbb{R}^{n}$ where A satisfies the $S O R$ conditions and for every $\mathbf{A}$ the conditional distribution of $\mathcal{D}$ given $\mathbf{A}$ over $\mathbb{R}^{n}$ is the truncated Gaussian. For every $T \geq 1$ consider the algorithm that draws $T$ samples $\left(\mathbf{A}_{t}, \mathbf{b}_{t}\right) \sim \mathcal{D}$ and outputs $\hat{\omega}=\arg \min _{\hat{\omega} \in \mathbf{g}} \sum_{t=1}^{T} \operatorname{SSOR}_{t}(\omega)$, where $\mathbf{g}_{[i]}=1+\left(\omega_{\max }-1\right) \frac{i-1 / 2}{d}$ and $d=\frac{L \sqrt{T}}{K}$ for $L$ as in Corollary F.1. Then $T=\tilde{\mathcal{O}}\left(\frac{1}{\alpha^{2}} \operatorname{poly} \log \frac{n}{\varepsilon \delta}\right)$ samples suffice to ensure $\mathbb{E}_{\mathcal{D}} \operatorname{SSOR}(\mathbf{A}, \mathbf{b}, \hat{\omega}) \leq \min _{\omega \in\left[1, \omega_{\text {max }}\right]} \operatorname{SSOR}(\mathbf{A}, \mathbf{b}, \omega)+\alpha$ holds w.p. $\geq 1-\delta$.

Proof. A standard covering bound (see e.g. Lafferty et al. (2010, Theorem 7.82)) followed by an application of Corollary F. 1 implies that w.p. $\geq 1-\delta$

$$
\begin{align*}
\mathbb{E}_{\mathcal{D}} \operatorname{SSOR}(\mathbf{A}, \mathbf{b}, \hat{\omega}) & \leq \min _{\omega \in \mathbf{g}} \mathbb{E}_{\mathcal{D}} \operatorname{SSOR}(\mathbf{A}, \mathbf{b}, \omega)+3 K \sqrt{\frac{2}{T} \log \frac{2 d}{\delta}} \\
& =\min _{\omega \in \mathbf{g}} \mathbb{E}_{\mathbf{A}}\left[\mathbb{E}_{\mathbf{b}} \operatorname{SSOR}(\mathbf{A}, \mathbf{b}, \omega) \mid \mathbf{A}\right]+3 K \sqrt{\frac{2}{T} \log \frac{2 d}{\delta}} \\
& \leq \min _{\omega \in\left[1, \omega_{\max }\right]} \mathbb{E}_{\mathbf{A}}\left[\left.\mathbb{E}_{\mathbf{b}} \operatorname{SSOR}(\mathbf{A}, \mathbf{b}, \omega)+\frac{L}{d} \right\rvert\, \mathbf{A}\right]+3 K \sqrt{\frac{2}{T} \log \frac{2 d}{\delta}}  \tag{10}\\
& =\min _{\omega \in\left[1, \omega_{\max }\right]} \mathbb{E}_{\mathcal{D}} \operatorname{SSOR}(\mathbf{A}, \mathbf{b}, \omega)+\frac{L}{d}+3 K \sqrt{\frac{2}{T} \log \frac{2 d}{\delta}} \\
& \leq \min _{\omega \in\left[1, \omega_{\max }\right]} \mathbb{E}_{\mathcal{D}} \operatorname{SSOR}(\mathbf{A}, \mathbf{b}, \omega)+4 K \sqrt{\frac{2}{T} \log \frac{2 L T}{K \delta}}
\end{align*}
$$

Noting that by Corollary F. 1 we have $L=\mathcal{O}\left(K^{4} \sqrt{n}\right)=\mathcal{O}\left(\sqrt{n} \log ^{4} \frac{n}{\varepsilon}\right)$ yields the result.
This matches directly applying the GJ framework of Bartlett et al. (2022, Theorem 3.3) to our problem:
Corollary A.2. In the same setting as Corollary A. 1 but generalizing the distribution to any one whose target vector support is $\sqrt{n}$-bounded, empirical risk minimization (running $\hat{\omega}= \arg \min _{\omega \in\left[1, \omega_{\text {max }}\right]} \sum_{t=1}^{T} \operatorname{SSOR}_{t}(\omega)$ ) has sample complexity $\tilde{\mathcal{O}}\left(\frac{1}{\alpha^{2}} \operatorname{poly} \log \frac{n}{\varepsilon \delta}\right)$.

Proof. For every ( $\mathbf{A}, \mathbf{b}$ ) pair in the support of $\mathcal{D}$ and any $r \in \mathbb{R}$ it is straightforward to define a GJ algorithm (Bartlett et al., 2022, Definition 3.1) that checks if $\operatorname{SSOR}(\mathbf{A}, \mathbf{b}, \omega)>r$ by computing $\left\|\mathbf{r}_{k}(\omega)\right\|_{2}^{2}=\left\|\breve{\mathbf{C}}_{\omega}^{k} \mathbf{b}\right\|_{2}^{2}$-a degree $2 k$ polynomial-for every $k \leq\lfloor r\rfloor$ and returning "True" if one of them satisfies $\left\|\mathbf{r}_{k}(\omega)\right\|_{2}^{2} \leq \varepsilon^{2}$ and "False" otherwise (and automatically return "True" for $r \geq K$ and "False" for $r<1$ ). Since the degree of this algorithm is at most $2 K$, the predicate complexity is at most $K$, and the parameter size is 1 , by Bartlett et al. (2022, Theorem 3.3) the pseudodimension of $\left\{\operatorname{SSOR}(\cdot, \cdot, \omega): \omega \in\left[1, \omega_{\text {max }}\right]\right\}$ is $\mathcal{O}(\log K)$. Using the bounded assumption on the target vector$\mathrm{SSOR} \leq K=\mathcal{O}\left(\log \frac{n}{\varepsilon}\right)$-completes the proof.

At the same, recent generalization guarantees for tuning regularization parameters of linear regression by Balcan et al. (2022, Theorem 3.2)-who applied dual function analysis (Balcan et al., 2021)-have a quadratic dependence on the instance dimension. Unlike both results-which use uniform convergence-our bound also uses a (theoretically) efficient learning procedure, at the cost of a strong (but in our view reasonable) distributional assumption on the target vectors.

## A. 2 Approximating the spectral radius of the Jacobi iteration matrix

Because the asymptotically optimal $\omega$ is a function of the spectral radius $\beta=\rho\left(\mathbf{M}_{1}\right)$ of the Jacobi iteration matrix, a reasonable baseline is to simply approximate $\beta$ using an eigenvalue solver and then run SOR with the corresponding approximately best $\omega$. It is difficult to compare our results to this approach directly, since the baseline will always run extra matrix iterations while bandit algorithms will asymptotically run no more than the comparator. Furthermore, $\mathbf{M}_{1}$ is not a normal matrix, a class for which it turns out to be surprisingly difficult to find bounds on the number of iterations required to approximate its largest eigenvalue within some tolerance $\alpha>0$.
A comparison can be made in the diagonal offset setting by modifying this baseline somewhat and making the assumption that $\mathbf{A}$ has a constant diagonal, so that $\mathbf{M}_{1}$ is symmetric and we can use randomized block-Krylov to obtain a $\hat{\beta}$ satisfying $\left|\hat{\beta}^{2}-\beta^{2}\right|=\mathcal{O}(\varepsilon)$ in $\tilde{\mathcal{O}}(1 / \sqrt{\varepsilon})$ iterations w.h.p. (Musco \& Musco, 2015, Theorem 1). To modify the baseline, we consider a preprocessing algorithm which discretizes $\left[c_{\text {min }}, c_{\text {min }}+C\right]$ into $d$ grid points, runs $k$ iterations of randomized blockKrylov on the Jacobi iteration matrix of each matrix $\mathbf{A}+c \mathbf{I}_{n}$ corresponding to offsets $c$ in this grid, and then for each new offset $c_{t}$ we set $\omega_{t}$ using the optimal parameter implied by the approximate spectral radius of the Jacobi iteration matrix of $\mathbf{A}+c \mathbf{I}_{n}$ corresponding to the closest $c$ in the grid. This algorithm thus does $\tilde{\mathcal{O}}(d k)$ matrix-vector products of preprocessing, and since the upper bounds $U_{t}$ are $\frac{1}{2}$-Hölder w.r.t. $\omega$ while the optimal policy is Lipschitz w.r.t. $\beta^{2}$ over an appropriate domain $\left[1, \omega_{\text {max }}\right]$ it will w.h.p. use at most $\tilde{\mathcal{O}}\left(\sqrt{1 / k^{2}+1 / d}\right)$ more iterations at each step $t \in[T]$ compared to the optimal policy. Thus w.h.p. the total regret compared to the optimal policy $\omega^{*}$ is

$$
\begin{equation*}
\sum_{t=1}^{T} \operatorname{SOR}_{t}\left(\omega_{t}\right)=\tilde{\mathcal{O}}(d k+T / d+T / \sqrt{k})+\sum_{t=1}^{T} U_{t}\left(\omega^{*}\left(c_{t}\right)\right) \tag{11}
\end{equation*}
$$

Setting $d=\sqrt[4]{T}$ and $k=\sqrt{T}$ yields the rate $\tilde{\mathcal{O}}\left(T^{3 / 4}\right)$, which can be compared directly to our $\tilde{\mathcal{O}}\left(T^{3 / 4}\right)$ rate for the discretized Tsallis-INF algorithm in Theorem 2.2. The rate of approximating $\rho\left(\mathrm{M}_{1}\right)$ thus matches that of our simplest approach, although unlike the latter (and also unlike ChebCB) it does not guarantee performance as good as the optimal policy in the semi-stochastic setting, where $\omega^{*}$ might not be optimal. Intuitively, the randomized block-Krylov baseline will also suffer from spending computation on points $c \in\left[c_{\text {min }}, c_{\text {min }}+C\right]$ that it does not end up seeing.

```
Algorithm 4: General form of Tsallis-INF. The probabilities can be computed using Newton's
method (e.g. Zimmert \& Seldin (2021, Algorithm 2)).
Input: loss sequence $\left\{\ell_{t}:[a, b] \mapsto[0, K]\right\}_{t=1}^{T}$, action set $\mathbf{g} \in[a, b]^{d}$, step-sizes $\eta_{1}, \ldots, \eta_{T}>0$
$\mathbf{k} \leftarrow \mathbf{0}_{d} \quad / /$ initialize vector of cumulative losses
for $t=1, \ldots, T$ do
    $\mathbf{p} \leftarrow \arg \min _{\mathbf{p} \in \triangle_{d}}\langle\mathbf{k}, \mathbf{p}\rangle-\frac{4 K}{\eta_{t}} \sum_{i=1}^{d} \sqrt{\mathbf{p}_{[i]}} \quad / /$ compute probabilities
    sample $i_{t} \in[d]$ with probability $\mathbf{p}_{\left[i_{t}\right]}$ // sample index of an action
    $\mathbf{k}_{\left[i_{t}\right]} \leftarrow \mathbf{k}_{\left[i_{t}\right]}+\ell_{t}\left(\mathbf{g}_{\left[i_{t}\right]}\right) / \mathbf{p}_{\left[i_{t}\right]} \quad / /$ play action and update losses
```


## B Semi-Lipschitz bandits

We consider a sequence of adaptively chosen loss functions $\ell_{1}, \ldots, \ell_{T}:[a, b] \mapsto[0, K]$ on an interval $[a, b] \subset \mathbb{R}$ and upper bounds $u_{1}, \ldots, u_{T}:[a, b] \mapsto \mathbb{R}$ satisfying $u_{t}(x) \geq \ell_{t}(x) \forall t \in[T], x \in[a, b]$, where $[T]$ denotes the set of integers from 1 to $T$. Our analysis will focus on the Tsallis-INF algorithm of Abernethy et al. (2015), which we write in its general form in Algorithm 4, although the analysis extends easily to the better-known (but sub-optimal) Exp3 (Auer et al., 2002). For Tsallis-INF, the following two facts follow directly from known results:
Theorem B. 1 (Corollary of Abernethy et al. (2015, Corollary 3.2)). If $\eta_{t}=1 / \sqrt{T} \forall t \in[T]$ then Algorithm 4 has regret $\mathbb{E} \sum_{t=1}^{T} \ell_{t}\left(\mathbf{g}_{\left[i_{t}\right]}\right)-\min _{i \in[d]} \sum_{t=1}^{T} \ell_{t}\left(\mathbf{g}_{[i]}\right) \leq 2 K \sqrt{2 d T}$.
Theorem B. 2 (Corollary of Zimmert \& Seldin (2021, Theorem 1)). If $\eta_{t}=2 / \sqrt{t} \forall t \in[T]$ then Algorithm 4 has regret $\mathbb{E} \sum_{t=1}^{T} \ell_{t}\left(\mathbf{g}_{\left[i_{t}\right]}\right)-\min _{i \in[d]} \sum_{t=1}^{T} \ell_{t}\left(\mathbf{g}_{[i]}\right) \leq 4 K \sqrt{d T}+1$.

We now define a generalization of the Lipschitzness condition that trivially generalizes regular $L$-Lipschitz functions, as well as the notion of one-sided Lipschitz functions studied in the stochastic setting by Dütting et al. (2023).
Definition B.1. Given a constant $L \geq 0$ and a point $z \in[a, b]$, we say a function $f:[a, b] \mapsto \mathbb{R}$ is $(L, z)$-semi-Lipschitz if $f(x)-f(y) \leq L|x-y| \forall x, y$ s.t. $|x-z| \leq|y-z|$.

We now show that Tsallis-INF with bandit access to $\ell_{t}$ on a discretization of $[a, b]$ attains $\mathcal{O}\left(T^{2 / 3}\right)$ regret w.r.t. any fixed $x \in[a, b]$ evaluated by any comparator sequence of semi-Lipschitz upper bounds $u_{t}$. Note that guarantees for the standard comparator can be recovered by just setting $\ell_{t}=u_{t} \forall t \in[T]$, and that the rate is optimal by Kleinberg (2004, Theorem 4.2).
Theorem B.3. If $u_{t} \geq \ell_{t}$ is ( $L_{t}, z$ )-semi-Lipschitz $\forall t \in[T]$ then Algorithm 4 using action space $\mathbf{g} \in[a, b]^{d}$ s.t. $\mathbf{g}_{[i]}=a+\frac{b-a}{d} i \forall i \in[d-1]$ and $\mathbf{g}_{[d]}=z$ has regret

$$
\begin{equation*}
\mathbb{E} \sum_{t=1}^{T} \ell_{t}\left(\mathbf{g}_{\left[i_{t}\right]}\right)-\min _{x \in[a, b]} \sum_{t=1}^{T} u_{t}(x) \leq 2 K \sqrt{2 d T}+\frac{b-a}{d} \sum_{t=1}^{T} L_{t} \tag{12}
\end{equation*}
$$

Setting $d=\sqrt[3]{\frac{(b-a)^{2} \bar{L}^{2} T}{2 K^{2}}}$ for $\bar{L}=\frac{1}{T} \sum_{t=1}^{T} L_{t}$ yields the bound $3 \sqrt[3]{2(b-a) \bar{L} K^{2} T^{2}}$.
Proof. Let $\lceil\cdot\rfloor_{\mathbf{g}}$ denote rounding to the closest element of $\mathbf{g}$ in the direction of $z$. Then for $x \in[a, b]$ we have $\left|[x]_{\mathbf{g}}-z\right| \leq|x-z|$ and $\left|\lceil x\rfloor_{\mathbf{g}}-x\right| \leq \frac{b-a}{d}$, so applying Theorem B. 1 and this fact yields

$$
\begin{align*}
\mathbb{E} \sum_{t=1}^{T} \ell_{t}\left(\mathbf{g}_{\left[i_{t}\right]}\right) \leq 2 K \sqrt{2 d T}+\min _{i \in[d]} \sum_{t=1}^{T} \ell_{t}\left(\mathbf{g}_{[i]}\right) & \leq 2 K \sqrt{2 d T}+\min _{i \in[d]} \sum_{t=1}^{T} u_{t}\left(\mathbf{g}_{[i]}\right) \\
& =2 K \sqrt{2 d T}+\min _{x \in[a, b]} \sum_{t=1}^{T} u_{t}\left(\lceil x\rfloor_{\mathbf{g}}\right) \\
& \leq 2 K \sqrt{2 d T}+\frac{b-a}{d} \sum_{t=1}^{T} L_{t}+\min _{x \in[a, b]} \sum_{t=1}^{T} u_{t}(x) \tag{13}
\end{align*}
$$

```
Algorithm 5: Contextual bandit algorithm using instances of Tsallis-INF over a grid of contexts.
Input: loss sequence $\left\{\ell_{t}:[a, b] \mapsto[0, K]\right\}_{t=1}^{T}$, context sequence $\left\{c_{t}\right\}_{t=1}^{T} \subset[c, c+C]$,
action set $\mathbf{g} \in[a, b]^{d}$, discretization $\mathbf{h} \in[c, c+C]^{m}$
for $j=1, \ldots, m$ do
    $\mathcal{A}_{j}=$ Tsallis-INF $\left(\mathbf{g},\left\{\frac{2}{\sqrt{t}}\right\}_{t=1}^{T}\right)$ // start $m$ instances of Algorithm 4
for $t=1, \ldots, T$ do
    $j_{t}=\min \arg \min _{j \in[m]}\left|\mathbf{h}_{[j]}-c_{t}\right| \quad / /$ pick element of $\mathbf{h}$ closest to $c_{t}$
    $i_{t} \leftarrow \mathcal{A}_{j_{t}} \quad / /$ get action from $j_{t}$ th instance of Algorithm 4
    $\ell_{t}\left(\mathbf{g}_{\left[i_{t}\right]}\right) \rightarrow \mathcal{A}_{j_{t}} \quad / /$ pass loss to $j_{t}$ th instance of Algorithm 4
```

For contextual bandits, we restrict to $\left(L_{t}, b\right)$-semi-Lipschitz functions and $L_{f}$-Lipschitz policies, obtaining $\mathcal{O}\left(T^{3 / 4}\right)$ regret; this rate matches known upper and lower bounds for the case where losses are Lipschitz in both actions and contexts (Lu et al., 2010, Theorem 1), although this does not imply optimality of our result.
Theorem B.4. If $u_{t} \geq \ell_{t}$ is ( $L_{t}, b$ )-semi-Lipschitz and $c_{t} \in[c, c+C] \forall t \in[T]$ then Algorithm 5 using action space $\mathbf{g}_{[i]}=a+\frac{b-a}{d} i$ and $\mathbf{h}_{[j]}=c+\frac{C}{m}\left(j-\frac{1}{2}\right)$ as the grid of contexts has regret w.r.t. any $L_{f}$-Lipschitz policy $f:[c, c+C] \mapsto[a, b]$ of

$$
\begin{equation*}
\mathbb{E} \sum_{t=1}^{T} \ell_{t}\left(\mathbf{g}_{\left[i_{t}\right]}\right)-\sum_{t=1}^{T} u_{t}\left(\pi\left(c_{t}\right)\right) \leq m+4 K \sqrt{d m T}+\left(\frac{C L_{f}}{m}+\frac{b-a}{d}\right) \sum_{t=1}^{T} L_{t} \tag{14}
\end{equation*}
$$

Setting $d=\sqrt[4]{\frac{(b-a)^{3} \bar{L}^{2} T}{4 C L_{f} K^{2}}}, m=\sqrt[4]{\frac{C^{3} L_{f}^{3} \bar{L}^{2} T}{4(b-a) K^{2}}}$ yields regret $4 \sqrt[4]{4 K^{2} \bar{L}^{2}(b-a) C L_{f} T^{3}}+\sqrt[4]{\frac{C^{3} L_{f}^{3} \bar{L}^{2} T}{4(b-a) K^{2}}}$.
Proof. Define $\lceil\cdot\rfloor_{\mathbf{h}}$ to be the operation of rounding to the closest element of $\mathbf{h}$, breaking ties arbitrarily, and set $[T]_{j}=\left\{t \in[T]:\left\lceil c_{t}\right\rfloor_{\mathbf{h}}=\mathbf{h}_{[j]}\right\}$. Furthermore, define $\lceil x\rceil_{\mathbf{g}}$ to be the smallest element $\mathbf{g}_{[i]}$ in g s.t. $x+\frac{C L_{f}}{2 m} \leq \mathbf{g}_{[i]}$ (or $\max _{i \in[d]} \mathbf{g}_{[i]}$ if such an element does not exist).

$$
\begin{align*}
\mathbb{E} \sum_{t=1}^{T} \ell_{t}\left(\mathbf{g}_{\left[i_{t}\right]}\right) & =\mathbb{E} \sum_{j=1}^{m} \sum_{t \in[T]_{j}} \ell_{t}\left(\mathbf{g}_{\left[i_{t}\right]}\right)-\min _{i \in[d]} \sum_{t \in[T]_{j}} \ell_{t}\left(\mathbf{g}_{[i]}\right)+\min _{i \in[d]} \sum_{t \in[T]_{j}} \ell_{t}\left(\mathbf{g}_{[i]}\right) \\
& \leq m+4 \sum_{j=1}^{m} K \sqrt{d \mid[T]_{j}}+\min _{i \in[d]} \sum_{t \in[T]_{j}} \ell_{t}\left(\mathbf{g}_{[i]}\right)  \tag{15}\\
& \leq m+4 K \sqrt{d m T}+\sum_{j=1}^{m} \min _{i \in[d]} \sum_{t \in[T]_{j}} u_{t}\left(\mathbf{g}_{[i]}\right) \\
& \leq m+4 K \sqrt{d m T}+\sum_{t=1}^{T} u_{t}\left(\left[f\left(\left[c_{t}\right]_{\mathbf{h}}\right)\right]_{\mathbf{g}}\right)
\end{align*}
$$

where the first inequality follows by Theorem B.2, the second applies Jensen's inequality to the left term and $u_{t} \geq \ell_{t}$ on the right, and the last uses optimality of each $i$ for each $j$. Now since $f$ is $L_{f}$-Lipschitz we have by definition of $\left\lceil\cdot_{\mathbf{h}}\right.$ that $\left|f\left(c_{t}\right)-f\left(\left\lceil c_{t}\right\rfloor_{\mathbf{h}}\right)\right| \leq \frac{C L_{f}}{2 m}$. This in turn implies that $f\left(c_{t}\right) \leq\left\lceil f\left(\left\lceil c_{t}\right\rfloor_{\mathbf{h}}\right)\right\rceil_{\mathbf{g}} \leq f\left(c_{t}\right)+\frac{C L_{f}}{m}+\frac{b-a}{d}$ by definition of $\mathbf{g}$ and $\lceil\cdot\rceil_{\mathbf{g}}$. Since $u_{t}$ is ( $L_{t}, b$ )-semi-Lipschitz, the result follows.

## C Chebyshev regression for contextual bandits

## C. 1 Preliminaries

We first state a Lipschitz approximation result that is standard but difficult-to-find formally. For all $j \in \mathbb{Z}_{\geq 0}$ we will use $P_{j}(x)=\cos (j \arccos (x))$ to denote the $j$ th Chebyshev polynomial of the first kind.
Theorem C.1. Let $f:[ \pm 1] \mapsto[ \pm K]$ be a $K$-bounded, Lipschitz function. Then for each integer $m \geq 0$ there exists $\theta \in \mathbb{R}^{m+1}$ satisfying the following properties:

1. $\left|\theta_{[0]}\right| \leq K$ and $\left|\theta_{[j]}\right| \leq 2 L / j \forall j \in[m]$
2. $\max _{x \in[ \pm 1]}\left|f(x)-\sum_{j=0}^{m} \theta_{[j]} P_{j}(x)\right| \leq \frac{\pi+\frac{2}{\pi} \log (2 m+1)}{m+1} L$

Proof. Define $\theta_{[0]}=\frac{1}{\pi} \int_{-1}^{1} \frac{f(x)}{\sqrt{1-x^{2}}} d x$ and for each $j \in[m]$ let $\theta_{[j]}=\frac{2}{\pi} \int_{-1}^{1} \frac{f(x) P_{j}(x)}{\sqrt{1-x^{2}}} d x$ be the $j$ th Chebyshev coefficient. Since $\int_{-1}^{1} \frac{d x}{\sqrt{1-x^{2}}}=\pi$ we trivially have $\left|\theta_{[0]}\right| \leq K$ and by Trefethen (2008, Theorem 4.2) we also have

$$
\begin{equation*}
\left|\theta_{[j]}\right| \leq \frac{2}{\pi j} \int_{-1}^{1} \frac{\left|f^{\prime}(x)\right|}{\sqrt{1-x^{2}}} d x \leq \frac{2 L}{\pi j} \int_{-1}^{1} \frac{d x}{\sqrt{1-x^{2}}}=2 L / j \tag{16}
\end{equation*}
$$

for all $j \in[m]$. This shows the first property. For the second, by Trefethen (2008, Theorem 4.4) we have that

$$
\begin{align*}
\max _{x \in[-1,1]}\left|f(x)-\sum_{j=0}^{m} \theta_{[j]} P_{j}(x)\right| & \leq\left(2+\frac{4 \log (2 m+1)}{\pi^{2}}\right) \max _{x \in[ \pm 1]}\left|f(x)-p_{m}^{*}(x)\right| \\
& \leq\left(2+\frac{4 \log (2 m+1)}{\pi^{2}}\right) \frac{L \pi}{2(m+1)}=\frac{\pi+\frac{2}{\pi} \log (2 m+1)}{m+1} L \tag{17}
\end{align*}
$$

where $p_{m}^{*}$ is the (at most) $m$-degree algebraic polynomial that best approximates $f$ on $[ \pm 1]$ and the second inequality is Jackson's theorem (Cheney, 1982, page 147).
Corollary C.1. Let $f:[a, b] \mapsto[ \pm K]$ be a $K$-bounded, L-Lipschitz function on the interval $[a, b]$. Then for each integer $m \geq 0$ there exists $\theta \in \mathbb{R}^{m+1}$ satisfying the following properties:

1. $\left|\theta_{[0]}\right| \leq K$ and $\left|\theta_{[j]}\right| \leq \frac{L(b-a)}{j}$
2. $\max _{x \in[a, b]}\left|f(x)-\sum_{j=0}^{m} \theta_{[j]} P_{j}\left(\frac{2}{b-a}(x-a)-1\right)\right| \leq \frac{\pi+\frac{2}{\pi} \log (2 m+1)}{2(m+1)} L(b-a)$

Proof. Define $g(x)=f\left(\frac{b-a}{2}(x+1)+a\right)$, so that $g:[ \pm 1] \mapsto[ \pm K]$ is $K$-bounded and $L \frac{b-a}{2}$ Lipschitz. Applying Theorem C. 1 yields the result.

We next state regret guarantees for the SquareCB algorithm of Foster \& Rakhlin (2020) in the non-realizable setting:
Theorem C. 2 (Foster \& Rakhlin (2020, Theorem 5)). Suppose for any sequence of actions $a_{1}, \ldots, a_{T}$ an online regression oracle $\mathcal{A}$ playing regressors $h_{1}, \ldots, h_{T} \in \mathcal{H}$ has regret guarantee

$$
\begin{equation*}
R_{T} \geq \sum_{t=1}^{T}\left(\ell_{t}\left(c_{t}, a_{t}\right)-h_{t}\left(c_{t}, a_{t}\right)\right)^{2}-\min _{h \in \mathcal{H}} \sum_{t=1}^{T}\left(\ell_{t}\left(c_{t}, a_{t}\right)-h\left(c_{t}, a_{t}\right)\right)^{2} \tag{18}
\end{equation*}
$$

If all losses and regressors have range $[0,1]$ and $\exists h \in \mathcal{H}$ s.t. $\mathbb{E} \ell_{t}(a)=h\left(c_{t}, a\right)+\alpha_{t}\left(c_{t}, a\right)$ for $\left|\alpha_{t}(a)\right| \leq \alpha$ then Algorithm 6 with learning rate $\eta=2 \sqrt{d T /\left(R_{T}+2 \alpha^{2} T\right)}$ has expected regret w.r.t the the optimal policy $f:[a, b] \mapsto \mathbf{g}$ bounded as

$$
\begin{equation*}
\mathbb{E} \sum_{t=1}^{T} \ell_{t}\left(\mathbf{g}_{\left[i_{t}\right]}\right)-\sum_{t=1}^{T} \ell_{t}\left(h\left(c_{t}\right)\right) \leq 2 \sqrt{d T R_{T}}+5 \alpha T \sqrt{d} \tag{19}
\end{equation*}
$$

```
Algorithm 6: SquareCB method for contextual bandits using an online regression oracle.
Input: loss sequence $\left\{\ell_{t}: \mathbf{g} \mapsto[0,1]\right\}_{t=1}^{T}$, context sequence $\left\{c_{t}\right\}_{t=1}^{T}$, learning rate $\eta>0$,
online regression oracle $\mathcal{A}$
for $t=1, \ldots, T$ do
    $\mathbf{s}_{[i]} \leftarrow \mathcal{A}\left(c_{t}, \mathbf{g}_{[i]}\right) \forall i \in[d] \quad / /$ compute oracle prediction
    $i^{*} \leftarrow \arg \min _{i} \mathbf{s}_{[i]}$
    $\mathbf{p}_{[i]} \leftarrow \frac{1}{d+\eta\left(\mathbf{s}_{[i]}-\mathbf{s}_{\left[i^{*}\right]}\right)} \forall i \neq i^{*} \quad / /$ compute action probabilities
    $\mathbf{p}_{\left[i^{*}\right]} \leftarrow 1-\sum_{i \neq i^{*}} \mathbf{p}_{[i]}$
    sample $i_{t} \in[d]$ with probability $\mathbf{p}_{\left[i_{t}\right]}$ // sample index of a grid point
    $\left(\left(c_{t}, \mathbf{g}_{\left[i_{t}\right]}\right), \ell_{t}\left(\mathbf{g}_{\left[i_{t}\right]}\right)\right) \rightarrow \mathcal{A} / /$ pass context, action, and loss to oracle
```

```
Algorithm 7: SquareCB method for Lipschitz contextual bandits using Follow-the-Leader.
Input: loss sequence $\left\{\ell_{t}:[a, b] \mapsto[0, K]\right\}_{t=1}^{T}$, context sequence $\left\{c_{t} \in[c, c+C]\right\}_{t=1}^{T}$, learning
        rate $\eta>0$, action set $\mathbf{g} \in[a, b]^{d}$, featurizer $\mathbf{f}:[c, c+C] \mapsto \mathbb{R}^{m}$, normalizations $L, N>0$
for $t=1, \ldots, T$ do
    $\theta_{i} \leftarrow \underset{\left|\theta_{[0]}\right| \leq \frac{1}{N},\left|\theta_{[j]}\right| \leq \frac{2 C L}{K N j}}{\arg \min } \sum_{s \in[t-1]_{i}}\left(\left\langle\theta, \mathbf{f}\left(c_{s}\right)\right\rangle-\frac{\ell_{t}\left(\mathbf{g}_{\left[i_{s}\right]}\right)}{K N}\right)^{2} \forall i \in[d] / /$ update models
    $\mathbf{s}_{[i]} \leftarrow\left\langle\theta_{i}, \mathbf{f}\left(c_{t}\right)\right\rangle \forall i \in[d] \quad / /$ compute model predictions
    $i^{*} \leftarrow \arg \min _{i} \mathbf{s}_{[i]}$
    $\mathbf{p}_{[i]} \leftarrow \frac{1}{d+\eta\left(\mathbf{s}_{[i]}-\mathbf{s}_{\left[i^{*}\right]}\right)} \forall i \neq i^{*} \quad / /$ compute action probabilities
    $\mathbf{p}_{\left[i^{*}\right]}=1-\sum_{i \neq i^{*}} \mathbf{p}_{[i]}$
    sample $i_{t} \in[d]$ with probability $\mathbf{p}_{\left[i_{t}\right]}$ and play action $\mathbf{g}_{\left[i_{t}\right]}$
```

SquareCB requires an online regression oracle to implement, for which we will use the Follow-theLeader scheme. It has the following guarantee for squared losses:
Theorem C. 3 (Corollary of Hazan et al. (2007, Theorem 5)). Consider the follow-the-leader algorithm, which sequentially sees feature-target pairs $\left(\mathrm{x}_{1}, y_{1}\right), \cdots,\left(\mathrm{x}_{T}, y_{T}\right) \in \mathcal{X} \times[0,1]$ for some subset $\mathcal{X} \subset[0,1]^{n}$ and at each step sets $\theta_{t+1}=\arg \min _{\theta \in \Theta} \sum_{t=1}^{T}\left(\left\langle\mathbf{x}_{t}, \theta\right\rangle-y_{t}\right)^{2}$ for some subset $\Theta \subset \mathbb{R}^{n}$. This algorithm has regret

$$
\begin{equation*}
\sum_{t=1}^{T}\left(\left\langle\mathbf{x}_{t}, \theta_{t}\right\rangle-y_{t}\right)^{2}-\min _{\theta \in \Theta}\left(\left\langle\mathbf{x}_{t}, \theta\right\rangle-y_{t}\right)^{2} \leq 4 B^{2} n\left(1+\log \frac{X D T}{2 B}\right) \tag{20}
\end{equation*}
$$

for $D_{\Theta}$ the diameter $\max _{\theta, \theta^{\prime}}\left\|\theta-\theta^{\prime}\right\|_{2}$ of $\Theta, X=\max _{t \in[T]}\left\|\mathrm{x}_{t}\right\|_{2}$, and $B=\max _{t \in[T], \theta \in \Theta}\left|\left\langle\mathrm{x}_{t}, \theta\right\rangle\right|$.

## C. 2 Regret of ChebCB

Theorem C.4. Suppose $\mathbb{E} \ell_{t}(x)$ is an $L_{x}$-Lipschitz function of actions $x \in[a, b]$ and an $L_{c}$ Lipschitz function of contexts $c_{t} \in[c, c+C]$. Then Algorithm 7 run with learning rate $\eta=2 \sqrt{d T /\left(R_{T}+2 \alpha^{2} T\right)}$ for $R_{T}$ and $\alpha$ as in Equations 22 and 23, respectively, action set $\mathbf{g}_{[i]}=a+(b-a) \frac{i-1 / 2}{d}$, Chebyshev features $\mathbf{f}_{[j]}\left(c_{t}\right)=P_{j}\left(c_{t}\right)$, and normalizations $L=L_{c}$ and $N=2+\frac{4 C L_{c}}{K}(1+\log m)$ has regret w.r.t. any policy $f:[c, c+C] \mapsto[a, b]$ of

$$
\begin{equation*}
\mathbb{E} \sum_{t=1}^{T} \ell_{t}\left(\mathbf{g}_{\left[i_{t}\right]}\right)-\ell_{t}\left(f\left(c_{t}\right)\right)=\tilde{\mathcal{O}}\left(L_{c} d \sqrt{m T}+\frac{L_{c} T \sqrt{d}}{m}+\frac{L_{x} T}{d}\right) \tag{21}
\end{equation*}
$$

Setting $d=\Theta\left(T^{2 / 11}\right)$ and $m=\Theta\left(T^{3 / 11}\right)$ yields a regret $\tilde{\mathcal{O}}\left(\max \left\{L_{c}, L_{x}\right\} T^{9 / 11}\right)$.

Proof. Observe that the above algorithm is equivalent to running Algorithm 6 with the follow-theleader oracle over an $d(m+1)$-dimensional space $\Theta$ with diameter $\sqrt{\frac{d}{N^{2}}\left(1+\frac{4 C^{2} L_{c}^{2}}{K^{2}} \sum_{j=1}^{m} \frac{1}{j^{2}}\right)} \leq \frac{\sqrt{d K^{2}+2 d C^{2} L_{c}^{2} \pi^{2} / 3}}{K N}$, features bounded by $\sqrt{1+\sum_{j=1}^{m} P_{j}\left(c_{t}\right)} \leq \sqrt{m+1}$, and predictions bounded by $|\langle\mathbf{f}(c), \theta\rangle| \leq\|\theta\|_{1}\|\mathbf{f}(c)\|_{\infty} \leq \frac{1}{N}+\frac{2 C L_{c}}{K N} \sum_{j=1}^{m} \leq \frac{1}{2}$. Thus by Theorem C. 3 the oracle has regret at most

$$
\begin{equation*}
R_{T}=d(m+1)\left(1+\log \frac{T \sqrt{d(m+1)\left(K^{2}+2 C^{2} L_{c}^{2} \pi^{2} / 3\right)}}{K N}\right) \tag{22}
\end{equation*}
$$

Note that, to ensure the regressors and losses have range in $[0,1]$ we can define the former as $h\left(c, \mathbf{g}_{[i]}\right)=\left\langle\mathbf{f}(c), \theta_{i}\right\rangle+\frac{1}{2}$ and the latter as $\frac{\ell_{t}}{K N}+\frac{1}{2}$ and Algorithm 7 remains the same. Furthermore, the error of the regression approximation is then

$$
\begin{equation*}
\alpha=\frac{\pi+\frac{2}{\pi} \log (2 m+1)}{2 K N(m+1)} C L_{c} \tag{23}
\end{equation*}
$$

We conclude by applying Theorem C.2, unnormalizing by multiplying the resulting regret by $K N$, and adding the approximation error $\frac{L_{x}(b-a)}{2 d}$ due to the discretization of the action space. $\square$

## D SOR preliminaries

We will use the following notation:

- $\mathbf{M}_{\omega}=\mathbf{I}_{n}-(\mathbf{D} / \omega+\mathbf{L})^{-1} \mathbf{A}$ is the matrix of the first normal form (Hackbusch, 2016, 2.2.1)
- $\mathbf{W}_{\omega}=\mathbf{D} / \omega+\mathbf{L}$ is the matrix of the third normal form (Hackbusch, 2016, Section 2.2.3)
- $\mathbf{C}_{\omega}=\mathbf{I}_{n}-\mathbf{A}(\mathbf{D} / \omega+\mathbf{L})^{-1}=\mathbf{I}_{n}-\mathbf{A} \mathbf{W}_{\omega}^{-1}=\mathbf{A} \mathbf{M}_{\omega} \mathbf{A}^{-1}$ is the defect reduction matrix
- $\breve{\mathbf{M}}_{\omega}=\mathbf{I}_{n}-\frac{2-\omega}{\omega}\left(\mathbf{D} / \omega+\mathbf{L}^{T}\right)^{-1} \mathbf{D}(\mathbf{D} / \omega+\mathbf{L})^{-1} \mathbf{A}$ is the matrix of the first normal form for SSOR (Hackbusch, 2016, 2.2.1)
- $\breve{\mathbf{W}}_{\omega}=\frac{\omega}{2-\omega}(\mathbf{D} / \omega+\mathbf{L}) \mathbf{D}^{-1}\left(\mathbf{D} / \omega+\mathbf{L}^{T}\right)$ is the matrix of the third normal form for SSOR (Hackbusch, 2016, Section 2.2.3)
- $\breve{\mathbf{C}}_{\omega}=\mathbf{I}_{n}-\frac{2-\omega}{\omega} \mathbf{A}\left(\mathbf{D} / \omega+\mathbf{L}^{T}\right)^{-1} \mathbf{D}(\mathbf{D} / \omega+\mathbf{L})^{-1}=\mathbf{I}_{n}-\mathbf{A} \breve{\mathbf{W}}_{\omega}^{-1}=\mathbf{A} \breve{\mathbf{M}}_{\omega} \mathbf{A}^{-1}$ is the defect reduction matrix for SSOR
- $\|\cdot\|_{2}$ denotes the Euclidean norm of a vector and the spectral norm of a matrix
- $\kappa(\mathbf{A})=\|\mathbf{A}\|_{2}\left\|\mathbf{A}^{-1}\right\|_{2}$ denotes the condition number of a matrix $\mathbf{A} \succ 0$
- $\rho(\mathbf{X})$ denotes the spectral radius of a matrix $\mathbf{X}$
- $\|\mathbf{x}\|_{\mathbf{A}}=\left\|\mathbf{A}^{\frac{1}{2}} \mathbf{x}\right\|_{2}$ denotes the energy norm of a vector $\mathbf{x} \in \mathbb{R}^{n}$ associated with the matrix $\mathbf{A} \succ 0$
- $\|\mathbf{X}\|_{\mathbf{A}}=\left\|\mathbf{A}^{\frac{1}{2}} \mathbf{X} \mathbf{A}^{-\frac{1}{2}}\right\|_{2}$ denotes the energy norm of a matrix $\mathbf{X} \in \mathbb{R}^{n \times n}$ associated with the matrix $\mathbf{A} \succ 0$

We further derive bounds on the number of iterations for SOR and SSOR using the following energy norm estimate:
Theorem D. 1 (Corollary of Hackbusch (2016, Theorem 3.44 \& Corollary 3.45)). If $\mathbf{A} \succ 0$ and $\omega \in(0,2)$ then $\left\|\mathbf{M}_{\omega}\right\|_{\mathbf{A}}^{2} \leq 1-\frac{\frac{2-\omega}{\omega} \gamma}{\left(\frac{2-\omega}{2 \omega}\right)^{2}+\frac{\gamma}{\omega}+\rho\left(\mathbf{D}^{-1} \mathbf{L D}^{-1} \mathbf{L}^{T}\right)-\frac{1}{4}}$, where $\gamma=1-\rho\left(\mathbf{D}^{-1}\left(\mathbf{L}+\mathbf{L}^{T}\right)\right)$.
Corollary D.1. Let $K_{\omega}$ be the maximum number of iterations that $S O R$ needs to reach error $\varepsilon>0$. Then for any $\omega \in(0,2)$ we have $K_{\omega} \leq 1+\frac{-\log \frac{\varepsilon}{2 \sqrt{\kappa(A)}}}{-\log \nu_{\omega}(\mathbf{A})}$, where $\nu_{\omega}(\mathbf{A})$ is the square root of the upper bound in Theorem D.1.

Proof. By Hackbusch (2016, Equations 2.22c \& B.28b) we have at each iteration $k$ of Algorithm 1 that

$$
\begin{equation*}
\frac{\left\|\mathbf{r}_{k}\right\|_{2}}{\left\|\mathbf{r}_{0}\right\|_{2}} \leq \frac{2}{\left\|\mathbf{r}_{0}\right\|_{2}}\left\|\mathbf{A}^{\frac{1}{2}}\right\|_{2}\left\|\mathbf{M}_{\omega}^{k}\right\|_{\mathbf{A}}\left\|\mathbf{A}^{-\frac{1}{2}} \mathbf{r}_{0}\right\|_{2} \leq 2 \sqrt{\kappa(\mathbf{A})}\left\|\mathbf{M}_{\omega}\right\|_{\mathbf{A}}^{k} \tag{24}
\end{equation*}
$$

Setting the r.h.s. equal to $\varepsilon$ and solving for $k$ yields the result.
Corollary D.2. Let $\breve{K}_{\omega}$ be the maximum number of iterations that SSOR (Algorithm 8) needs to reach (absolute) error $\varepsilon>0$. Then for any $\omega \in(0,2)$ we have $\breve{K}_{\omega} \leq 1+\frac{-\log \frac{\varepsilon}{2\|\mathbf{b}\|_{2} \sqrt{\kappa(A)}}}{-2 \log \nu_{\omega}(\mathbf{A})}$, where $\nu_{\omega}(\mathbf{A})$ is the square root of the upper bound in Theorem D.1.

Proof. By Hackbusch (2016, Equations 2.22c \& B.28b) we have at each iteration $k$ of Algorithm 1 that

$$
\begin{equation*}
\left\|\mathbf{r}_{k}\right\|_{2} \leq 2\left\|\mathbf{A}^{\frac{1}{2}}\right\|_{2}\left\|\breve{\mathbf{M}}_{\omega}^{k}\right\|_{\mathbf{A}}\left\|\mathbf{A}^{-\frac{1}{2}} \mathbf{r}_{0}\right\|_{2} \leq 2\|\mathbf{b}\|_{2} \sqrt{\kappa(\mathbf{A})}\left\|\mathbf{M}_{\omega}\right\|_{\mathbf{A}}^{2 k} \tag{25}
\end{equation*}
$$

Setting the r.h.s. equal to $\varepsilon$ and solving for $k$ yields the result.

![](https://cdn.mathpix.com/cropped/2025_11_24_6122a0dad929fc569471g-23.jpg?height=502&width=633&top_left_y=276&top_left_x=739)
Figure 3: Values of $\tau$ and $\beta$ for $\mathbf{A}+c \mathbf{I}_{n}$ for different $c$.

## E Near-asymptotic proofs

## E. 1 Proof of Lemma 2.1 and associated discussion

Proof. For the first claim, suppose $l=\min _{\left\|\mathbf{C}_{\omega}^{k} \mathbf{b}\right\|_{2}<\varepsilon\|\mathbf{b}\|_{2}} k>U(\omega)$. Then

$$
\begin{equation*}
\varepsilon \leq \frac{\left\|\mathbf{C}_{\omega}^{l-1} \mathbf{b}\right\|_{2}}{\|\mathbf{b}\|_{2}} \leq \frac{\left\|\mathbf{C}_{\omega}^{l-1}\right\|_{2}\|\mathbf{b}\|_{2}}{\|\mathbf{b}\|_{2}} \leq\left(\rho\left(\mathbf{C}_{\omega}\right)+\tau\left(1-\rho\left(\mathbf{C}_{\omega}\right)\right)\right)^{l-1}<\varepsilon \tag{26}
\end{equation*}
$$

so by contradiction we must have $\min _{\left\|\mathbf{C}_{\omega}^{k} \mathbf{b}\right\|_{2}<\varepsilon\|\mathbf{b}\|_{2}} k \leq U(\omega)$. Now note that by Hackbusch (2016, Theorem 4.27) and similarity of $\mathbf{C}_{\omega}$ and $\mathbf{M}_{\omega}$ we have that $\rho\left(\mathbf{C}_{\omega}\right)=\frac{1}{4}\left(\omega \beta+\sqrt{\omega^{2} \beta^{2}-4(\omega-1)}\right)^{2}$ for $\omega<\omega^{*}=1+\left(\frac{\beta}{1+\sqrt{1-\beta^{2}}}\right)^{2}$ and $\omega-1$ otherwise. Therefore on $\omega<\omega^{*}$ we have $\rho\left(\mathbf{C}_{\omega}\right) \leq \rho\left(\mathbf{C}_{1}\right)=\beta^{2}$ and on $\omega \geq \omega^{*}$ we have $\rho\left(\mathbf{C}_{\omega}\right) \leq \omega_{\text {max }}-1$. This concludes the second part of the first claim. The first part of the second claim follows because $\rho\left(\mathbf{C}_{\omega}\right)$ is decreasing on $\omega<\omega^{*}$. For the second part, we compute the derivative $\left|\partial_{\omega} U(\omega)\right|=\frac{(\tau-1) \log \varepsilon}{(\tau+(1-\tau)(\omega-1)) \log ^{2}(\tau+(1-\tau)(\omega-1))}$. Since $\tau+(1-\tau)(\omega-1) \geq \frac{1}{e^{2}}$ by assumption-either by nonnegativity of $\omega-1$ if $\tau \geq \frac{1}{e^{2}}$ or because otherwise $\beta^{2} \geq \frac{4}{e^{2}}\left(1-\frac{1}{e^{2}}\right)$ implies $\tau+(1-\tau)(\omega-1) \geq\left(1-\frac{1}{e^{2}}\right)\left(\frac{\beta}{1+\sqrt{1-\beta^{2}}}\right)^{2} \geq \frac{1}{e^{2}}$-the derivative is increasing in $\omega$ and so is at most $\frac{-(1-\tau) \log \varepsilon}{\alpha \log ^{2} \alpha}$. $\square$

Note that the fourth item's restriction on $\tau$ and $\beta$ does not really restrict the matrices our analysis is applicable to, as we can always re-define $\tau$ in Assumption 2.1 to be at least $1 / e^{2}$. Our analysis does not strongly depend on this restriction; it is largely done for simplicity and because it does not exclude too many settings of interest. In-particular, we find for $\varepsilon \geq 10^{-8}$ that $\tau$ is typically indeed larger than $\frac{1}{e^{2}}$, and furthermore $\tau$ is likely quite high whenever $\beta$ is small, as it suggests the matrix is near-diagonal and so $\omega$ near one will converge extremely quickly (c.f. Figure 3).

## E. 2 Proof of Theorem 2.1

Proof. The first bound follows from Theorem B. 3 by noting that Lemma 2.1 implies that the functions $U_{t}-1$ are $\left(\frac{-\left(1-\tau_{t}\right) \log \varepsilon}{\alpha_{t} \log ^{2} \alpha_{t}}, \omega_{\text {max }}\right)$-semi-Lipschitz over $\left[1, \omega_{\text {max }}\right]$ and the functions $\operatorname{SOR}_{t}-1 \leq U_{t}-1$ are $\frac{-\log \varepsilon}{-\log \alpha_{t}}$-bounded. To extend the comparator domain to $\left(0, \omega_{\text {max }}\right]$, note that Lemma 2.1.2 implies that all $U_{t}$ are decreasing on $\omega \in(0,1)$. To extend the comparator domain again in the second bound, note that the setting of $\omega_{\text {max }}$ implies that the minimizer $1+\beta_{t}^{2} /\left(1+\sqrt{1-\beta_{t}^{2}}\right)$ of each $U_{t}$ is at most $\omega_{\text {max }}$, and so all functions $U_{t}$ are increasing on $\omega \in\left(\omega_{\text {max }}, 2\right)$. $\square$

## E. 3 Approximating the optimal policy

Lemma E.1. Define $\mathbf{A}(c)=\mathbf{A}+c \mathbf{I}_{n}$ for all $c \in\left[c_{\text {min }}, \infty\right)$, where $c_{\text {min }}>-\lambda_{\text {min }}(\mathbf{A})$. Then $\omega^{*}(c)= 1+\left(\frac{\beta_{c}}{\sqrt{1-\beta_{c}^{2}}+1}\right)^{2}$ is $\frac{6 \beta_{\text {max }}\left(1+\beta_{\text {max }}\right) / \sqrt{1-\beta_{\text {max }}^{2}}}{\left(\sqrt{1-\beta_{\text {max }}^{2}}+1\right)^{2}}\left(\frac{\lambda_{\text {min }}(\mathbf{D})+c_{\text {min }}+1}{\lambda_{\text {min }}(\mathbf{D})+c_{\text {min }}}\right)^{2}$-Lipschitz, where $\beta_{\text {max }}=\max _{c} \beta_{c}$ is the maximum over $\beta_{c}=\rho\left(\mathbf{I}_{n}-\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-1}\left(\mathbf{A}+c \mathbf{I}_{n}\right)\right)$.

Proof. We first compute

$$
\begin{align*}
\partial_{c} \beta_{c}= & \partial_{c} \rho\left(\mathbf{I}_{n}-\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-1}\left(\mathbf{A}+c \mathbf{I}_{n}\right)\right) \\
= & \partial_{c} \lambda_{\max }\left(\mathbf{I}_{n}-\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}}\left(\mathbf{A}+c \mathbf{I}_{n}\right)\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}}\right) \\
= & \mathbf{v}_{1}^{T} \partial_{c}\left(\left(\mathbf{I}_{n}-\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}}\left(\mathbf{A}+c \mathbf{I}_{n}\right)\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}}\right)\right) \mathbf{v}_{1} \\
= & -\mathbf{v}_{1}^{T}\left(\partial_{c}\left(\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}}\right)\left(\mathbf{A}+c \mathbf{I}_{n}\right)\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}}\right. \\
& \quad+\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}} \partial_{c}\left(\mathbf{A}+c \mathbf{I}_{n}\right)\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}} \\
& \left.\quad+\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}}\left(\mathbf{A}+c \mathbf{I}_{n}\right) \partial_{c}\left(\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}}\right)\right) \mathbf{v}_{1} \\
= & \frac{1}{2} \mathbf{v}_{1}^{T}\left(\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{3}{2}}\left(\mathbf{A}+c \mathbf{I}_{n}\right)\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}}-2 c\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-1}\right. \\
& \left.\quad+\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}}\left(\mathbf{A}+c \mathbf{I}_{n}\right)\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{3}{2}}\right) \mathbf{v}_{1} \\
= & \frac{1}{2} \mathbf{v}_{1}^{T}\left(\mathbf{I}_{n}+\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-1}\right)\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}}(\mathbf{A}+c \mathbf{I})\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}}\left(\mathbf{I}_{n}+\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-1}\right) \mathbf{v}_{1} \\
& \quad-\frac{1}{2} \mathbf{v}_{1}^{T}\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}}\left(\mathbf{A}+c \mathbf{I}_{n}\right)\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}} \mathbf{v}_{1} \\
& \quad-\frac{1}{2} \mathbf{v}_{1}^{T}\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{3}{2}}\left(\mathbf{A}+c \mathbf{I}_{n}\right)\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{3}{2}} \mathbf{v}_{1}-c \mathbf{v}_{1}^{T}\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-1} \mathbf{v}_{1} \\
= & \frac{1}{2} \mathbf{v}_{1}^{T}\left(\mathbf{I}_{n}+\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-1}\right)\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}}(\mathbf{A}+c \mathbf{I})\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}}\left(\mathbf{I}_{n}+\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-1}\right) \mathbf{v}_{1} \\
& \quad-\frac{1}{2} \mathbf{v}_{1}^{T}\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}}\left(\mathbf{A}+3 c \mathbf{I}_{n}\right)\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{1}{2}} \mathbf{v}_{1} \\
& \quad-\frac{1}{2} \mathbf{v}_{1}^{T}\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{3}{2}}\left(\mathbf{A}+c \mathbf{I}_{n}\right)\left(\mathbf{D}+c \mathbf{I}_{n}\right)^{-\frac{3}{2}} \mathbf{v}_{1} \tag{27}
\end{align*}
$$

The first component is positive and the matrix has eigenvalues bounded by $\left(1+\frac{1}{\lambda_{\min }(\mathbf{D})+c}\right)^{2} \frac{1+\beta_{c}}{2}$, while the last term is negative and the matrix has If $c \geq 0$ the positive component has spectral radius at most $\frac{1+\beta_{c}}{2}\left(1+\frac{1}{\left(\lambda_{\text {min }}(\mathbf{D})+c\right)^{2}}\right)$. If the middle term is negative, subtracting $2 \mathbf{A}$ from the middle matrix shows that its magnitude is bounded by $\frac{3}{2}\left(1+\beta_{c}\right)$. If the middle term is positive-which can only happen for negative $c$-its magnitude is bounded by $\frac{-3 c / 2}{\lambda_{\min }(\mathbf{D})+c} \leq \frac{3 \lambda_{\min }(\mathbf{D}) / 2}{\lambda_{\min }(\mathbf{D})+c}$. Combining all terms yields a bound of $3\left(\frac{\lambda_{\text {min }}(\mathbf{D})+c+1}{\lambda_{\text {min }}(\mathbf{D})+c}\right)^{2}\left(1+\beta_{c}\right)$. We then have that

$$
\begin{equation*}
\left|\partial_{c} \omega^{*}(c)\right|=\frac{2 \beta_{c} / \sqrt{1-\beta_{c}^{2}}}{\left(\sqrt{1-\beta_{c}^{2}}+1\right)^{2}}\left|\partial_{c} \beta_{c}\right|=\frac{6 \beta_{c}\left(1+\beta_{c}\right) / \sqrt{1-\beta_{c}^{2}}}{\left(\sqrt{1-\beta_{c}^{2}}+1\right)^{2}}\left(\frac{\lambda_{\min }(\mathbf{D})+c+1}{\lambda_{\min }(\mathbf{D})+c}\right)^{2} \tag{28}
\end{equation*}
$$

The result follows because $\frac{2 x(1+x) / \sqrt{1-x^{2}}}{\left(\sqrt{1-x^{2}}+1\right)^{2}}$ increases monotonically on $x \in[0,1)$ and the bound itself decreases monotonically in $c$

![](https://cdn.mathpix.com/cropped/2025_11_24_6122a0dad929fc569471g-25.jpg?height=502&width=1356&top_left_y=271&top_left_x=382)
Figure 4: Comparison of actual cost of running SSOR-preconditioned CG and the upper bounds computed in Section E. 4 as functions of the tuning parameter $\omega \in[2 \sqrt{2}-2,1.9]$ on various domains.

Theorem E.1. Suppose $c_{t} \in\left[c_{\text {min }}, c_{\text {min }}+C\right] \forall t \in[T]$, where $c_{\text {min }}>-\lambda_{\text {min }}(\mathbf{A})$, and define $\beta_{\text {max }}= \max _{t} \beta_{t}$. Then if we run Algorithm 5 with losses $\left(\operatorname{SOR}_{t}(\cdot)-1\right) / K$ normalized by $K \geq \frac{-\log \varepsilon}{-\log \alpha_{\max }}$ for $\alpha_{\text {max }}=\max _{t} \alpha_{t}$, action set $\mathbf{g}_{[i]}=1+\left(\omega_{\text {max }}-1\right) \frac{i}{d}$ for $\omega_{\text {max }} \geq 1+\left(\frac{\beta_{\text {max }}}{1+\sqrt{1-\beta_{\text {max }}^{2}}}\right)^{2}$, and context discretization $\mathbf{h}_{[j]}=c_{\text {min }}+\frac{C}{m}\left(j-\frac{1}{2}\right)$, then the number of iterations will be bounded in expectation as

$$
\begin{equation*}
\mathbb{E} \sum_{t=1}^{T} \operatorname{SOR}_{t}\left(\omega_{t}\right) \leq m+4 K \sqrt{d m T}+\left(\frac{C L^{*} / m}{\omega_{\max }-1}+\frac{1}{d}\right) \sum_{t=1}^{T} \frac{-\log \varepsilon}{\log ^{2} \alpha_{t}}+\sum_{t=1}^{T} U_{t}\left(\omega^{*}\left(c_{t}\right)\right) \tag{29}
\end{equation*}
$$

where $L^{*}$ is the Lipschitz constant from Lemma E.1. Setting $\omega_{\max }=1+\left(\frac{\beta_{\max }^{2}}{1+\sqrt{1-\beta_{\max }^{2}}}\right)^{2}$, $K=\frac{-\log \varepsilon}{-\log \alpha_{\text {max }}}, d=\sqrt[4]{\frac{\bar{\gamma}^{2} T \log ^{2} \alpha_{\text {max }}}{24 C L}}$, and $m=\sqrt[4]{54 C^{3} L^{3} \bar{\gamma}^{2} T \log ^{2} \alpha_{\text {max }}}$ where $\bar{\gamma}=\frac{1}{T} \sum_{t=1}^{T} \frac{1}{\log ^{2} \alpha_{t}}$ and $\tilde{L}=\left(\frac{\lambda_{\text {min }}(\mathbf{D})+c_{\text {min }}+1}{\lambda_{\text {min }}(\mathbf{D})+c_{\text {min }}}\right)^{2} \frac{1+\beta_{\text {max }}}{\beta_{\text {max }} \sqrt{1-\beta_{\text {max }}^{2}}}$, yields

$$
\begin{align*}
E \sum_{t=1}^{T} \operatorname{SOR}_{t}\left(\omega_{t}\right) & \leq \sqrt[4]{54 C^{3} L^{3} \bar{\gamma}^{2} T \log ^{2} \alpha_{\max }}+4 \log \frac{1}{\varepsilon} \sqrt[4]{\frac{24 C L \bar{\gamma} T^{3}}{\log ^{2} \alpha_{\max }}}+\sum_{t=1}^{T} U_{t}\left(\omega^{*}\left(c_{t}\right)\right) \\
& \leq \sqrt[4]{\frac{54 C^{3} L^{3} T}{\log ^{2} \alpha_{\max }}}+\frac{4 \log \frac{1}{\varepsilon}}{\log \frac{1}{\alpha_{\max }}} \sqrt[4]{24 C L T^{3}}+\sum_{t=1}^{T} U_{t}\left(\omega^{*}\left(c_{t}\right)\right) \tag{30}
\end{align*}
$$

Proof. The bound follows from Theorem B. 4 by noting that Lemma 2.1 implies that the functions $U_{t}-1$ are $\left(\frac{-\left(1-\tau_{t}\right) \log \varepsilon}{\alpha_{t} \log ^{2} \alpha_{t}}, \omega_{\text {max }}\right)$-semi-Lipschitz over $\left[1, \omega_{\text {max }}\right]$ and the functions $\operatorname{SOR}_{t}-1 \leq U_{t}-1$ are $\frac{-\log \varepsilon}{-\log \alpha_{t}}$-bounded. Note that for the choice of $\omega_{\text {max }}$ we the interval $\left[1, \omega_{\text {max }}\right]$ contains the range of the optimal policy $\omega^{*}$, and further by Lemma E. 1 it is $L^{*}$-Lipschitz over $\left[c_{\text {min }}, c_{\text {min }}+C\right]$. $\square$

## E. 4 Extension to preconditioned CG

While CG is an iterative algorithm, for simplicity we define it as the solution to a minimization problem in the Krylov subspace:
Definition E.1. $C G(\mathbf{A}, \mathbf{b}, \omega)=\min _{\left\|\mathbf{A} \mathbf{x}_{k}-\mathbf{b}\right\|_{2} \leq \varepsilon} k$ for $\mathbf{x}_{k}=\underset{\mathbf{x}=\mathbf{P}_{k}\left(\breve{\mathbf{W}}_{\omega}^{-1} \mathbf{A}\right) \breve{\mathbf{W}}_{\omega}^{-1} \mathbf{b}}{\arg \min }\left\|\mathbf{x}-\mathbf{A}^{-1} \mathbf{b}\right\|_{\mathbf{A}}$ where the minimum is taken over all degree $k$ polynomials $\mathbf{P}_{k}: \mathbb{R}^{n \times n} \mapsto \mathbb{R}^{n \times n}$.

Lemma E.2. Let $\mathbf{A}$ be a positive-definite matrix and $\mathbf{b} \in \mathbb{R}^{n}$ any vector. Define

$$
\begin{equation*}
U^{C G}(\omega)=1+\frac{\tau \log \left(\frac{\sqrt{\kappa(\mathbf{A})}}{\varepsilon}+\sqrt{\frac{\kappa(\mathbf{A})}{\varepsilon^{2}}-1}\right)}{-\log \left(1-\frac{4}{2+\sqrt{\frac{4}{2-\omega}+\frac{\mu(2-\omega)}{\omega}+\frac{4 \nu \omega}{2-\omega}}}\right)} \tag{31}
\end{equation*}
$$

for $\mu=\lambda_{\max }\left(\mathbf{D A}^{-1}\right) \geq 1, \nu=\lambda_{\max }\left(\left(\mathbf{L D}^{-1} \mathbf{L}^{T}-\mathbf{D} / 4\right) \mathbf{A}^{-1}\right) \in[-1 / 4,0]$, and $\tau$ the smallest constant (depending on $\mathbf{A}$ and $\mathbf{b}$ ) s.t. $U^{C G} \geq C G(\mathbf{A}, \mathbf{b}, \cdot)$. Then the following holds

1. $\tau \in(0,1]$
2. if $\mu>1$ then $U^{C G}$ is minimized at $\omega^{*}=\frac{2}{1+\sqrt{\frac{2}{\mu}(1+2 \nu)}}$ and monotonically increases away from $\omega^{*}$ in both directions
3. $U^{C G}$ is $\left(\frac{\mu+4 \nu+4}{4 \mu \nu+2 \mu-1} \tau \sqrt{\mu \sqrt{2}}, 2 \sqrt{2}-2\right)$-semi-Lipschitz on $[2 \sqrt{2}-2,2)$
4. if $\mu \leq \mu_{\text {max }}$ then $U^{C G} \leq 1+\frac{\tau \log \left(\frac{2}{\varepsilon} \sqrt{\kappa(\mathbf{A})}\right)}{-\log \left(1-\frac{2}{1+\sqrt{\gamma}}\right)}$ on $\left[2 \sqrt{2}-2, \frac{2}{1+1 / \sqrt{\mu_{\text {max }}}}\right]$, where $\gamma \leq \frac{7+3 \mu_{\text {max }}}{8}$.

Proof. By Hackbusch (2016, Theorem 10.17) we have that the $k$ th residual of SSOR-preconditioned CG satisfies

$$
\begin{align*}
\left\|\mathbf{r}_{k}(\omega)\right\|_{2}=\left\|\mathbf{b}-\mathbf{A} \mathbf{x}_{k}\right\|_{2} \leq \sqrt{\|\mathbf{A}\|}\left\|\mathbf{A}^{-1} \mathbf{b}-\mathbf{x}_{k}\right\|_{\mathbf{A}} & \leq \sqrt{\|\mathbf{A}\|} \frac{2 x^{k}}{1+x^{2 k}}\left\|\mathbf{A}^{-1} \mathbf{b}-\mathbf{x}_{0}\right\|_{\mathbf{A}} \\
& \leq \frac{2 \sqrt{\kappa(\mathbf{A})} x^{k}}{1+x^{2 k}}\left\|\mathbf{r}_{0}\right\|_{2} \tag{32}
\end{align*}
$$

for $x=\frac{\sqrt{\kappa\left(\breve{\mathbf{W}}_{\omega}^{-1} \mathbf{A}\right)}-1}{\sqrt{\kappa\left(\breve{\mathbf{W}}_{\omega}^{-1} \mathbf{A}\right)}+1}=1-\frac{2}{\sqrt{\kappa\left(\breve{\mathbf{W}}_{\omega}^{-1} \mathbf{A}\right)}+1}$. By Axelsson (1994, Theorem 7.17) we have

$$
\begin{equation*}
\kappa\left(\breve{\mathbf{W}}_{\omega}^{-1} \mathbf{A}\right) \leq \frac{1+\frac{\mu}{4 \omega}(2-\omega)^{2}+\omega \nu}{2-\omega} \tag{33}
\end{equation*}
$$

Combining the two inequalities above yields the first result. For the second, we compute the derivative w.r.t. $\omega$ :

$$
\begin{equation*}
\frac{\partial_{\omega} U^{\mathrm{CG}}}{\tau}=\frac{8(2 \nu+1) \omega^{2}-4 \mu(2-\omega)^{2}}{(2-\omega) \omega \sqrt{\frac{4}{2-\omega}+\frac{\mu(2-\omega)}{\omega}+\frac{4 \nu \omega}{2-\omega}}\left(\mu(2-\omega)^{2}+4 \omega(\nu \omega+\omega-1)\right)} \tag{34}
\end{equation*}
$$

Since $\nu \in[-1 / 4,0]$ and $\mu>1$, we have that $\mu(2-\omega)^{2}+4 \omega(\nu \omega+\omega-1) \geq(2-\omega)^{2}+3 \omega^{2}-4 \omega$, which is nonnegative. Therefore the derivative only switches signs once, at the zero of specified in the second result. The monotonic increase property follows by positivity of the numerator on $\omega>\omega^{*}$. The third property follows by noting that since $U^{\mathrm{CG}}$ is increasing on $\omega>\omega^{*}$ we only needs to consider $\omega \in\left[2 \sqrt{2}-2, \omega^{*}\right]$, where the numerator of the derivative is negative; here we have

$$
\begin{equation*}
\frac{\left|\partial_{\omega} U^{\mathrm{CG}}\right|}{\tau} \leq \frac{4 \mu(2-\omega)}{\omega \sqrt{\frac{4}{2-\omega}+\frac{\mu(2-\omega)}{\omega}+\frac{4 \nu \omega}{2-\omega}}\left(\mu(2-\omega)^{2}+4 \omega(\nu \omega+\omega-1)\right)} \leq \frac{\mu+4 \nu+4}{4 \mu \nu+2 \mu-1} \sqrt{\mu \sqrt{2}} \tag{35}
\end{equation*}
$$

where we have used $\mu(2-\omega)^{2}+4 \omega(\nu \omega+\omega-1) \geq \frac{16 \mu \nu+8 \mu-4}{\mu+4 \nu+4}$ and $\omega \geq 2 \sqrt{2}-2$. For the last result we use the fact that $\kappa\left(\breve{\mathbf{W}}_{\omega}^{-1} \mathbf{A}\right)$ is maximal at the endpoints of the interval and evaluate it on those endpoints to bound $\gamma \leq \frac{1}{2}+\frac{\max \left\{\left(\mu_{\text {max }}+1\right) \sqrt{2}, 3 \sqrt{\mu_{\text {max }}}\right\}}{4} \leq \frac{1}{2}+\frac{3\left(\mu_{\text {max }}+1\right)}{8}$.

We plot the bounds from Lemma E. 2 in Figure 4. Note that $\tau \in(0,1]$ is an instance-dependent parameter that is defined to effectively scale down the function as much as possible while still being an upper bound on the cost; it thus allows us to exploit the shape of the upper bound without having it be too loose. This is useful since upper bounds for CG are known to be rather pessimistic, and we are able to do this because our learning algorithms do not directly access the upper bound anyway. Empirically, we find $\tau$ to often be around 3/4 or larger.

## E.4.1 Proof of Theorem 2.3

Proof. By Lemma E. 2 the functions $U_{t}-1 \geq \mathrm{CG}_{t}-1$ are $\left(\frac{\mu_{t}+4 \nu_{t}+4}{4 \mu_{t} \nu_{t}+2 \mu_{t}-1} \sqrt{\mu_{t} \sqrt{2}}, 2 \sqrt{2}-2\right)$-semiLipschitz and $\frac{\log \left(\frac{2}{\varepsilon} \sqrt{\kappa_{\text {max }}}\right)}{\log \frac{\sqrt{6 \mu_{\text {max }}+14+4}}{\sqrt{6 \mu_{\text {max }}+14-4}}}$-bounded on $\left[2 \sqrt{2}-2, \frac{2}{1+1 / \sqrt{\mu_{\text {max }}}}\right]$; note that by the assumption on $\min _{t} \mu_{t}$ and the fact that $\nu_{t} \geq 1 / 4$ the semi-Lipschitz constant is $\mathcal{O}\left(\sqrt{\mu_{t}}\right)$. Therefore the desired regret w.r.t. any $\omega \in\left[2 \sqrt{2}-2, \frac{2}{1+1 / \sqrt{\mu_{\text {max }}}}\right]$ follows, and extends to the rest of the interval because Lemma E.2.2 also implies all functions $U_{t}$ are increasing away from this interval. $\square$

## F Semi-stochastic proofs

## F. 1 Regularity of the criterion

Lemma F.1. $\left\|\breve{\mathbf{C}}_{\omega}^{k} \mathbf{b}\right\|_{2}$ is $\rho\left(\breve{\mathbf{C}}_{\omega}\right)^{k-1}\|\mathbf{b}\|_{2} k \sqrt{\kappa(\mathbf{A})}\left(\frac{1}{2-\omega_{\max }}+2 \rho\left(\mathbf{D A}^{-1}\right)\right)$-Lipschitz w.r.t. $\omega \in \Omega$.
Proof. Taking the derivative, we have that

$$
\begin{align*}
\left|\partial_{\omega}\left\|\breve{\mathbf{C}}_{\omega}^{k} \mathbf{b}\right\|_{2}\right| & =\frac{\left|\partial_{\omega}\left[\left(\breve{\mathbf{C}}_{\omega}^{k} \mathbf{b}\right)^{T} \breve{\mathbf{C}}_{\omega}^{k} \mathbf{b}\right]\right|}{\left\|\breve{\mathbf{C}}_{\omega}^{k} \mathbf{b}\right\|_{2}} \\
& =\frac{\left|\left(\breve{\mathbf{C}}_{\omega}^{k} \mathbf{b}\right)^{T} \sum_{i=1}^{k}\left[\breve{\mathbf{C}}_{\omega}^{i-1}\left(\partial_{\omega} \breve{\mathbf{C}}_{\omega}\right) \breve{\mathbf{C}}_{\omega}^{k-i} \mathbf{b}\right]\right|}{\left\|\breve{\mathbf{C}}_{\omega}^{k} \mathbf{b}\right\|_{2}} \\
& \leq\left\|\sum_{i=1}^{k} \breve{\mathbf{C}}_{\omega}^{i-1}\left(\partial_{\omega} \breve{\mathbf{C}}_{\omega}\right) \breve{\mathbf{C}}_{\omega}^{k-i} \mathbf{b}\right\|_{2} \\
& \leq \sum_{i=1}^{k}\left\|\breve{\mathbf{C}}_{\omega}^{i-1}\left(\partial_{\omega} \breve{\mathbf{C}}_{\omega}\right) \breve{\mathbf{C}}_{\omega}^{k-i} \mathbf{b}\right\|_{2} \\
& =\sum_{i=1}^{k}\left\|\mathbf{A}^{\frac{1}{2}} \mathbf{A}^{-\frac{1}{2}} \breve{\mathbf{C}}_{\omega}^{i-1} \mathbf{A}^{\frac{1}{2}} \mathbf{A}^{-\frac{1}{2}}\left(\partial_{\omega} \breve{\mathbf{C}}_{\omega}\right) \mathbf{A}^{\frac{1}{2}} \mathbf{A}^{-\frac{1}{2}} \breve{\mathbf{C}}_{\omega}^{k-i} \mathbf{A}^{\frac{1}{2}} \mathbf{A}^{-\frac{1}{2}} \mathbf{b}\right\|_{2} \\
& \leq\|\mathbf{b}\|_{2} \sqrt{\kappa(\mathbf{A})} \sum_{i=1}^{k}\left\|\left(\mathbf{A}^{-\frac{1}{2}} \breve{\mathbf{C}}_{\omega} \mathbf{A}^{\frac{1}{2}}\right)^{i-1}\right\|_{2}\left\|\mathbf{A}^{-\frac{1}{2}}\left(\partial_{\omega} \breve{\mathbf{C}}_{\omega}\right) \mathbf{A}^{\frac{1}{2}}\right\|_{2}\left\|\left(\mathbf{A}^{-\frac{1}{2}} \breve{\mathbf{C}}_{\omega} \mathbf{A}^{\frac{1}{2}}\right)^{k-i}\right\|_{2} \\
& =\rho\left(\breve{\mathbf{C}}_{\omega}\right)^{k-1}\|\mathbf{b}\|_{2} k\left\|\mathbf{A}^{-\frac{1}{2}}\left(\partial_{\omega} \breve{\mathbf{C}}_{\omega}\right) \mathbf{A}^{\frac{1}{2}}\right\|_{2} \sqrt{\kappa(\mathbf{A})} \tag{36}
\end{align*}
$$

where the first inequality is due to Cauchy-Schwartz, the second is the triangle inequality, and the third is due to the sub-multiplicativity of the norm. The last line follows by symmetry of $\mathbf{A}^{-\frac{1}{2}} \breve{\mathbf{C}}_{\omega} \mathbf{A}^{\frac{1}{2}}$, which implies that the spectral norm of any of power equals that power of its spectral radius, which by similarity is also the spectral radius of $\mathbf{C}_{\omega}$. Next we use a matrix calculus tool (Laue et al., 2018) to compute

$$
\begin{align*}
\partial_{\omega} \breve{\mathbf{C}}_{\omega}= & \left(\frac{1}{\omega}+\frac{2-\omega}{\omega^{2}}\right) \mathbf{A}\left(\mathbf{D} / \omega+\mathbf{L}^{T}\right)^{-1} \mathbf{D}(\mathbf{D} / \omega+\mathbf{L})^{-1} \\
& -\frac{2-\omega}{\omega^{3}} \mathbf{A}\left(\mathbf{D} / \omega+\mathbf{L}^{T}\right)^{-1} \mathbf{D}\left(\mathbf{D} / \omega+\mathbf{L}^{T}\right)^{-1} \mathbf{D}(\mathbf{D} / \omega+\mathbf{L})^{-1} \\
& +\frac{2-\omega}{\omega^{3}} \mathbf{A}\left(\mathbf{D} / \omega+\mathbf{L}^{T}\right)^{-1} \mathbf{D}(\mathbf{D} / \omega+\mathbf{L})^{-1} \mathbf{D}(\mathbf{D} / \omega+\mathbf{L})^{-1} \\
= & \left(\frac{1}{2-\omega}+\frac{1}{\omega}\right) \mathbf{A} \breve{\mathbf{W}}_{\omega}^{-1}-\frac{1}{\omega^{2}} \mathbf{A}\left(\mathbf{D} / \omega+\mathbf{L}^{T}\right)^{-1} \mathbf{D} \breve{\mathbf{W}}_{\omega}^{-1}-\frac{1}{\omega^{2}} \mathbf{A} \breve{\mathbf{W}}_{\omega}^{-1} \mathbf{D}(\mathbf{D} / \omega+\mathbf{L})^{-1} \tag{37}
\end{align*}
$$

so since $\left\|\mathbf{A}^{\frac{1}{2}} \breve{\mathbf{W}}_{\omega}^{-1} \mathbf{A}^{\frac{1}{2}}\right\|_{2}=\left\|\mathbf{I}_{n}-\mathbf{A}^{-\frac{1}{2}} \breve{\mathbf{C}}_{\omega} \mathbf{A}^{\frac{1}{2}}\right\|_{2} \leq 1+\rho\left(\mathbf{A}^{-\frac{1}{2}} \breve{\mathbf{C}}_{\omega} \mathbf{A}^{\frac{1}{2}}\right) \leq 2$ and

$$
\begin{align*}
\left\|\mathbf{A}^{\frac{1}{2}}\left(\mathbf{D} / \omega+\mathbf{L}^{T}\right)^{-1} \mathbf{D} \breve{\mathbf{W}}_{\omega}^{-1} \mathbf{A}^{\frac{1}{2}}\right\|_{2} & =\left\|\mathbf{A}^{\frac{1}{2}} \breve{\mathbf{W}}_{\omega}^{-1} \mathbf{D}(\mathbf{D} / \omega+\mathbf{L})^{-1} \mathbf{A}^{\frac{1}{2}}\right\|_{2} \\
& =\left\|\breve{\mathbf{W}}_{\omega}^{-1} \mathbf{D} \mathbf{W}_{\omega}^{-1} \mathbf{A}\right\|_{\mathbf{A}} \\
& \leq\left\|\breve{\mathbf{W}}_{\omega}^{-1} \mathbf{D}\right\|_{\mathbf{A}}\left\|\mathbf{I}_{n}-\mathbf{M}_{\omega}\right\|_{\mathbf{A}}  \tag{38}\\
& \leq 2\left\|\breve{\mathbf{W}}_{\omega}^{-1} \mathbf{A}\right\|_{\mathbf{A}}\left\|\mathbf{A}^{-\frac{1}{2}} \mathbf{D} \mathbf{A}^{-\frac{1}{2}}\right\|_{2} \\
& =2 \rho\left(\mathbf{D} \mathbf{A}^{-1}\right)\left\|\mathbf{I}_{n}-\breve{\mathbf{M}}_{\omega}\right\|_{\mathbf{A}} \leq 4 \rho\left(\mathbf{D} \mathbf{A}^{-1}\right)
\end{align*}
$$

we have by applying $\omega \in\left[1, \omega_{\text {max }}\right]$ that

$$
\begin{equation*}
\left\|\mathbf{A}^{-\frac{1}{2}}\left(\partial_{\omega} \breve{\mathbf{C}}_{\omega}\right) \mathbf{A}^{\frac{1}{2}}\right\|_{2} \leq \frac{2}{2-\omega}+\frac{2}{\omega}+\frac{8 \rho\left(\mathbf{D} \mathbf{A}^{-1}\right)}{\omega^{2}} \leq \frac{4}{2-\omega_{\max }}+8 \rho\left(\mathbf{D} \mathbf{A}^{-1}\right) \tag{39}
\end{equation*}
$$

```
Algorithm 8: Symmetric successive over-relaxation with an absolute convergence condition.
Input: $\mathbf{A} \in \mathbb{R}^{n \times n}, \mathbf{b} \in \mathbb{R}^{n}$, parameter $\omega \in(0,2)$, initial vector $\mathbf{x} \in \mathbb{R}^{n}$, tolerance $\varepsilon>0$
$\mathbf{D}+\mathbf{L}+\mathbf{L}^{T} \leftarrow \mathbf{A} / / \mathbf{D}$ is diagonal, $\mathbf{L}$ is strictly lower triangular
$\breve{\mathbf{W}}_{\omega} \leftarrow \frac{\omega}{2-\omega}(\mathbf{D} / \omega+\mathbf{L}) \mathbf{D}^{-1}\left(\mathbf{D} / \omega+\mathbf{L}^{T}\right) \quad / /$ compute third normal form
$\mathbf{r}_{0} \leftarrow \mathbf{b}-\mathbf{A} \mathbf{x} \quad / /$ compute initial residual
for $k=0, \ldots$ do
    if $\left\|\mathbf{r}_{k}\right\|_{2}>\varepsilon$ then
        return $k$ // return iteration count (for use in learning)
    $\mathbf{x}=\mathbf{x}+\breve{\mathbf{W}}_{\omega}^{-1} \mathbf{r}_{k} / /$ solve two triangular systems and update vector
    $\mathbf{r}_{k+1} \leftarrow \mathbf{b}-\mathbf{A} \mathbf{x} \quad / /$ compute the next residual
```

Output: $k$

Lemma F.2. $\left\|\breve{\mathbf{C}}_{\omega}^{k}(c) \mathbf{b}\right\|_{2}$ is $\frac{10}{\lambda_{\min }(\mathbf{A})+c_{\min }} \rho\left(\breve{\mathbf{C}}_{\omega}\right)^{k-1}(c)\|\mathbf{b}\|_{2} k \sqrt{\kappa(\mathbf{A})}$-Lipschitz w.r.t. all $c \geq c_{\min }> -\lambda_{\text {min }}(\mathbf{A}(c))$, where ( $c$ ) denotes matrices derived from $\mathbf{A}(c)=\mathbf{A}+c \mathbf{I}_{n}$.

Proof. We take the derivative as in the above proof of Lemma F.1:

$$
\begin{equation*}
\left|\partial_{c}\left\|\breve{\mathbf{C}}_{\omega}^{k}(c) \mathbf{b}\right\|_{2}\right|=\rho\left(\breve{\mathbf{C}}_{\omega}(c)\right)^{k-1}\|\mathbf{b}\|_{2} k\left\|\mathbf{A}^{-\frac{1}{2}}(c)\left(\partial_{c} \breve{\mathbf{C}}_{\omega}(c)\right) \mathbf{A}^{\frac{1}{2}}(c)\right\|_{2} \sqrt{\kappa(\mathbf{A}(c))} \tag{40}
\end{equation*}
$$

We then again apply the matrix calculus tool of Laue et al. (2018) to get

$$
\begin{align*}
\partial_{c} \breve{\mathbf{C}}_{\omega}(c)= & -\frac{2-\omega}{\omega}\left(\mathbf{D}(c) / \omega+\mathbf{L}^{T}\right)^{-1} \mathbf{D}(c)(\mathbf{D}(c) / \omega+\mathbf{L})^{-1} \\
& +\frac{2-\omega}{\omega^{2}} \mathbf{A}(c)\left(\mathbf{D}(c) / \omega+\mathbf{L}^{T}\right)^{-2} \mathbf{D}(c)(\mathbf{D}(c) / \omega+\mathbf{L})^{-1} \\
& -\frac{2-\omega}{\omega} \mathbf{A}(c)\left(\mathbf{D}(c) / \omega+\mathbf{L}^{T}\right)^{-1}(\mathbf{D}(c) / \omega+\mathbf{L})^{-1} \\
& +\frac{2-\omega}{\omega^{2}} \mathbf{A}(c)\left(\mathbf{D}(c) / \omega+\mathbf{L}^{T}\right)^{-1} \mathbf{D}(c)(\mathbf{D}(c) / \omega+\mathbf{L})^{-2}  \tag{41}\\
= & -\frac{2-\omega}{\omega}\left(\breve{\mathbf{W}}_{\omega}^{-1}(c)+\mathbf{A}(c) \mathbf{W}_{\omega}^{-T}(c) \mathbf{W}_{\omega}^{-1}(c)\right) \\
& +\frac{2-\omega}{\omega^{2}} \mathbf{A}(c)\left(\mathbf{W}_{\omega}^{-T}(c) \breve{\mathbf{W}}_{\omega}^{-1}(c)+\breve{\mathbf{W}}_{\omega}^{-1}(c) \mathbf{W}_{\omega}^{-1}(c)\right)
\end{align*}
$$

By symmetry of $\breve{\mathbf{W}}_{\omega}^{-1}(c)$ we have

$$
\begin{align*}
\left\|\mathbf{A}^{-\frac{1}{2}}(c) \breve{\mathbf{W}}_{\omega}^{-1}(c) \mathbf{A}^{\frac{1}{2}}(c)\right\|_{2}=\left\|\breve{\mathbf{W}}_{\omega}^{-1}(c)\right\|_{\mathbf{A}(c)} & \leq\left\|\breve{\mathbf{W}}_{\omega}^{-1}(c) \mathbf{A}(c)\right\|_{\mathbf{A}(c)}\left\|\mathbf{A}^{-1}(c)\right\|_{2} \\
& =\left\|\mathbf{I}_{n}-\breve{\mathbf{M}}_{\omega}(c)\right\|_{\mathbf{A}(c)} \rho\left(\mathbf{A}^{-1}(c)\right) \leq 2 \rho\left(\mathbf{A}^{-1}(c)\right) \tag{42}
\end{align*}
$$

Furthermore

$$
\begin{align*}
\left\|\mathbf{A}^{\frac{1}{2}}(c) \mathbf{W}_{\omega}^{-T}(c) \mathbf{W}_{\omega}^{-1}(c) \mathbf{A}^{\frac{1}{2}}(c)\right\|_{2} & =\left\|\mathbf{A}^{-\frac{1}{2}}(c)\left(\mathbf{I}_{n}-\mathbf{M}_{\omega}^{T}(c)\right)\left(\mathbf{I}_{n}-\mathbf{M}_{\omega}(c)\right) \mathbf{A}^{-\frac{1}{2}}(c)\right\|_{2} \\
& \leq\left\|\mathbf{A}^{-1}\right\|_{2}\left\|\mathbf{I}_{n}-\mathbf{M}_{\omega}(c)\right\|_{\mathbf{A}(c)}^{2} \leq 4 \rho\left(\mathbf{A}^{-1}(c)\right) \tag{43}
\end{align*}
$$

and

$$
\begin{equation*}
\left\|\mathbf{A}^{\frac{1}{2}}(c) \mathbf{W}_{\omega}^{-T}(c) \breve{\mathbf{W}}_{\omega}^{-1}(c) \mathbf{A}^{\frac{1}{2}}(c)\right\|_{2}=\left\|\mathbf{A}^{\frac{1}{2}}(c) \breve{\mathbf{W}}_{\omega}^{-1}(c) \mathbf{W}_{\omega}^{-1}(c) \mathbf{A}^{\frac{1}{2}}(c)\right\|_{2} \leq 4 \rho\left(\mathbf{A}^{-1}(c)\right) \tag{44}
\end{equation*}
$$

so by the lower bound of $\frac{1}{\lambda_{\text {min }}(\mathbf{A})+c_{\text {min }}}$ on $\rho\left(\mathbf{A}^{-1}(c)\right)$ we have the result.

## F. 2 Anti-concentration

Lemma F.3. Let $\mathbf{X} \in \mathbb{R}^{n \times n}$ be a nonzero matrix and $\mathbf{b}=m \mathbf{u}$ be a product of independent random variables $m \geq 0$ and $\mathbf{u} \in \mathbb{R}^{n}$ with $m^{2} \in[0, n]$ a $\chi^{2}$-squared random variable with $n$ degrees of freedom truncated to the interval $[0, n]$ and $\mathbf{u}$ distributed uniformly on the surface of the unit sphere.
Then for any interval $I=(\varepsilon, \varepsilon+\Delta] \subset \mathbb{R}$ for $\varepsilon, \Delta>0$ we have that $\operatorname{Pr}\left(\|\mathbf{X b}\|_{2} \in I\right) \leq \frac{2 \Delta}{\rho(\mathbf{X})} \sqrt{\frac{2}{\pi}}$.
Proof. Let $f$ be the p.d.f. of $\mathbf{b}$ and $g$ be the p.d.f. of $\mathbf{g} \sim \mathcal{N}\left(\mathbf{0}_{n}, \mathbf{I}_{n}\right)$. Then by the law of total probability and the fact that $\mathbf{b}$ follows the distribution of $\mathbf{g}$ conditioned on $\|\mathbf{b}\|_{2}^{2} \leq n$ we have that

$$
\begin{align*}
\operatorname{Pr}\left(\|\mathbf{X} \mathbf{b}\|_{2} \in I\right) & =\int_{\|\mathbf{x}\|_{2}^{2} \leq n} \operatorname{Pr}\left(\|\mathbf{X} \mathbf{b}\|_{2} \in I \mid \mathbf{b}=\mathbf{x}\right) d f(\mathbf{x}) \\
& =\frac{\int_{\|\mathbf{x}\|_{2}^{2} \leq n} \operatorname{Pr}\left(\|\mathbf{X} \mathbf{b}\|_{2} \in I \mid \mathbf{b}=\mathbf{x}\right) d g(\mathbf{x})}{\int_{\|\mathbf{x}\|_{2}^{2}>n} d g(\mathbf{x})}  \tag{45}\\
& \leq 2 \int_{\|\mathbf{x}\|_{2}^{2} \leq n} \operatorname{Pr}\left(\|\mathbf{X} \mathbf{g}\|_{2} \in I \mid \mathbf{g}=\mathbf{x}\right) d g(\mathbf{x}) \\
& \leq 2 \int_{\mathbb{R}^{n}} \operatorname{Pr}\left(\|\mathbf{X} \mathbf{g}\|_{2} \in I \mid \mathbf{g}=\mathbf{x}\right) d g(\mathbf{x})=2 \operatorname{Pr}\left(\|\mathbf{X} \mathbf{g}\|_{2} \in I\right)
\end{align*}
$$

where the second inequality uses the fact that a $\chi^{2}$ random variable with $n$ degrees of freedom has more than half of its mass below $n$. Defining the orthogonal diagonalization $\mathbf{Q}^{T} \Lambda \mathbf{Q}=\mathbf{X}^{T} \mathbf{X}$ and noting that $\mathbf{Q g} \sim \mathcal{N}\left(\mathbf{0}_{n}, \mathbf{I}_{n}\right)$, we then have that

$$
\begin{equation*}
\|\mathbf{X g}\|_{2}^{2}=(\mathbf{Q g})^{T} \Lambda \mathbf{Q g}=\sum_{i=1}^{n} \Lambda_{[i, i]} \chi_{i}^{2} \tag{46}
\end{equation*}
$$

for i.i.d. $\chi_{1}, \ldots, \chi_{n} \sim \mathcal{N}(0,1)$. Let $h, h_{1}$, and $h_{-1}$ be the densities of $\sum_{i=1}^{n} \Lambda_{[i, i]} \chi_{i}^{2}, \Lambda_{[1,1]} \chi_{1}^{2}$, and $\sum_{i=2}^{n} \Lambda_{[i, i]} \chi_{i}^{2}$, respectively, and let $u(a)$ be the uniform measure on the interval ( $a, a+2 \varepsilon \Delta+\Delta^{2}$ ]. Then since the density of the sum of independent random variables is their convolution, we can apply Young's inequality to obtain

$$
\begin{align*}
\operatorname{Pr}\left(\|\mathbf{X g}\|_{2} \in I\right) & =\operatorname{Pr}\left(\|\mathbf{X g}\|_{2}^{2} \in\left(\varepsilon^{2},(\varepsilon+\Delta)^{2}\right]\right) \\
& \leq \max _{a \geq \varepsilon^{2}} \int_{a}^{a+2 \varepsilon \Delta+\Delta^{2}} h(x) d x \\
& =\max _{a \geq \varepsilon^{2}} \int_{-\infty}^{\infty} u(x-a) h(x) d x \\
& =\|u * h\|_{L^{\infty}([\varepsilon, \infty))} \\
& =\left\|u * h_{1} * h_{-1}\right\|_{L^{\infty}([\varepsilon, \infty))} \\
& \leq\left\|u * h_{1}\right\|_{L^{\infty}([\varepsilon, \infty))}\left\|h_{-1}\right\|_{L^{1}([\varepsilon, \infty))}  \tag{47}\\
& \leq \max _{a \geq \varepsilon^{2}} \int_{a}^{a+2 \varepsilon \Delta+\Delta^{2}} h_{1}(x) d x \\
& =\max _{a \geq \varepsilon} \int_{a}^{a+2 \varepsilon \Delta+\Delta^{2}} \frac{e^{-\frac{x}{2 \Lambda_{[1,1]}}}}{\sqrt{2 \pi \Lambda_{[i, i]} x}} d x \\
& \leq \max _{a \geq \varepsilon^{2}} \sqrt{\frac{2\left(a+2 \varepsilon \Delta+\Delta^{2}\right)}{\pi \Lambda_{[i, i]}}}-\sqrt{\frac{2 a}{\pi \Lambda_{[i, i]}}}=\Delta \sqrt{\frac{2}{\pi \Lambda_{[i, i]}}}
\end{align*}
$$

Substituting into the first equation and using $\Lambda_{[i, i]}=\|\mathbf{X}\|_{2}^{2} \geq \rho(\mathbf{X})^{2}$ yields the result.

## F. 3 Lipschitz expectation

Lemma F.4. Suppose $\mathbf{b}=m \mathbf{u}$, where $m$ and $\mathbf{u}$ are independent random variables with $\mathbf{u}$ distributed uniformly on the surface of the unit sphere and $m^{2} \in[0, n]$ a $\chi^{2}$-squared random variable with $n$ degrees of freedom truncated to the interval $[0, n]$. Define $K$ as in Corollary D.2, $\beta=\min _{x} \rho\left(\mathbf{I}_{n}-\right. \mathbf{D}_{x}^{-1} \mathbf{A}_{x}$ ), and $\operatorname{SSOR}(x)=\min _{\left\|\breve{\mathbf{C}}_{x}^{k} \mathbf{b}\right\|_{2} \leq \varepsilon} k$ to be the number of iterations to convergence when the defect reduction matrix depends on some scalar $x \in \mathcal{X}$ for some bounded interval $\mathcal{X} \subset \mathbb{R}$. If $\left\|\breve{\mathbf{C}}_{x}^{k} \mathbf{b}\right\|_{2}$ is $L \rho\left(\breve{\mathbf{C}}_{x}\right)^{k-1}$-Lipschitz a.s. w.r.t. any $x \in \mathcal{X}$ then $\mathbb{E} S S O R$ is $\frac{32 K^{3} L \sqrt{2 / \pi}}{\beta^{4}}$-Lipschitz w.r.t. $x$.

Proof. First, note that by Hackbusch (2016, Theorem 6.26)

$$
\begin{equation*}
\rho\left(\breve{\mathbf{C}}_{x}\right)=\rho\left(\breve{\mathbf{M}}_{x}\right)=\left\|\breve{\mathbf{M}}_{x}^{2}\right\|_{\mathbf{A}_{x}} \geq \rho\left(\mathbf{M}_{x}\right)^{2} \geq\left(\frac{\beta}{1+\sqrt{1-\beta^{2}}}\right)^{4} \geq \frac{\beta^{4}}{16} \tag{48}
\end{equation*}
$$

Now consider any $x_{1}, x_{2} \in \mathcal{X}$ s.t. $\left|x_{1}-x_{2}\right| \leq \frac{\varepsilon \beta^{4} \sqrt{\pi / 2}}{2 K^{3} L}$, assume w.l.o.g. that $x_{1}<x_{2}$, and pick $x^{\prime} \in\left[x_{1}, x_{2}\right]$ with maximal $\rho\left(\breve{\mathbf{C}}_{x}\right)$. Then setting $\rho_{x^{\prime}}=\rho\left(\breve{\mathbf{C}}_{x^{\prime}}\right)$ we have that $\left\|\breve{\mathbf{C}}_{x_{i}}^{k} \mathbf{b}\right\|_{2}$ is $L \rho_{x^{\prime}}^{k-1}$ Lipschitz for both $i=1,2$ and all $k \in[K]$. Therefore starting with Jensen's inequality we have that

$$
\begin{align*}
&\left|\mathbb{E S S O R}\left(x_{i}\right)-\mathbb{E S S O R}\left(x^{\prime}\right)\right| \\
& \leq \mathbb{E}\left|\operatorname{SSOR}\left(x_{i}\right)-\operatorname{SSOR}\left(x^{\prime}\right)\right| \\
&= \sum_{k=1}^{K} \sum_{l=1}^{K}|k-l| \operatorname{Pr}\left(\operatorname{SSOR}\left(x_{i}\right)=k \cap \operatorname{SSOR}\left(x^{\prime}\right)=l\right) \\
& \leq K \sum_{k=1}^{K}\left(\sum_{l<k} \operatorname{Pr}\left(\left\|\breve{\mathbf{C}}_{x_{i}}^{l} \mathbf{b}\right\|_{2}>\varepsilon \cap\left\|\breve{\mathbf{C}}_{x^{\prime}}^{l} \mathbf{b}\right\|_{2} \leq \varepsilon\right)+\sum_{l>k} \operatorname{Pr}\left(\left\|\breve{\mathbf{C}}_{x_{i}}^{k} \mathbf{b}\right\|_{2} \leq \varepsilon \cap\left\|\breve{\mathbf{C}}_{x^{\prime}}^{k} \mathbf{b}\right\|_{2}>\varepsilon\right)\right) \\
& \leq K \sum_{k=1}^{K} \sum_{l<k} \operatorname{Pr}\left(\left\|\breve{\mathbf{C}}_{x^{\prime}}^{l} \mathbf{b}\right\|_{2} \in\left(\varepsilon-L \rho_{x^{\prime}}^{l-1}\left|x_{i}-x^{\prime}\right|, \varepsilon\right]\right) \\
&+K \sum_{k=1}^{K} \sum_{l>k} \operatorname{Pr}\left(\left\|\breve{\mathbf{C}}_{x^{\prime}}^{k} \mathbf{b}\right\|_{2} \in\left(\varepsilon, \varepsilon+L \rho_{x^{\prime}}^{k-1}\left|x_{i}-x^{\prime}\right|\right]\right) \\
& \leq K \sum_{k=1}^{K}\left(\sum_{l<k} \frac{2 L \rho_{x^{\prime}}^{l-1} \sqrt{2 / \pi}}{\rho\left(\breve{\mathbf{C}}_{x^{\prime}}^{l}\right)}\left|x_{i}-x^{\prime}\right|+\sum_{k=1}^{K} \sum_{l>k} \frac{2 L \rho_{x^{\prime}}^{k-1} \sqrt{2 / \pi}}{\rho\left(\breve{\mathbf{C}}_{x^{\prime}}^{k}\right)}\left|x_{i}-x^{\prime}\right|\right) \\
& \leq \frac{2 K^{3} L \sqrt{2 / \pi}}{\rho_{x^{\prime}}}\left|x_{i}-x^{\prime}\right| \leq \frac{32 K^{3} L \sqrt{2 / \pi}}{\beta^{4}} \tag{49}
\end{align*}
$$

where the second inequality follows by the definition of SSOR, the third by Lipschitzness, and the fourth by the anti-concentration result of Lemma F.3. Since this holds for any nearby pairs $x_{1}<x_{2}$, taking the summation over the interval $\mathcal{X}$ completes the proof.

Corollary F.1. Under the assumptions of Lemma E.1, the function $\mathbb{E}_{\mathbf{b}} \operatorname{SSOR}(\mathbf{A}, \mathbf{b}, \omega)$ is $\frac{32 K^{4} \sqrt{2 n \kappa(\mathbf{A}) / \pi}}{\beta^{4}}\left(\frac{1}{2-\omega_{\text {max }}}+2 \rho\left(\mathbf{D A}^{-1}\right)\right)$-Lipschitz w.r.t. $\omega \in\left[1, \omega_{\max }\right] \subset(0,2)$.

Proof. Apply Lemmas F. 1 and E.1, noting that $\|\mathbf{b}\|_{2} \leq \sqrt{n}$ by definition.
Corollary F.2. Under the assumptions of Lemma E.1, the function $\mathbb{E}_{\mathbf{b}} \operatorname{SSOR}(\mathbf{A}(c), \mathbf{b}, \omega)$ is $\max _{c} \frac{320 K^{4} \sqrt{2 n \kappa(\mathbf{A}(c)) / \pi}}{\beta^{4}\left(\lambda_{\text {min }}(\mathbf{A})+c_{\text {min }}\right)}$-Lipschitz w.r.t. $c \geq c_{\text {min }}>-\lambda_{\text {min }}$.

Proof. Apply Lemma F. 2 and E.1, noting that $\|\mathbf{b}\|_{2} \leq \sqrt{n}$ by definition.

![](https://cdn.mathpix.com/cropped/2025_11_24_6122a0dad929fc569471g-32.jpg?height=345&width=1359&top_left_y=271&top_left_x=368)
Figure 5: Average across forty trials of the time needed to solve 5 K diagonally shifted systems with $\mathbf{A}_{t}=\mathbf{A}+\frac{12 c-3}{20} \mathbf{I}_{n}$ for $c \sim \operatorname{Beta}\left(\frac{1}{2}, \frac{3}{2}\right)$ (center) and $c \sim \operatorname{Beta}(2,6)$ (otherwise).

## G Experimental details

All numerical results were generated in MATLAB on a laptop and can be re-generated by running the scripts available at https://github.com/mkhodak/learning-to-relax. Note that, since we do not have access to problem parameters, we experimented with a few approaches to setting them automatically or heuristically on the simplest (low variance) setting below and then used the same settings for the rest of the experiments (high variance and heat equation). Furthermore, because the default step-size/learning rate settings in both algorithms are rather pessimistic, we use more aggressive time-varying approaches in practice. For Tsallis-INF we set $\eta_{t}=2 / \sqrt{t}$, which is what is used in the anytime variant (Zimmert \& Seldin, 2021). As for ChebCB, we use an increasing schedule $\eta_{t}=\mathcal{O}(t)$; note that Simchi-Levi \& Xu (2021) also use an increasing learning rate schedule for setting inverse gap-weighted probabilities.

## G. 1 Basic experiments

For the experiments in Figure 2 (center-left), we sample $T=5 \mathrm{~K}$ scalars $c_{t} \sim \operatorname{Beta}(2,6)$ and run Tsallis-INF, Tsallis-INF-CB ChebCB, the instance optimal policy $\omega^{*}(c)$, and five values of $\omega$-evenly spaced on $[1,1.8]$-on all instances $\mathbf{A}_{t}=\mathbf{A}+\frac{12 c_{t}-3}{20} \mathbf{I}_{n}$, in random order. Note that Figure 5 (left) contains results of the same setup, except with $c_{t} \sim \operatorname{Beta}\left(\frac{1}{2}, \frac{3}{2}\right)$, the higher-variance setting from Figure 1. The center and right figures contain results for the sub-optimal fixed $\omega$ parameters, compared to Tsallis-INF. For both experiments the matrix $\mathbf{A}$ is again the $100 \times 100$ Laplacian of a square-shaped domain generated in MATLAB, and the targets $\mathbf{b}$ are re-sampled at each instance from the Gaussian truncated radially to have norm $\leq n$. The reported results are averaged of forty trials.

## G. 2 Accelerating a 2D heat equation solver

We then consider applying our methods to the task of numerical simulation of the 2D heat equation

$$
\begin{equation*}
\partial_{t} u(t, \mathbf{x})=\kappa(t) \Delta_{\mathbf{x}} u(t, \mathbf{x})+f(t, \mathbf{x}) \tag{50}
\end{equation*}
$$

over the domain $\mathbf{x} \in[0,1]^{2}$ and $t \in[0,5]$. We use a five-point finite difference discretization with size denoted $n_{\mathbf{x}}=1 / \Delta_{\mathbf{x}}$, so that when an implicit time-stepping method such as Crank-Nicolson is applied with timestep $\Delta_{t}$ the numerical simulation requires sequentially solving a sequence of linear systems ( $\mathbf{A}_{t}, \mathbf{b}_{t}$ ) with $\mathbf{A}_{t}=\mathbf{I}_{\left(n_{\mathbf{x}}-1\right)^{2}}-\kappa\left((t+1 / 2) \Delta_{t}\right) \mathbf{A}$ for a fixed matrix $\mathbf{A}$ (that depends on $\Delta_{t}$ and $\Delta_{\mathrm{x}}$ ) corresponding to the discrete Laplacian of the system (LeVeque, 2007, Equation 12.29). Each $\mathbf{A}_{t}$ is positive definite, and moreover note that mathematically the setting is equivalent to an instantiation of the diagonal offset setting introduced in Section 2.4, since the linear system is equivalent to $c_{t} \mathbf{I}_{\left(n_{\mathbf{x}}-1\right)^{2}}-\mathbf{A}=c_{t} \mathbf{b}_{t}$ for $c_{t}=1 / \kappa\left((t+1 / 2) \Delta_{t}\right)$. However, for simplicity we will simply pass $\kappa\left(\left(t+1 / 2 \Delta_{t}\right)\right)$ as contexts to CB methods.
To complete the problem specification, define the bump function $b_{\mathbf{c}, r}(\mathbf{x})$ centered at $\mathbf{c} \in \mathbb{R}^{2}$ with radius $r>0$ to be $\exp \left(-\frac{1}{1-\|\mathbf{x}-\mathbf{c}\|_{2}^{2} / r^{2}}\right)$ if $\|\mathbf{x}-\mathbf{c}\|_{2}<r$ and 0 otherwise. We set the initial condition $u(0, \mathbf{x})=\mathbf{b}_{\left(\frac{1}{2} \frac{1}{2}\right), \frac{1}{4}}(\mathbf{x})$, forcing function $f(t, \mathbf{x})=32 \mathbf{b}_{\left(\frac{1}{2}+\cos (16 \pi t) / 4, \frac{1}{2}+\cos (16 \pi t) / 4\right), 1 / 8}(\mathbf{x})$, and diffusion coefficient $\kappa(t)=\max \{0.01 \sin (2 \pi t)),-10 \sin (2 \pi t)\}$. The forcing function-effectively a bump circling around the center of the domain-is chosen to ensure that the linear system solutions

![](https://cdn.mathpix.com/cropped/2025_11_24_6122a0dad929fc569471g-33.jpg?height=366&width=1324&top_left_y=271&top_left_x=398)
Figure 6: Diffusion coefficient as a function of time (left) and normalized total wallclock time required to run 5 K steps of the numerical simulation (center and right). The numbers within the middle plot corresponding to the average number of seconds required to run a step of the simulation using vanilla CG. The right-hand plot shows $95 \%$ confidence intervals across the three trials for Tsallis-INF and ChebCB at the three higher-dimensional evaluations.

are not too close to each other or to zero, and the diffusion coefficient function-plotted in Figure 6 (left)-is chosen to make the instance-optimal $\omega$ behave roughly periodically (c.f. Figure 2 (right)).
We set $\Delta_{t}=10^{-3}$, thus making $T=5000$, and evaluate our approach across five spatial discretizations: $n_{\mathbf{x}}=25,50,100,200,400$. The resulting linear systems have size $n=\left(n_{\mathbf{x}}-1\right)^{2}$. At each timestep, we solve each linear system using CG to relative precision $\varepsilon=10^{-8}$. The baselines we consider are vanilla (unpreconditioned) CG and SSOR-CG with $\omega=1$ or $\omega=1.5$; as comparators we also evaluate performance when using the best fixed $\omega$ in hindsight at each round, and when using the instance-optimal $\omega$ at each round. Recall that we showed that Tsallis-INF has sublinear regret w.r.t. the surrogate cost of the best fixed $\omega$ of SSOR-CG (Theorem 2.3), and that ChebCB has sublinear regret w.r.t. the instance-optimal $\omega$ for SOR in the semi-stochastic setting of Section 3. Since both methods are randomized, we take the average of three runs.

In Figure 2 (center-right) we show that both methods substantially outperform all three baselines, except at $n_{\mathbf{x}}=25$ and $n_{\mathbf{x}}=50$, when $\omega=1.5$ almost recovers the best fixed parameter in hindsight; furthermore, ChebCB does better then the best fixed $\omega$ in hindsight in most cases. In Figure 6 (right) we also show that-at high-enough dimensions-this reduction in the number of iterations leads to an overall improvement in the runtime of the simulation. Several other pertinent notes include:

1. At lower dimensions the learning-based approaches have slower overall runtime because of overhead associated with learning; ChebCB in particular solves a small constrained linear regression at each step. However, this overhead does not scale with matrix dimension, and we expect data-driven approaches to have the greatest impact in higher dimensions.
2. Vanilla (unpreconditioned) CG is faster than SSOR-preconditioned CG with $\omega=1$ despite having more iterations because each iteration is more costly.
3. To get a comparative sense of the scale of the improvement, we can consider the results in Li et al. (2023, Table 1), who learn a (deep-learning-based) preconditioner for CG to simulate the 2D heat equation. In the precision $10^{-8}$ case their solver takes 2.3 seconds, while Gauss-Seidel (i.e. SSOR with $\omega=1$ ) takes 2.995 seconds, a roughly $1.3 x$ improvement. In our most closely comparable setting, Tsallis-INF and ChebCB are roughly 2.4 x and 3.5 x faster than Gauss-Seidel, respectively (and have other advantages such as simplicity and being deployable in an online fashion without pretraining). We caveat this comparison by noting that Li et al. (2023) consider a statistical, not online, learning setup, and their matrix structure may be significantly different-it results from a finite element method rather than finite differences. The only way to achieve a direct comparisons is via access to code; as of this writing it is not public.

Lastly, we give additional details for the plot in Figure 2 (right), which shows the actions taken by the various algorithms for a simulation at $n_{\mathbf{x}}=100$. For clarity all lines are smoothed using a moving average with a window of 25 , and for Tsallis-INF and ChebCB we also shade ± one standard deviation computed over this window. The plot shows that Tsallis-INF converges to an action close to the best fixed $\omega$ in hindsight, and that ChebCB fairly quickly follows the instance-optimal path, with the standard deviation of both decreasing over time.

