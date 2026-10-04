"""Reproduce the anchored minimax calculation and audit the experiment profile."""
from __future__ import annotations

# Import first to preserve the experiment's single-thread numerical environment.
from experiments.paper_final import run_05_policy as first
import argparse
from pathlib import Path
import numpy as np
import sympy as sp


def eta(a):
    """Exact stationary-point candidates; no sampling of the spectral interval."""
    roots = np.roots([5*a, -3*(a+1), 1]) if a else [1/3]
    points = [0., 1.] + [float(t.real) for t in roots if abs(t.imag) < 1e-14 and 0 < t.real < 1]
    return max(t*(1-t)**2*(1-a*t)**2 for t in points)


def verify_math():
    t = sp.symbols("t", positive=True)
    a = (3+sp.sqrt(5))/2
    peaks = [1-2/sp.sqrt(5), (1+1/sp.sqrt(5))/2]
    q = t*(1-t)**2*(1-a*t)**2
    expected = (10-2*sp.sqrt(5))/125
    for peak in peaks:
        assert sp.simplify(q.subs(t, peak)-expected) == 0
        assert sp.simplify(sp.diff(q,t).subs(t,peak)) == 0
    grid = [i/20 for i in range(20,61)]
    assert min(grid,key=eta) == 2.6
    assert min([1,1.5,2,2.5,3],key=eta) == 2.5
    rng = np.random.default_rng(92705400)
    worst = 0.; count = 0
    for n in (2,3,5,10):
        for _ in range(25):
            u = np.linalg.qr(rng.normal(size=(n,n)))[0]
            eigenvalues = rng.uniform(.03,1,n)
            T = (u*eigenvalues)@u.T
            v = np.linalg.qr(rng.normal(size=(n,n)))[0][:,:max(1,n//2)]
            Q = v@v.T; I = np.eye(n)
            K = np.linalg.norm((u/np.sqrt(eigenvalues))@u.T@Q,2)**2
            for high in (1.,2.5,2.6,float(a),3.):
                S = I-T; R = I-high*T; P = S@R
                radius = max(abs(np.linalg.eigvals(S@Q@S@R@Q@R)))
                compressed = np.linalg.norm(Q@P@Q,2)**2
                worst = max(worst,abs(radius-compressed)); count += 1
                assert abs(radius-compressed) < 1e-11
                assert compressed <= K*eta(high)+1e-12
    # Row-l1 domination is not restricted to matrices with negative off-diagonals.
    domination = []
    for n in (3,8,20):
        R = rng.normal(size=(n,n)); A = R.T@R + np.eye(n)
        d = np.abs(A).sum(axis=1)
        spectrum = np.linalg.eigvalsh(A/np.sqrt(d[:,None]*d[None,:]))
        assert spectrum.min()>0 and spectrum.max()<=1+1e-12
        domination.append({"n":n,"min_eigenvalue":spectrum.min(),"max_eigenvalue":spectrum.max()})
    return {"passed":True,"continuous_minimizer":float(a),"eta_continuous":float(expected),
        "fine_grid_minimizer":2.6,"coarse_grid_minimizer":2.5,
        "grid":[{"a":x,"eta":eta(x)} for x in grid],
        "symbolic_peaks_verified":True,"two_grid_checks":count,"max_identity_error":worst,
        "row_l1_spd_checks":domination,
        "interpretation":"Numerical checks supplement the written proof; they do not prove a multilevel runtime optimum."}


def verify_profile(parent):
    jobs = first.read(parent/"jobs_base_test.json")
    assert len(jobs)==600
    for j in jobs:
        m=j["mkw"]
        assert m["stencil"]==0 and all(m[k]==0 for k in ("a1","a2","a3"))
        assert all(m[k]>0 for k in ("k","c","a0"))
        assert all(m[k]==60 for k in ("nx","ny","nz"))
        assert not set(j["params"]) & {"relax_order","relax_type","num_sweeps","cycle_type"}
    checks=[]
    first.configure_smoother_profile(first.PROFILE)
    for seed in range(1,7):
        config=first.read(parent/f"checkpoints/seed_{seed}/experiment_config.json")
        assert config["solve"]["action_grid"]=={"start":1,"stop":3,"step":.05}
        j=next(j for j in jobs if j["seed"]==seed)
        with first.create_env(**j["mkw"]) as env:
            prep=env.prepare_rl(params=j["params"])
            assert env.cycle_type==1 and env.cycle_relax_types==(18,18,9)
            residuals=[]
            for w in (2.6,1.):
                residual,_=env.step_rl(relax_weight=w,sweeps_down=1,sweeps_up=1,tol=1e-6,max_cycles=50)
                residuals.append(residual)
            checks.append({"seed":seed,"cycle_type":env.cycle_type,
                "cycle_relax_types":env.cycle_relax_types,"initial_weight":prep.initial_relax_weight,"action_grid":config["solve"]["action_grid"],
                "two_cycle_residuals":residuals})
    return {"passed":True,"audited_jobs":len(jobs),"all_inputs_positive_diffusion_zero_advection":True,
        "profile":first.PROFILE,"native_checks":checks,
        "relax_order_evidence":"Native default 0 in par_amg.c; no parameter override in any job or schedule step.",
        "denominator_evidence":"par_amg_setup.c uses ComputeL1Norms option 1 with NULL CF marker; ams.c computes the full absolute row sum.",
        "scope":"Native profile plus source audit; no assertion that the multilevel coarse correction is an exact projection."}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result={"at":first.now(),"math":verify_math(),"implementation":verify_profile(args.parent)}
    first.dump(args.output,result)
    print("PASS: symbolic extrema, 41-grid minimizer, 500 two-grid identities/bounds, 600 inputs and six native profiles")
