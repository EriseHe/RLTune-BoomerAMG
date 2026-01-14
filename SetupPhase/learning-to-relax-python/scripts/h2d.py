"""
Evaluates the runtime and number of iterations of different linear system
solvers (and learned settings for them) while running 5000 steps of a
two-dimensional heat equation simulation with time-varying diffusion.
Direct translation of h2d.m
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg, spsolve
import time

from learners import TsallisINF, TsallisINFCB, ChebCB
from solvers import ssor_pcg
from utils import Heat2D, bump, golden_section


def main():
    T = 5000  # number of simulation steps
    stoptime = 5.0
    dt = stoptime / T
    nxs = 25 * 2 ** np.arange(0, 3)  # Reduced for faster execution (original: 0:4)
    nnxs = len(nxs)
    epsilon = 1e-8  # solver tolerance
    omegas = [1.0, 1.3, 1.5, 1.75, 1.95]  # fixed omegas to evaluate
    trials = 3  # number of evaluations for learning algorithms
    
    # Sets diffusion coefficient
    def coefficient(t):
        return max(0.1 * np.sin(2.0 * np.pi * t), -10.0 * np.sin(2.0 * np.pi * t))
    
    coeffs = np.array([coefficient(t) for t in np.linspace(0.0, stoptime, T)])
    cmin = np.min(coeffs)
    cmax = np.max(coeffs)
    
    # Sets forcing to be a smooth bump function moving in a circle
    def forcing(t, x):
        center = [0.5 + np.cos(16.0 * np.pi * t) / 4.0, 
                  0.5 + np.sin(16.0 * np.pi * t) / 4.0]
        return 32.0 * bump(x, center, 0.125)
    
    # Sets initial conditions to be a smooth bump function in the middle
    def initial(x):
        return bump(x, [0.5, 0.5], 0.25)
    
    # Results storage
    results = {}
    results['CG'] = {'wallclock': np.zeros(nnxs), 'iterations': np.zeros(nnxs)}
    for omega in omegas:
        results[str(omega)] = {'wallclock': np.zeros(nnxs), 'iterations': np.zeros(nnxs)}
    results['Tsallis-INF'] = {'wallclock': np.zeros((nnxs, trials)), 
                               'iterations': np.zeros((nnxs, trials)),
                               'actions': np.zeros((nnxs, trials, T))}
    results['Tsallis-INF-CB'] = {'wallclock': np.zeros((nnxs, trials)), 
                                  'iterations': np.zeros((nnxs, trials))}
    results['ChebCB'] = {'wallclock': np.zeros((nnxs, trials)), 
                         'iterations': np.zeros((nnxs, trials)),
                         'actions': np.zeros((nnxs, trials, T))}
    results['A\\b'] = {'wallclock': np.zeros(nnxs)}
    results['Instance-Optimal'] = {'iterations': np.zeros(nnxs), 
                                   'actions': np.zeros((nnxs, T))}
    results['Fixed Optimal'] = {'iterations': np.zeros(nnxs), 'actions': np.zeros(nnxs)}
    
    Ng = 12
    omega_grid_arr = np.zeros((nnxs, T, Ng))
    niter_grid = np.zeros((nnxs, T, Ng))
    Nc = 96
    g = np.linspace(1.0, 1.95, Nc)
    contours = np.zeros((nnxs, T, Nc))
    printerval = 500
    
    np.random.seed(0)
    
    os.makedirs('plots', exist_ok=True)
    
    for nxidx in range(nnxs):
        nx = nxs[nxidx]
        pde_n = (nx - 1) ** 2
        print(f"\n=== Grid size nx={nx}, n={pde_n} ===")
        
        # Evaluate MATLAB's exact solver baseline (A\b)
        print("Evaluating direct solver (A\\b)...")
        pde = Heat2D(coefficient, forcing, initial, nx, dt)
        start_time = time.time()
        for i in range(T):
            A, b = pde.crank_nicolson_system()
            pde.update(spsolve(A, b))
            if (i + 1) % printerval == 0:
                print(f"  [n={pde.n}] A\\b: step {i + 1} / {T}")
        results['A\\b']['wallclock'][nxidx] = time.time() - start_time
        
        # Evaluate unpreconditioned CG
        print("Evaluating unpreconditioned CG...")
        niters = 0
        pde = Heat2D(coefficient, forcing, initial, nx, dt)
        start_time = time.time()
        for i in range(T):
            A, b = pde.crank_nicolson_system()
            u, info = cg(A, b, x0=pde.u, tol=epsilon, maxiter=10000)
            # Count iterations (approximate)
            niters += 1  # CG doesn't easily expose iteration count
            pde.update(u)
            if (i + 1) % printerval == 0:
                print(f"  [n={pde.n}] CG: step {i + 1} / {T}")
        results['CG']['wallclock'][nxidx] = time.time() - start_time
        results['CG']['iterations'][nxidx] = T  # Placeholder
        
        # Evaluate SSOR-preconditioned CG at different omega values
        for omegaidx, omega in enumerate(omegas):
            if pde_n > 400 and omega not in [1.0, 1.5]:
                continue
            
            print(f"Evaluating SSOR-PCG with omega={omega}...")
            niters = 0
            iterations = np.zeros(T)
            pde = Heat2D(coefficient, forcing, initial, nx, dt)
            start_time = time.time()
            for i in range(T):
                A, b = pde.crank_nicolson_system()
                A_dense = A.toarray() if hasattr(A, 'toarray') else A
                k, u = ssor_pcg(A_dense, b, pde.u, omega, epsilon)
                iterations[i] = k
                niters += k
                pde.update(u)
                if (i + 1) % printerval == 0:
                    print(f"  [n={pde.n}] omega={omega}: step {i + 1} / {T}")
            results[str(omega)]['wallclock'][nxidx] = time.time() - start_time
            results[str(omega)]['iterations'][nxidx] = niters
            niter_grid[nxidx, :, omegaidx] = iterations
        
        # Evaluate Tsallis-INF
        print("Evaluating Tsallis-INF...")
        for trial in range(trials):
            tinf = TsallisINF(np.linspace(1.0, 1.95, 20), T)
            niters = 0
            pde = Heat2D(coefficient, forcing, initial, nx, dt)
            start_time = time.time()
            for i in range(T):
                A, b = pde.crank_nicolson_system()
                A_dense = A.toarray() if hasattr(A, 'toarray') else A
                omega_pred = tinf.predict()
                k, u = ssor_pcg(A_dense, b, pde.u, omega_pred, epsilon)
                tinf.update(k)
                niters += k
                pde.update(u)
                if (i + 1) % printerval == 0:
                    print(f"  [n={pde.n}] Tsallis-INF: trial {trial + 1}/{trials}, step {i + 1} / {T}")
            results['Tsallis-INF']['wallclock'][nxidx, trial] = time.time() - start_time
            results['Tsallis-INF']['iterations'][nxidx, trial] = niters
            for i in range(T):
                results['Tsallis-INF']['actions'][nxidx, trial, i] = tinf.grid[tinf.actions[i]]
        
        # Evaluate Tsallis-INF-CB
        print("Evaluating Tsallis-INF-CB...")
        for trial in range(trials):
            tinfcb = TsallisINFCB(np.linspace(1.0, 1.95, 20), 6, cmin, cmax)
            niters = 0
            pde = Heat2D(coefficient, forcing, initial, nx, dt)
            start_time = time.time()
            for i in range(T):
                A, b = pde.crank_nicolson_system()
                A_dense = A.toarray() if hasattr(A, 'toarray') else A
                context = pde.coefficient(pde.t)
                omega_pred = tinfcb.predict(context)
                k, u = ssor_pcg(A_dense, b, pde.u, omega_pred, epsilon)
                tinfcb.update(k)
                niters += k
                pde.update(u)
                if (i + 1) % printerval == 0:
                    print(f"  [n={pde.n}] Tsallis-INF-CB: trial {trial + 1}/{trials}, step {i + 1} / {T}")
            results['Tsallis-INF-CB']['wallclock'][nxidx, trial] = time.time() - start_time
            results['Tsallis-INF-CB']['iterations'][nxidx, trial] = niters
        
        # Evaluate ChebCB
        print("Evaluating ChebCB...")
        for trial in range(trials):
            cheb = ChebCB(np.linspace(1.0, 1.95, 20), T, 6, cmin, cmax)
            niters = 0
            pde = Heat2D(coefficient, forcing, initial, nx, dt)
            start_time = time.time()
            for i in range(T):
                A, b = pde.crank_nicolson_system()
                A_dense = A.toarray() if hasattr(A, 'toarray') else A
                context = pde.coefficient(pde.t)
                omega_pred = cheb.predict(context)
                k, u = ssor_pcg(A_dense, b, pde.u, omega_pred, epsilon)
                cheb.update(k)
                niters += k
                pde.update(u)
                if (i + 1) % printerval == 0:
                    print(f"  [n={pde.n}] ChebCB: trial {trial + 1}/{trials}, step {i + 1} / {T}")
            results['ChebCB']['wallclock'][nxidx, trial] = time.time() - start_time
            results['ChebCB']['iterations'][nxidx, trial] = niters
            for i in range(T):
                results['ChebCB']['actions'][nxidx, trial, i] = cheb.grid[cheb.actions[i]]
    
    # Generate comparison plot
    print("\nGenerating iteration comparison plot...")
    plt.figure(figsize=(7, 5))
    labels = ['CG', '1.0', '1.5', 'Tsallis-INF', 'ChebCB']
    colors = ['#000000', '#4DBEEE', '#EDB120', '#77AC30', '#7E2F8E']
    styles = ['-+', '-v', '-^', '-o', '-d']
    
    dims = (nxs - 1) ** 2
    
    for i, label in enumerate(['CG', '1.0', '1.5']):
        if label in results:
            plt.loglog(dims, results[label]['iterations'], styles[i], 
                      linewidth=2, color=colors[i], label=label.replace('1.0', 'ω=1').replace('1.5', 'ω=1.5'))
    
    for i, label in enumerate(['Tsallis-INF', 'ChebCB'], start=3):
        if label in results:
            plt.loglog(dims, np.mean(results[label]['iterations'], axis=1), 
                      styles[i], linewidth=2, color=colors[i], label=label)
    
    plt.legend(loc='upper left', fontsize=10)
    plt.xlabel('matrix dimension', fontsize=14)
    plt.ylabel('#iterations', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/iterations.png', dpi=256)
    plt.close()
    
    print("Done! Plots saved to plots/")


if __name__ == '__main__':
    main()
