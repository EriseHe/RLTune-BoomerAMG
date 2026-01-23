"""
Computes and plots the number of iterations required for SSOR-preconditioned 
CG to converge as well as the condition-number-based bound for comparison, 
all at different values of omega and for several different linear systems.
Direct translation of cg.m
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import eye as speye

from solvers import ssor_pcg, cgbound
from utils import truncated_normal, delsq, numgrid


def main():
    os.makedirs('plots', exist_ok=True)
    
    for region in ['L', 'S']:
        for s in [12, 32]:
            for offset in [0.0, 0.5]:
                print(f"Processing region={region}, s={s}, offset={offset}")
                
                A = delsq(numgrid(region, s))
                n = A.shape[0]
                A_offset = A + offset * speye(n, format="csr")
                b = truncated_normal(n)
                
                K = 100
                epsilon = 1e-8
                omegas = np.linspace(2.0 * np.sqrt(2.0) - 2.0, 1.9, K)
                costs = np.zeros(K)
                bounds = cgbound(A_offset, omegas, epsilon) - 1.0
                
                for i, omega in enumerate(omegas):
                    if (i + 1) % 20 == 0:
                        print(f"  {i + 1}/{K}")
                    x = np.zeros(n)
                    costs[i], _ = ssor_pcg(A_offset, b, x, omega, epsilon)
                
                tau = np.max((costs - 1.0) / bounds)
                
                plt.figure(figsize=(7, 5))
                plt.plot(omegas, costs, linewidth=2, label='actual cost')
                plt.plot(omegas, 1.0 + tau * bounds, linewidth=2, linestyle='--', 
                         label='upper bound')
                plt.legend(loc='upper center', fontsize=12)
                plt.title(f'{region}-shaped domain: size={n}, offset={offset}', fontsize=14)
                plt.xlabel('ω', fontsize=16)
                plt.ylabel('iterations', fontsize=14)
                plt.tight_layout()
                plt.savefig(f'plots/cgbound-{region}-{n}-{offset}.png', dpi=256)
                plt.close()
    
    print("Done! Plots saved to plots/")


if __name__ == '__main__':
    main()
