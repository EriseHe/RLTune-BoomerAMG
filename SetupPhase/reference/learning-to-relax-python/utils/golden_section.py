"""
Runs golden section search on a function and returns all of the intermediate
queries and evaluations; designed to heuristically use a coarse initial
grid of evaluations.
Direct translation of golden_section.m
"""

import numpy as np


def golden_section(eval_func, start_grid, start_evals, N):
    """
    Golden section search with heuristic initial grid.
    
    Parameters
    ----------
    eval_func : callable
        Function to minimize
    start_grid : array_like
        Initial grid of evaluation points
    start_evals : array_like
        Function values at initial grid points
    N : int
        Total number of function evaluations to perform
    
    Returns
    -------
    grid : ndarray
        All evaluation points
    evals : ndarray
        All function values
    """
    start_grid = np.asarray(start_grid).flatten()
    start_evals = np.asarray(start_evals).flatten()
    n = len(start_grid)
    
    grid = np.zeros(N)
    evals = np.zeros(N)
    grid[:n] = start_grid
    evals[:n] = start_evals
    
    best = np.min(start_evals)
    argmins = np.where(start_evals == best)[0]
    left = max(0, np.min(argmins) - 1)  # 0-indexed
    right = min(n - 1, np.max(argmins) + 1)  # 0-indexed
    oleft = start_grid[left]
    oright = start_grid[right]
    
    tau = (np.sqrt(5) - 1) / 2
    
    # Current index for new evaluations
    current_n = n
    
    if right - left == 2:
        ocenter = start_grid[left + 1]
        if oright - ocenter > ocenter - oleft:
            o1 = ocenter
            f1 = start_evals[left + 1]
            o2 = oleft + tau * (oright - oleft)
            grid[current_n] = o2
            f2 = eval_func(o2)
            evals[current_n] = f2
            current_n += 1
        elif oright - ocenter < ocenter - oleft:
            o1 = oleft + (1 - tau) * (oright - oleft)
            grid[current_n] = o1
            f1 = eval_func(o1)
            evals[current_n] = f1
            current_n += 1
            o2 = ocenter
            f2 = start_evals[left + 1]
        else:
            o1 = oleft + (1 - tau) * (oright - oleft)
            grid[current_n] = o1
            f1 = eval_func(o1)
            evals[current_n] = f1
            current_n += 1
            o2 = oleft + tau * (oright - oleft)
            grid[current_n] = o2
            f2 = eval_func(o2)
            evals[current_n] = f2
            current_n += 1
    elif right - left == 3:
        o1 = start_grid[left + 1]
        f1 = start_evals[left + 1]
        o2 = start_grid[right - 1]
        f2 = start_evals[right - 1]
    else:
        o1 = oleft + (1 - tau) * (oright - oleft)
        grid[current_n] = o1
        f1 = eval_func(o1)
        evals[current_n] = f1
        current_n += 1
        o2 = oleft + tau * (oright - oleft)
        grid[current_n] = o2
        f2 = eval_func(o2)
        evals[current_n] = f2
        current_n += 1
    
    # Main golden section loop
    for i in range(current_n, N):
        # Note: MATLAB uses 1-indexed i starting from n+1
        # The condition (f1 > f2) || (f1 == f2 && rem(i, 2)) uses 1-indexed i
        # We'll use (i+1) to match MATLAB's 1-indexed behavior
        if (f1 > f2) or (f1 == f2 and (i + 1) % 2 == 1):
            oleft = o1
            o1 = o2
            f1 = f2
            o2 = oleft + tau * (oright - oleft)
            grid[i] = o2
            f2 = eval_func(o2)
            evals[i] = f2
        else:
            oright = o2
            o2 = o1
            f2 = f1
            o1 = oleft + (1 - tau) * (oright - oleft)
            grid[i] = o1
            f1 = eval_func(o1)
            evals[i] = f1
    
    return grid, evals
