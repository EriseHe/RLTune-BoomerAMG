"""
Implements the Tsallis-INF bandit algorithm for BoomerAMG.

This variant uses Work Units (WU = iterations * cum_nnz_AP) as the loss
function instead of iteration count, which properly accounts for the
varying cost of different AMG hierarchies.

NOTE: uses a time-varying setting of eta = 2 / sqrt(t)
"""

import numpy as np


class TsallisINF_AMG:
    """
    Tsallis-INF bandit algorithm for BoomerAMG tuning.
    
    Uses work units (WU = iterations * cum_nnz_AP) as the loss function,
    which accounts for both the number of V-cycles and the hierarchy complexity.
    
    Parameters
    ----------
    grid : array_like
        Action space (grid of strong_threshold values, typically 0.1 to 0.9)
    T : int
        Number of rounds (can be set to zero if not known)
    """
    
    def __init__(self, grid, T):
        self.grid = np.asarray(grid)
        self.d = len(grid)
        self.t = 1
        self.k = np.zeros(self.d)
        self.index = None
        self.prob = None
        self.scale = 1.0
        
        self.actions = np.zeros(T, dtype=int)
        self.losses = np.zeros(T)  # Stores work units (WU)
    
    def predict(self):
        """
        Samples action to be taken.
        
        Returns
        -------
        out : float
            Selected strong_threshold value from grid
        """
        eta = 2.0 / np.sqrt(self.t)
        x = -1.0
        
        # Newton's method to find x (20 iterations)
        for i in range(20):
            probs = 4.0 * (eta * (self.k / self.scale - x)) ** (-2.0)
            x = x - (np.sum(probs) - 1.0) / (eta * np.sum(probs ** 1.5))
        
        probs = np.maximum(probs, 0)  # Ensure non-negative
        probs_sum = np.sum(probs)

        # `np.random.choice` expects a normalized probability vector, but to
        # match the MATLAB reference implementation we store the *unnormalized*
        # weight `probs[index]` as `self.prob` for the importance-weighted update.
        try:
            if probs_sum > 0 and np.all(np.isfinite(probs)):
                probs_normalized = probs / probs_sum
                self.index = np.random.choice(self.d, p=probs_normalized)
                self.prob = probs[self.index]
            else:
                self.index = np.random.choice(self.d)
                self.prob = 1.0 / self.d
        except Exception:
            self.index = np.random.choice(self.d)
            self.prob = 1.0 / self.d

        out = self.grid[self.index]
        
        if self.t <= len(self.actions):
            self.actions[self.t - 1] = self.index
        
        return out
    
    def update(self, iterations, cum_nnz_AP):
        """
        Updates action distribution using the incurred cost.
        
        For BoomerAMG, the loss is Work Units (WU) = iterations * cum_nnz_AP,
        which accounts for both the number of V-cycles and the cost per cycle.
        
        Parameters
        ----------
        iterations : int
            Number of V-cycles (iterations) to converge
        cum_nnz_AP : float
            Cumulative nnz ratio: (sum nnz(A_l) + sum nnz(P_l)) / nnz(A_0)
            This represents the memory/work complexity of the AMG hierarchy
        """
        # Compute work units as the loss
        loss = iterations * cum_nnz_AP
        
        if self.t <= len(self.losses):
            self.losses[self.t - 1] = loss
            self.scale = np.mean(self.losses[:self.t]) - 1.0
        
        # Avoid division by zero / invalid importance weights
        if self.scale == 0:
            self.scale = 1.0
        if not np.isfinite(self.prob) or self.prob == 0:
            self.prob = 1.0 / self.d

        self.k[self.index] = self.k[self.index] + (loss - 1.0) / self.prob
        self.t = self.t + 1
    
    def get_work_units(self):
        """
        Returns the accumulated work units (losses) so far.
        
        Returns
        -------
        losses : ndarray
            Array of work units for each round played
        """
        return self.losses[:self.t - 1]
