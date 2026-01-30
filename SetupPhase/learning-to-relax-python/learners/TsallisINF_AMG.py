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
        
        # Ensure numerical stability: clamp to small positive, then renormalize
        probs = np.maximum(probs, 1e-18)
        probs_normalized = probs / np.sum(probs)

        # Sample according to the normalized distribution and store the
        # *actual sampling probability* for correct importance weighting
        try:
            if np.all(np.isfinite(probs_normalized)):
                self.index = np.random.choice(self.d, p=probs_normalized)
                self.prob = probs_normalized[self.index]  # Store actual probability!
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
    
    def update(self, loss):
        """
        Updates action distribution using the incurred cost.
        
        For BoomerAMG, the caller should pass a **normalized** loss in [1, 2]:
            WU = iterations * cum_nnz_AP
            u = min(WU, K) / K   # in [0, 1], where K is a WU budget cap
            loss = 1.0 + u       # in [1, 2]
        
        This matches the original MATLAB implementation's assumptions where
        (loss - 1) is in [0, 1].
        
        Parameters
        ----------
        loss : float
            Normalized loss in [1, 2] (or at least bounded and close to 1)
        """
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
