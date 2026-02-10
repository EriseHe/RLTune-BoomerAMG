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
        self.losses = np.zeros(T)
        
        # Debug tracking
        self.k_history = []
        self.fallback_count = 0  # Count uniform fallbacks
    
    def predict(self):
        """
        Samples action to be taken.
        
        Returns
        -------
        out : float
            Selected strong_threshold value from grid
        """
        eta = 2.0 / np.sqrt(self.t)

        # Newton's method to find x such that sum(probs) = 1, where
        #
        #   probs_i = 4 * (eta * (k_i/scale - x))^(-2).
        #
        # The MATLAB reference uses x = -1 as a fixed start. For AMG-style losses
        # (raw WU can be huge and heavy-tailed), k/scale can become O(t) and that
        # start causes Newton to take enormous steps and overshoot, producing
        # invalid intermediate values in NumPy. Here we use:
        # - a safe initial guess near min(k/scale)
        # - a hard constraint x < min(k/scale) to keep (k/scale - x) positive
        k_over_scale = self.k / self.scale
        min_k_over_scale = float(np.min(k_over_scale))

        # Keep x strictly below the minimum to preserve the derivation used
        # by the Newton update (the gradient expression assumes diff > 0).
        eps = max(1e-12, 1e-12 * abs(min_k_over_scale))
        x_upper = min_k_over_scale - eps

        # Initial guess: choose a delta so that if all diffs were equal, the
        # weights would approximately sum to 1. This makes Newton stable even
        # when k/scale is large (common at big T).
        delta0 = (2.0 * np.sqrt(self.d)) / eta
        x = x_upper - float(delta0)

        probs = None
        for _ in range(20):
            diff = k_over_scale - x
            if np.any(diff <= 0):
                x = x_upper
                diff = k_over_scale - x

            probs = 4.0 * (eta * diff) ** (-2.0)
            sum_probs = float(np.sum(probs))
            grad = float(eta * np.sum(probs ** 1.5))
            if not np.isfinite(sum_probs) or not np.isfinite(grad) or grad <= 0.0:
                probs = None
                break

            x_new = x - (sum_probs - 1.0) / grad
            # Enforce the domain constraint required by the update.
            x = min(float(x_new), x_upper)

        if probs is None:
            self.fallback_count += 1
            self.index = int(np.random.choice(self.d))
            self.prob = 1.0 / self.d
            out = float(self.grid[self.index])
            if self.t <= len(self.actions):
                self.actions[self.t - 1] = self.index
            return out

        # Normalize for NumPy sampling (MATLAB's randsample accepts weights).
        sum_probs = float(np.sum(probs))
        if not np.isfinite(sum_probs) or sum_probs <= 0.0:
            self.fallback_count += 1
            self.index = int(np.random.choice(self.d))
            self.prob = 1.0 / self.d
        else:
            p = probs / sum_probs
            if not np.all(np.isfinite(p)):
                self.fallback_count += 1
                self.index = int(np.random.choice(self.d))
                self.prob = 1.0 / self.d
            else:
                self.index = int(np.random.choice(self.d, p=p))
                # Store the actual sampling probability for importance weighting.
                self.prob = float(p[self.index])

        # Debug: store for inspection
        self._last_probs = probs / (sum_probs if sum_probs > 0 else 1.0)
        self._last_x = float(x)

        out = self.grid[self.index]
        
        if self.t <= len(self.actions):
            self.actions[self.t - 1] = self.index
        
        return out
    
    def update(self, loss):
        """
        Updates action distribution using the incurred cost.

        `loss` is a scalar "cost" derived from the BoomerAMG solve, typically
        built from Work Units (WU = iterations * cum_nnz_AP). 
        
        Parameters
        ----------
        loss : float
            Scalar loss (expects loss >= 1 in the same convention as the
            original MATLAB code, which updates with (loss - 1)).
        """
        if self.t <= len(self.losses):
            self.losses[self.t - 1] = loss
            self.scale = np.mean(self.losses[:self.t]) - 1.0
        
        self.k[self.index] = self.k[self.index] + (loss - 1.0) / self.prob
        
        # Debug: record k values periodically
        if self.t % 500 == 0 or self.t <= 10:
            self.k_history.append((
                self.t, 
                self.scale, 
                self.k.copy(), 
                self._last_probs.copy() if hasattr(self, '_last_probs') else None, 
                self._last_x if hasattr(self, '_last_x') else None
            ))
        
        self.t = self.t + 1
    
    def get_work_units(self):
        """
        Returns the accumulated losses so far.
        
        Returns
        -------
        losses : ndarray
            Array of losses for each round played
        """
        return self.losses[:self.t - 1]
