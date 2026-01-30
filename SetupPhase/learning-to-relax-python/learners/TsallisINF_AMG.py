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
        
        # Newton's method to find x such that sum(probs) = 1
        # CRITICAL: x must stay below min(k/scale) to keep all (k/scale - x) positive
        # Otherwise the algorithm degenerates (probability mass goes to arbitrary arms)
        k_over_scale = self.k / self.scale
        
        # Compute upper bound: x must stay below min(k/scale)
        min_k_over_scale = np.min(k_over_scale)
        # For numerical stability, use small relative margin
        eps = max(1e-8, 1e-8 * abs(min_k_over_scale))
        x_upper_bound = min_k_over_scale - eps
        
        # IMPORTANT: Start Newton well below the upper bound
        # Starting too far (e.g., x=-1 when k/scale~600) causes Newton to diverge
        # A good starting point: x_upper_bound - some_offset where offset scales with range
        k_range = np.max(k_over_scale) - min_k_over_scale
        x_start_offset = max(1.0, k_range + 1.0)  # At least 1 below, more if k values vary
        x = x_upper_bound - x_start_offset
        
        for i in range(20):
            diff = k_over_scale - x
            # Ensure all differences are positive (required for valid probabilities)
            if np.any(diff <= 0):
                x = np.min(k_over_scale) - 1.0
                diff = k_over_scale - x
            
            probs = 4.0 * (eta * diff) ** (-2.0)
            grad = eta * np.sum(probs ** 1.5)
            
            if grad > 0:
                x_new = x - (np.sum(probs) - 1.0) / grad
                # Don't let x exceed the upper bound
                x = min(x_new, x_upper_bound)
            else:
                break  # Gradient is invalid, stop iterating
        
        # Recompute with final x
        diff = k_over_scale - x
        
        # (B) Assert that diff is valid - should never trigger if Newton worked
        if np.any(diff <= 0):
            raise RuntimeError(
                f"TsallisINF: invalid x (diff<=0); root solver failed. "
                f"x={x}, min(k/scale)={np.min(k_over_scale)}, diff={diff}"
            )
        
        probs = 4.0 * (eta * diff) ** (-2.0)
        
        # Tiny clamp as numerical guard (should rarely trigger)
        probs = np.maximum(probs, 1e-18)
        probs_normalized = probs / np.sum(probs)

        # Debug: store for inspection
        self._last_probs = probs_normalized.copy()
        self._last_x = x

        # Sample according to the normalized distribution and store the
        # *actual sampling probability* for correct importance weighting
        try:
            if np.all(np.isfinite(probs_normalized)):
                self.index = np.random.choice(self.d, p=probs_normalized)
                self.prob = probs_normalized[self.index]
            else:
                # Fallback to uniform
                self.fallback_count += 1
                self.index = np.random.choice(self.d)
                self.prob = 1.0 / self.d
        except Exception:
            # Fallback to uniform
            self.fallback_count += 1
            self.index = np.random.choice(self.d)
            self.prob = 1.0 / self.d

        out = self.grid[self.index]
        
        if self.t <= len(self.actions):
            self.actions[self.t - 1] = self.index
        
        return out
    
    def update(self, loss):
        """
        Updates action distribution using the incurred cost.
        
        For BoomerAMG, the caller passes a **log-transformed** loss:
        
            WU = iterations * cum_nnz_AP  (raw work units)
            loss = 1 + log(1 + WU / a)    where a ≈ 40 is a "knee" parameter
        
        This log transform:
        - Keeps easy-instance differences visible (WU 30-50 → loss 1.6-1.8)
        - Keeps hard instances informative without saturation (WU 1000 → loss ~4.3)
        - Works well with the repo's `scale = mean(losses) - 1` heuristic
        
        NOTE: This loss is NOT bounded in [1, 2] as the original paper assumes,
        but works well in practice. For strict bounded loss, use:
            loss = 1 + log(1 + min(WU, K) / a) / log(1 + K / a)
        which maps to [1, 2] with K as timeout-scale.
        
        Parameters
        ----------
        loss : float
            Log-transformed loss (typically in range [1.5, 5] for BoomerAMG)
        """
        if self.t <= len(self.losses):
            self.losses[self.t - 1] = loss
            self.scale = np.mean(self.losses[:self.t]) - 1.0
        
        # Avoid division by zero / invalid importance weights
        if self.scale <= 0:
            self.scale = 1.0
        if not np.isfinite(self.prob) or self.prob <= 0:
            self.prob = 1.0 / self.d

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
