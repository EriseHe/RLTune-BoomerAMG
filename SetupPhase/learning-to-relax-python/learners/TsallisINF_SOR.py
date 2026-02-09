"""
Implements the Tsallis-INF bandit algorithm.
NOTE: uses a time-varying setting of eta = 2 / sqrt(t)
Direct translation of TsallisINF.m
"""

import numpy as np


class TsallisINF:
    """
    Tsallis-INF bandit algorithm.
    
    Parameters
    ----------
    grid : array_like
        Action space (grid of omega values)
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
    
    def predict(self):
        """
        Samples action to be taken.
        
        Returns
        -------
        out : float
            Selected omega value from grid
        """
        eta = 2.0 / np.sqrt(self.t)
        x = -1.0
        
        # Newton's method to find x (20 iterations)
        for i in range(20):
            probs = 4.0 * (eta * (self.k / self.scale - x)) ** (-2.0)
            x = x - (np.sum(probs) - 1.0) / (eta * np.sum(probs ** 1.5))

        try:
            p = probs / np.sum(probs)
            self.index = np.random.choice(self.d, p=p)
        except Exception:
            self.index = np.random.choice(self.d)
            self.prob = 1.0 / self.d

        self.prob = float(probs[self.index])
        out = self.grid[self.index]

        self.actions[self.t - 1] = self.index
        
        return out
    
    def update(self, loss):
        """
        Updates action distribution using the incurred cost.
        
        Parameters
        ----------
        loss : float
            Cost incurred (number of iterations)
        """
        self.losses[self.t - 1] = loss
        self.scale = np.mean(self.losses[:self.t]) - 1.0

        self.k[self.index] = self.k[self.index] + (loss - 1.0) / self.prob
        self.t = self.t + 1
