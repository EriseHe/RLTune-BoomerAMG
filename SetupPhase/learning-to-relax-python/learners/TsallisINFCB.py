"""
Implements a contextual bandit algorithm that discretizes the context 
space and runs Tsallis-INF independently in each bin.
Direct translation of TsallisINFCB.m
"""

import numpy as np
from .TsallisINF import TsallisINF


class TsallisINFCB:
    """
    Contextual bandit using discretized context space with Tsallis-INF.
    
    Parameters
    ----------
    grid : array_like
        Action space (grid of omega values)
    m : int
        Number of bins for discretizing the context space
    a : float
        Lower bound on the context space
    b : float
        Upper bound on the context space
    """
    
    def __init__(self, grid, m, a, b):
        # Create m instances of TsallisINF
        self.handles = [TsallisINF(grid, 0) for _ in range(m)]
        
        # Discretization: centers of m bins in [a, b]
        # MATLAB: a + (b-a) * linspace(.5/m, 1-.5/m, m)
        self.disc = a + (b - a) * np.linspace(0.5 / m, 1.0 - 0.5 / m, m)
        
        self.action = None
    
    def predict(self, context):
        """
        Predicts which action should be taken given a context.
        
        Parameters
        ----------
        context : float
            Current context value (e.g., diagonal offset)
        
        Returns
        -------
        out : float
            Selected omega value
        """
        # Find closest discretization bin
        # MATLAB: [~, i] = min(abs(obj.disc - context))
        i = np.argmin(np.abs(self.disc - context))
        out = self.handles[i].predict()
        self.action = i
        return out
    
    def update(self, loss):
        """
        Updates the algorithm using the incurred cost.
        
        Parameters
        ----------
        loss : float
            Cost incurred (number of iterations)
        """
        self.handles[self.action].update(loss)
