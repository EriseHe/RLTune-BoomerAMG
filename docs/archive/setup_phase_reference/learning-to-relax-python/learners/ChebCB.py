"""
Implements the ChebCB contextual bandit algorithm for continuous
one-dimensional contexts.
NOTE: uses a time-varying setting of eta that increases linearly in t
Direct translation of ChebCB.m
"""

import numpy as np
from scipy.optimize import lsq_linear


class ChebCB:
    """
    ChebCB contextual bandit using Chebyshev polynomial regression.
    
    Parameters
    ----------
    grid : array_like
        Action space (grid of omega values)
    T : int
        Number of rounds (can be set to zero if not known)
    m : int
        Largest power of the Chebyshev approximation
    a : float
        Lower bound on the context space
    b : float
        Upper bound on the context space
    """
    
    def __init__(self, grid, T, m, a, b):
        self.grid = np.asarray(grid)
        self.T = T
        self.d = len(grid)
        self.m = m
        self.t = 1
        self.actions = np.zeros(T, dtype=int)
        self.theta = np.zeros((self.d, m))
        self.features = np.zeros((T, m))
        self.losses = np.zeros(T)
        self.eta = 0.0
        self.a = a
        self.bma = b - a  # b minus a
        self.scale = 0.0
        
        self.predicted = np.zeros(T)
        self.contexts = np.zeros(T)
        
        # Precompute Chebyshev polynomial coefficients
        # T_0(x) = 1
        # T_1(x) = x
        # T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
        # Store coefficients for each polynomial degree
        if m > 1:
            self.chebyshev_coefs = self._compute_chebyshev_coefficients(m)
        else:
            self.chebyshev_coefs = None
    
    def _compute_chebyshev_coefficients(self, m):
        """
        Compute coefficients for Chebyshev polynomials T_0 through T_{m-1}.
        
        Returns coefficients in standard form: sum(c[i] * x^i for i in range(len(c)))
        """
        # Coefficients for each polynomial
        # coefs[j] contains coefficients for T_j(x), from x^0 to x^j
        coefs = []
        
        # T_0(x) = 1
        coefs.append(np.array([1.0]))
        
        if m >= 2:
            # T_1(x) = x
            coefs.append(np.array([0.0, 1.0]))
        
        # Recurrence: T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
        for j in range(2, m):
            prev = coefs[j - 1]
            prev2 = coefs[j - 2]
            
            # 2*x*T_{j-1}(x): shift coefficients and multiply by 2
            new_len = len(prev) + 1
            term1 = np.zeros(new_len)
            term1[1:] = 2.0 * prev
            
            # -T_{j-2}(x): pad with zeros to match length
            term2 = np.zeros(new_len)
            term2[:len(prev2)] = -prev2
            
            coefs.append(term1 + term2)
        
        return coefs
    
    def chebT(self, x):
        """
        Evaluate Chebyshev polynomials T_0(x) through T_{m-1}(x).
        
        Parameters
        ----------
        x : float
            Point at which to evaluate (should be in [-1, 1])
        
        Returns
        -------
        out : ndarray
            Array of length m with [T_0(x), T_1(x), ..., T_{m-1}(x)]
        """
        out = np.ones(self.m)
        
        for j in range(1, self.m):
            if j == 1:
                out[j] = x
            else:
                # Use precomputed coefficients
                # polyval in MATLAB evaluates with coefficients in descending order
                # Our coefficients are stored in ascending order
                coef = self.chebyshev_coefs[j]
                out[j] = np.polyval(coef[::-1], x)
        
        return out
    
    def show(self, action):
        """
        Plots the current regression approximation of the cost of an
        action over the entire context interval.
        
        Parameters
        ----------
        action : int
            Index of the action to plot
        """
        import matplotlib.pyplot as plt
        
        n = 100
        contexts = np.linspace(self.a, self.a + self.bma, n)
        features = np.zeros((n, self.m))
        for i in range(n):
            features[i, :] = self.chebT(2.0 / self.bma * (contexts[i] - self.a) - 1.0)
        
        plt.plot(contexts, 1.0 + self.scale * features @ self.theta[action, :])
        plt.show()
    
    def predict(self, context):
        """
        Predicts which action should be taken given a context.
        
        Parameters
        ----------
        context : float
            Current context value
        
        Returns
        -------
        out : float
            Selected omega value
        """
        # Compute features: Chebyshev polynomials on normalized context
        feature = self.chebT(2.0 / self.bma * (context - self.a) - 1.0)
        
        if self.t <= len(self.features):
            self.features[self.t - 1, :] = feature
        
        # Compute predicted values for each action
        yhat = self.theta @ feature
        
        # Find best action
        istar = np.argmin(yhat)
        ystar = yhat[istar]
        
        # Compute probabilities using SquareCB formula
        probs = np.zeros(self.d)
        for i in range(self.d):
            if i != istar:
                denom = self.d + self.eta * (yhat[i] - ystar)
                probs[i] = 1.0 / max(denom, 1e-10)  # Avoid division by zero
        
        probs[istar] = 1.0 - np.sum(probs[np.arange(self.d) != istar])
        probs = np.maximum(probs, 0)  # Ensure non-negative
        
        # Normalize probabilities
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs /= probs_sum
        else:
            probs = np.ones(self.d) / self.d
        
        # Sample action
        i = np.random.choice(self.d, p=probs)
        
        if self.t <= len(self.actions):
            self.actions[self.t - 1] = i
        
        out = self.grid[i]
        
        if self.t <= len(self.predicted):
            self.predicted[self.t - 1] = self.theta[i, :] @ feature
            self.contexts[self.t - 1] = context
        
        return out
    
    def update(self, loss):
        """
        Updates the algorithm using the incurred cost.
        
        Parameters
        ----------
        loss : float
            Cost incurred (number of iterations)
        """
        if self.t <= len(self.losses):
            self.losses[self.t - 1] = loss
        
        K = np.max(self.losses[:self.t])
        L = (np.max(self.losses[:self.t]) - np.min(self.losses[:self.t])) / self.bma
        if L == 0:
            L = 1.0
        
        N = 2.0 + 4.0 * self.bma * L / K * (1.0 + np.log(self.m))
        old_scale = self.scale
        self.scale = K * N
        
        update_all = (old_scale != self.scale)
        
        # Bounds on theta
        ub = np.zeros(self.m)
        ub[0] = 1.0 / N
        for j in range(1, self.m):
            ub[j] = 2.0 * self.bma * L / self.scale / j
        lb = -ub
        
        resnorm = 0.0
        
        for i in range(self.d):
            if update_all or i == self.actions[self.t - 1]:
                # Find indices where this action was taken
                idx = (self.actions[:self.t] == i)
                
                if np.sum(idx) > 0:
                    X = self.features[idx, :]
                    y = (self.losses[idx] - 1.0) / self.scale
                    
                    # Least squares with minimum norm
                    # MATLAB: lsqminnorm(X, y)
                    try:
                        self.theta[i, :], residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
                    except:
                        self.theta[i, :] = np.zeros(self.m)
                    
                    # Check if bounds are violated
                    if np.any(np.abs(self.theta[i, :]) > ub):
                        # Use constrained least squares
                        # MATLAB: lsqlin(X, y, [], [], [], [], lb, ub, theta0, options)
                        try:
                            result = lsq_linear(X, y, bounds=(lb, ub))
                            self.theta[i, :] = result.x
                            resnorm = result.cost
                        except:
                            pass
        
        # Update eta
        alpha = (np.pi + 2.0 / np.pi * np.log(2.0 * self.m + 1)) / (2.0 * self.scale * (self.m + 1)) * self.bma * L
        R = np.sum((self.predicted[:self.t] - self.losses[:self.t] / K / N) ** 2)
        
        denom = R - resnorm + 2.0 * alpha ** 2 * self.t
        if denom > 0:
            self.eta = 2.0 * self.t * np.sqrt(self.d * self.t / denom)
        else:
            self.eta = 0.0
        
        self.t = self.t + 1
