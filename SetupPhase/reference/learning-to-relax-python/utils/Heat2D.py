"""
Convenience class for solving the heat equation with a time-varying 
diffusion coefficient using Crank-Nicolson on a uniform 2D grid.
Direct translation of Heat2D.m
"""

import numpy as np
from scipy.sparse import spdiags, eye as speye, csr_matrix


class Heat2D:
    """
    2D heat equation solver using Crank-Nicolson.
    
    Parameters
    ----------
    coefficient : callable
        Function of time returning diffusion coefficient
    forcing : callable
        Function of (time, spatial_coords) returning forcing term
    initial : callable
        Function of spatial_coords returning initial condition
    nx : int
        Number of grid points in each dimension
    dt : float
        Length of each time-step
    """
    
    def __init__(self, coefficient, forcing, initial, nx, dt):
        self.coefficient = coefficient
        self.forcing = forcing
        self.nx = nx
        self.n = (nx - 1) ** 2
        dx = 1.0 / nx
        
        # Build the Laplacian matrix
        # X is the 1D finite difference matrix
        o = np.ones(nx - 1)
        # spdiags in MATLAB: spdiags([o, -4*o, o], [-1, 0, 1], nx-1, nx-1)
        X = spdiags([o, -4*o, o], [-1, 0, 1], nx - 1, nx - 1).tocsr()
        
        # Build the 2D Laplacian by Kronecker-like construction
        # In MATLAB, this is done by explicit index manipulation
        # We'll replicate the exact logic
        Xi, Xj = X.nonzero()
        Xv = X.data
        
        n_entries = len(Xi)
        Li = np.zeros(n_entries * (nx - 1), dtype=int)
        Lj = np.zeros(n_entries * (nx - 1), dtype=int)
        Lv = np.zeros(n_entries * (nx - 1))
        
        for i in range(nx - 1):
            start = i * n_entries
            end = (i + 1) * n_entries
            Li[start:end] = Xi + i * (nx - 1)
            Lj[start:end] = Xj + i * (nx - 1)
            Lv[start:end] = Xv
        
        # Create sparse matrix from COO format
        L_base = csr_matrix((Lv, (Li, Lj)), shape=(self.n, self.n))
        
        # Add off-diagonal coupling between rows (the [o, o] at [-nx+1, nx-1])
        o_full = np.ones(self.n)
        coupling = spdiags([o_full, o_full], [-(nx-1), nx-1], self.n, self.n)
        
        self.L = (L_base + coupling) / (dx ** 2)
        self.I = speye(self.n, format='csr')
        
        # Build grid coordinates
        self.ijdx = np.zeros((self.n, 2))
        k = 0
        for i in range(1, nx):
            for j in range(1, nx):
                self.ijdx[k, :] = [i * dx, j * dx]
                k += 1
        
        self.u = initial(self.ijdx)
        self.t = 0.0
        self.dt = dt
    
    def crank_nicolson_system(self):
        """
        Returns the current linear system to advance the simulation.
        
        Returns
        -------
        A : sparse matrix
            System matrix
        b : ndarray
            Right-hand side vector
        """
        halfstep = self.t + 0.5 * self.dt
        CL = 0.5 * self.coefficient(halfstep) * self.dt * self.L
        A = self.I - CL
        b = (self.I + CL) @ self.u + self.dt * self.forcing(halfstep, self.ijdx)
        return A, b
    
    def update(self, u):
        """
        Updates the simulation using the provided solution vector.
        
        Parameters
        ----------
        u : ndarray
            New solution vector
        """
        self.u = u
        self.t = self.t + self.dt
    
    def show(self):
        """
        Plots the temperature at each grid coordinate.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        V = np.zeros((self.nx + 1, self.nx + 1))
        V[1:self.nx, 1:self.nx] = self.u.reshape(self.nx - 1, self.nx - 1).T
        
        x = np.arange(self.nx + 1) / self.nx
        y = np.arange(self.nx + 1) / self.nx
        X, Y = np.meshgrid(x, y)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, V)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        plt.show()
