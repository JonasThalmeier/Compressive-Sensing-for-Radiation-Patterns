import numpy as np
from typing import Tuple, Optional

class SBL_CoFEM:
    def __init__(self, t, Phi, num_probes=10, max_iter=1000, threshold=1e-6, beta=1.0, precondition=False):
        """
        Initialize SBL with CoFEM algorithm
        
        Parameters:
            t: observed signal (N x 1)
            Phi: measurement matrix (N x M)
            num_probes: number of Rademacher probes for variance estimation
            max_iter: maximum number of iterations
            threshold: convergence threshold
            beta: noise precision (1/sigma^2)
            precondition :  Whether or not to use the diagonal preconditioner of
            (Lin et al., 2022) designed for matrices satisfying the
            restricted isometry property (RIP).
        """
        self.t = t
        self.Phi = Phi
        self.N, self.M = Phi.shape
        self.max_iter = max_iter
        self.threshold = threshold
        self.num_probes = num_probes
        self.precondition = precondition
        
        # Initialize hyperparameters
        self.alpha = np.ones(self.M)  # Hyperparameters for precision of w
        self.beta = beta  # Noise precision (1/sigma^2)
        
    def _samp_probes(self, size: Tuple[int, ...]) -> np.ndarray:
        """Sample Rademacher probes {±1}."""
        return 2 * (np.random.rand(*size) > 0.5) - 1
    
    def _conj_grad(self, A, b, M_inv=None, max_iters=1000, tol=1e-10):
        """Parallel Conjugate Gradient solver following Algorithm 2."""
        x = np.zeros_like(b)
        r = b.copy()
        if M_inv is None:
            z = r
        else:
            z = M_inv(r)
        # Initialize residual
        p = r.copy()
        
        # Compute initial rho
        rho = np.sum(r * z, axis=1)
        
        for u in range(max_iters):
            # Apply matrix A
            Ap = A(p)
            
            # Compute step size
            pi = np.sum(p * Ap, axis=1)
            gamma = rho / pi
            
            # Update solution and residual
            x = x + gamma[:, np.newaxis] * p
            r = r - gamma[:, np.newaxis] * Ap
            
            # Check convergence
            delta = np.linalg.norm(r) / np.linalg.norm(b)
            if delta <= tol:
                return x, u + 1
            
            # Prepare for next iteration
            rho_old = rho
            if M_inv is None:
                z = r
            else:
                z = M_inv(r)
            rho = np.sum(r * z, axis=1)
            eta = rho / rho_old
            p = z + eta[:, np.newaxis] * p
            
        return x, None

    def estimate_posterior(self):
        """Simplified E-step following Algorithm 1."""
        # Define matrix A = βΦ^T Φ + diag{α}
        def A(x):
            # x has shape (batch_size, M)
            # Need to handle batched matrix multiplication
            batch_size = x.shape[0]
            Phi_x = np.stack([self.Phi @ x[i] for i in range(batch_size)])
            Phi_T_Phi_x = np.stack([self.Phi.T @ Phi_x[i] for i in range(batch_size)])
            return self.beta * Phi_T_Phi_x + (self.alpha * x)
        
        # Sample Rademacher probes (K x M)
        K = self.num_probes
        probes = self._samp_probes((K, self.M))  # Shape: (K, M)
        
        # Define matrix B = [p1|p2|...|pK|βΦ^T y]
        beta_phi_T_y = (self.beta * self.Phi.T @ self.t).reshape(1, -1)
        B = np.vstack([probes, beta_phi_T_y])  # Shape: (K+1, M)
        
        # Set preconditioner if required
        if self.precondition:
            M_inv = lambda x: 1 / (self.beta + self.alpha) * x
        else:
            M_inv = lambda x: x

        # Solve linear system
        X, _ = self._conj_grad(A, B, M_inv=M_inv)
        
        # Extract results
        mu = X[-1]  # Last row is μ
        probe_solutions = X[:-1]  # First K rows are x1,...,xK
        
        # Compute variances sj = 1/K Σ(pk,j * xk,j)
        sigma_diag = np.mean(probes * probe_solutions, axis=0)
        
        return mu, sigma_diag
    
    def maximize(self, mu: np.ndarray, sigma_diag: np.ndarray):
        """M-step: Update alpha following Algorithm 1."""
        self.alpha = 1.0 / (np.square(mu) + sigma_diag)
        
    def fit(self, track_iterations=np.arange(1, 1001, 100)):
        """Run CoFEM algorithm and track weight evolution."""
        old_alpha = np.zeros_like(self.alpha)
        tracked_weights = np.zeros((len(track_iterations), self.M))
        
        for iter in range(self.max_iter):
            # E-step
            mu, sigma_diag = self.estimate_posterior()
            
            # Save weights if current iteration is in track_iterations
            if iter + 1 in track_iterations:
                idx = np.where(track_iterations == iter + 1)[0][0]
                tracked_weights[idx] = mu.copy()
            
            # M-step
            self.maximize(mu, sigma_diag)
            
            # Check convergence
            change = np.max(np.abs(old_alpha - self.alpha))
            if change < self.threshold:
                print(f"Converged after {iter+1} iterations")
                break
                
            old_alpha = self.alpha.copy()
        
        return mu, tracked_weights
