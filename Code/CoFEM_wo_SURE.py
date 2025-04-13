import numpy as np
from typing import Tuple, Optional

class SBL_CoFEM:
    def __init__(self, t, Phi, num_probes=10, max_iter=1000, threshold=1e-6, beta=1.0, precondition=False):
        """
        Initialize SBL with CoFEM algorithm
        
        Parameters:
            t: observed signal (N x 1)
            Phi: measurement matrix (N x D)
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
        self.N, self.D = Phi.shape
        self.max_iter = max_iter
        self.threshold = threshold
        self.num_probes = num_probes
        self.precondition = precondition
        
        # Initialize hyperparameters
        self.alpha = np.ones(self.D)  # Hyperparameters for precision of w
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
        """def A(x):
            # x has shape (batch_size, D)
            batch_size = x.shape[0]
            Phi_x = np.stack([self.Phi @ x[i] for i in range(batch_size)])
            Phi_T_Phi_x = np.stack([self.Phi.T @ Phi_x[i] for i in range(batch_size)])
            return self.beta * Phi_T_Phi_x + (self.alpha[np.newaxis, :] * x)  # Correct broadcasting"""
        
        # Sample Rademacher probes (K x D)
        K = self.num_probes
        probes = self._samp_probes((self.D,K))  # Shape: (D,K)
        
        # Define matrix B = [p1|p2|...|pK|βΦ^T y]
        beta_phi_T_y = (self.beta * self.Phi.T @ self.t).reshape((self.D, 1))  # Shape: (D, 1)
        B = np.hstack([probes, beta_phi_T_y])  # Shape: (D, K+1)
        
        # Set preconditioner if required
        if self.precondition:
            # M_inv = lambda x: 1 / (self.beta + self.alpha) * x
            M_inv = lambda x: 1 / (1 + alpha / self.beta) * x
        else:
            M_inv = lambda x: x

        # Solve linear system
        # X, _ = self._conj_grad(A, B, M_inv=M_inv)
        
       # Construct matrix A explicitly: A = βΦ^T Φ + diag{α}
        PhiT_Phi = self.Phi.T @ self.Phi  # Shape: (D, D)
        # A = self.beta * PhiT_Phi + np.diag(self.alpha)  # Shape: (D, D)
        A = self.beta*PhiT_Phi + np.diag(self.alpha)  # Shape: (D, D)
    
        # Solve system AX = B^T
        # B has shape (K+1, D), so B^T has shape (D, K+1)
        # Result X will have shape (D, K+1)
        X = np.linalg.solve(A, B)
        
        # Extract results
        mu = X[:,-1]  # Last row is μ
        x = X[:,:-1]  # First K rows are x1,...,xK
        
        # Compute variances sj = 1/K Σ(pk,j * xk,j)
        # s = np.mean(probes * x, axis=0)
        s = (1 / self.beta) * np.clip(np.mean(probes * x, axis=1), 0, None)        
        return mu, s
    
    def maximize(self, mu: np.ndarray, s: np.ndarray):
        """M-step: Update alpha following Algorithm 1."""
        self.alpha = 1.0 / (np.square(mu) + s)
        
    def fit(self, track_iterations=np.arange(1, 1001, 100)):
        """Run CoFEM algorithm and track weight evolution."""
        old_alpha = np.ones_like(self.alpha)
        tracked_weights = np.zeros((len(track_iterations), self.D))
        
        for iter in range(self.max_iter):
            if iter%50 == 0:
                print(f"Iteration {iter+1}/{self.max_iter}")
                change = np.log(np.max(np.abs(old_alpha - self.alpha)))
                print(f"Current change: {change}")
            # E-step
            mu, s = self.estimate_posterior()
            
            # Save weights if current iteration is in track_iterations
            if iter + 1 in track_iterations:
                idx = np.where(track_iterations == iter + 1)[0][0]
                tracked_weights[idx] = mu.copy()
            
            # M-step
            self.maximize(mu, s)
            
            # Check convergence
            change = np.max(np.abs(old_alpha - self.alpha))
            if change < self.threshold:
                print(f"Converged after {iter+1} iterations")
                break
                
            old_alpha = self.alpha.copy()
        
        return mu, tracked_weights
