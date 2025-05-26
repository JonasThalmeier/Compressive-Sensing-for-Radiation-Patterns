import numpy as np
from typing import Tuple, Optional
_sentinel = object()
"""
Algorithm according to paper 'Sparse Bayesian Learning for Basis Selection' from Wimpf & Rao
"""
class EM_noSURE:
    def __init__(self, t, Phi, max_iter=1000, threshold=1e-6, beta_in=_sentinel):
        """
        Initialize SBL with EM algorithm
        
        Parameters:
            t: observed signal (N x 1)
            Phi: measurement matrix (N x D)
            max_iter: maximum number of iterations
            threshold: convergence threshold
        """
        self.t = t
        self.Phi = Phi
        self.N, self.D = Phi.shape
        self.max_iter = max_iter
        self.threshold = threshold
        self.beta_in = beta_in
        self.real = np.allclose(Phi, Phi.real) and np.allclose(t, t.real)
        
        # Initialize hyperparameters
        # self.alpha = np.ones(self.D)  # Hyperparameters for precision of w
        self.alpha = np.ones(self.D)
        if beta_in is _sentinel:
            self.beta = 1.0  # Noise precision (1/sigma^2)
        else:
            self.beta = beta_in

        
    def estimate_posterior(self):
        """E-step: Estimate posterior distribution of w"""
        # Use complex conjugate transpose
        Sigma = np.linalg.inv(self.beta * self.Phi.conj().T @ self.Phi + np.diag(self.alpha))
        mu = self.beta * Sigma @ self.Phi.conj().T @ self.t
        return mu, Sigma
    
    def maximize(self, mu, Sigma):
        """M-step: Update hyperparameters alpha and beta"""
        # Update alpha (variance of weights)
        self.alpha = 1/(np.square(np.abs(mu)) + np.diag(np.abs(Sigma)))

        # Update beta (noise precision)
        if self.beta_in is _sentinel:
            # Update alpha (precision of weights)
        
            error = self.t - self.Phi @ mu
            sum = np.sum(1 - self.alpha * np.diag(Sigma))
            error_term = np.sum(np.square(np.abs(error))) if not self.real else np.sum(np.square(error))
        
            # Add small constant to prevent division by zero
            eps = 1e-10
            self.beta = self.N / (error_term + sum/self.beta + eps)
            #self.beta = self.N/(np.sum(np.square(error))+sum/self.beta)
        
        # Use absolute value for complex error
        error = self.t - self.Phi @ mu


    def fit(self, track_iterations=np.arange(1, 1001, 5)):
        """Run EM algorithm and track weight evolution
        
        Parameters:
            track_iterations: list of iteration numbers at which to save weights
        
        Returns:
            mu: final weight estimates
            tracked_weights: dictionary with weights at specified iterations
        """
        if self.real:
            old_mu = np.ones(self.D)
            tracked_weights = np.zeros((len(track_iterations), self.D))
        else:
            old_mu = np.ones(self.D, dtype=complex)
            tracked_weights = np.zeros((len(track_iterations), self.D), dtype=complex)
        
        for iter in range(self.max_iter):
            # E-step
            mu, Sigma = self.estimate_posterior()
            
            # Save weights if current iteration is in track_iterations
            if iter + 1 in track_iterations:
                idx = np.where(track_iterations == iter + 1)[0][0]
                tracked_weights[idx] = mu.copy()
            
            # M-step
            self.maximize(mu, Sigma)
            
            # Check convergence
            change = np.max(np.abs(old_mu - mu))
            MSE = np.linalg.norm(self.t-self.Phi@mu)/np.linalg.norm(self.t)
            if change < 1e-4 and MSE<self.threshold:
                print(f"Converged after {iter+1} iterations")
                break
                
            old_mu = mu.copy()
        
        return mu, tracked_weights
    


class CoFEM_noSURE:
    def __init__(self, t, Phi, num_probes=10, max_iter=1000, threshold=1e-6, beta_in=_sentinel):
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
        self.beta_in = beta_in
        self.real = np.allclose(Phi, Phi.real) and np.allclose(t, t.real)

        # Initialize hyperparameters
        self.alpha = np.ones(self.D)  # Hyperparameters for precision of w
        if beta_in is _sentinel:
            self.beta = 1.0  # Noise precision (1/sigma^2)
        else:
            self.beta = beta_in
        
    def _samp_probes(self, size: Tuple[int, ...]) -> np.ndarray:
        """Sample Rademacher probes {±1}."""
        if self.real:
            # For real-valued data, use Rademacher probes
            return 2 * (np.random.rand(*size) > 0.5) - 1
        else:
            real_part = 2 * (np.random.rand(*size) > 0.5) - 1
            imag_part = 2 * (np.random.rand(*size) > 0.5) - 1
            return real_part + 1j * imag_part
    
    def _conj_grad(self, A, B, max_iters=1000, tol=1e-3):
        """Parallel Conjugate Gradient solver following Algorithm 2."""
        # Pre-compute constant terms
        X = np.zeros_like(B)
        R = B.copy()
        P = B.copy()
        
        # Use diagonal multiplication instead of matrix multiplication for M_inv
        M_inv = 1.0 / (self.beta * np.sum(self.Phi**2, axis=0) + self.alpha)
        W = M_inv[:, np.newaxis] * R  # Replace M_inv@R with element-wise multiplication
        
        # Keep dimensions with axis=0 and keepdims=True
        rho = np.sum(R * W, axis=0, keepdims=True)
        b_norm = np.linalg.norm(B, ord='fro')
        
        for u in range(max_iters):
            Psi = A(P)
            pi = np.sum(P * Psi, axis=0, keepdims=True)
            gamma = rho / pi
            
            # In-place operations
            X += P * gamma
            R -= Psi * gamma
            
            # Faster convergence check
            if np.linalg.norm(R, ord='fro') <= tol * b_norm:
                return X
                
            # Apply preconditioner with element-wise multiplication
            W = M_inv[:, np.newaxis] * R
            rho_old = rho
            rho = np.sum(R * W, axis=0, keepdims=True)
            P = W + P * (rho / rho_old)
        
        return X

    def estimate_posterior(self):
        """Simplified E-step following Algorithm 1."""
        def A(x):
            # Handle complex conjugate transpose for Fourier matrix
            Phi_T_Phi_x = self.Phi.conj().T @ (self.Phi @ x)
            return self.beta * Phi_T_Phi_x + np.diag(self.alpha) @ x

        # Sample Rademacher probes (K x D)
        K = self.num_probes
        probes = self._samp_probes((self.D,K))  # Shape: (D,K)
        
        # Define matrix B = [p1|p2|...|pK|βΦ^T y]
        beta_phi_T_y = (self.beta * self.Phi.conj().T @ self.t).reshape((self.D, 1))  # Shape: (D, 1)
        B = np.hstack([probes, beta_phi_T_y])  # Shape: (D, K+1)

        # Solve linear system
        # X = self._conj_grad(A, B)
        
       # Construct matrix A explicitly: A = βΦ^T Φ + diag{α}
        PhiT_Phi = self.Phi.conj().T @ self.Phi  # Shape: (D, D)
        A = self.beta * PhiT_Phi + np.diag(self.alpha)  # Shape: (D, D)
    
        # Solve system AX = B^T
        # B has shape (K+1, D), so B^T has shape (D, K+1)
        # Result X will have shape (D, K+1)
        X = np.linalg.solve(A, B)
        
        # Extract results
        mu = X[:,-1]  # Last row is μ
        x = X[:,:-1]  # First K rows are x1,...,xK
        
        # Compute variances sj = 1/K Σ(pk,j * xk,j)
        # s = np.mean(probes * x, axis=0)
        # s = (1 / self.beta) * np.clip(np.mean(probes * x, axis=1), 0, None)
        s = np.clip(np.mean(np.conj(probes) * x, axis=1), a_min=0, a_max=None)        
        return mu, s
    
    def maximize(self, mu: np.ndarray, s: np.ndarray):
        """M-step: Update alpha following Algorithm 1."""
        self.alpha = 1.0 / (np.square(mu) + s)

        # Update beta (noise precision)
        if self.beta_in is _sentinel:
            # Update alpha (precision of weights)
            error = self.t - self.Phi @ mu
            sum = np.sum(1 - self.alpha * s)
            self.beta = self.N/(np.sum(np.square(error))+sum/self.beta)
        
    def fit(self, track_iterations=np.arange(1, 1001, 5)):
        """Run CoFEM algorithm and track weight evolution."""
        # Initialize with complex dtype
        if self.real:
            old_mu = np.ones(self.D)
            tracked_weights = np.zeros((len(track_iterations), self.D))
        else:
            old_mu = np.ones(self.D, dtype=complex)
            tracked_weights = np.zeros((len(track_iterations), self.D), dtype=complex)
        for iter in range(self.max_iter):
            # E-step
            mu, s = self.estimate_posterior()
            
            # Save weights if current iteration is in track_iterations
            if iter + 1 in track_iterations:
                idx = np.where(track_iterations == iter + 1)[0][0]
                tracked_weights[idx] = mu.copy()  # Will preserve complex values
            
            # M-step
            self.maximize(mu, s)
            
            # Check convergence
            change = np.max(np.abs(old_mu - mu))
            MSE = np.linalg.norm(self.t-self.Phi@mu)/np.linalg.norm(self.t)
            """ if iter%1 == 0:
                print(f"Iteration {iter+1}/{self.max_iter}")
                print(f"Current change: {change}") """

            if change < 1e-2 and MSE<self.threshold:
                print(f"Converged after {iter+1} iterations")
                break
                
            old_mu = mu.copy()
        
        return mu, tracked_weights

