import numpy as np
from utils.synthetic_data import generate_synthetic_data
_sentinel = object()
"""
Algorithm according to paper 'Sparse Bayesian Learning for Basis Selection' from Wimpf & Rao
"""
class SBL_EM:
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
        self.alpha = 1e-1*np.ones(self.D)
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
