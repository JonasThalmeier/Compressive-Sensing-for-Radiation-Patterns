import numpy as np
from synthetic_data import generate_synthetic_data
"""
Algorithm according to paper 'Sparse Bayesian Learning for Basis Selection' from Wimpf & Rao
"""
class SBL_EM:
    def __init__(self, t, Phi, max_iter=1000, threshold=1e-6, beta=1.0):
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
        
        # Initialize hyperparameters
        self.alpha = np.ones(self.D)  # Hyperparameters for precision of w
        self.beta = beta  # Noise precision (1/sigma^2)

        self.gamma = np.ones(self.D)  # Variances of weights
        self.sigma_squared = 1/beta  # Noise variance
        
    def estimate_posterior(self):
        """E-step: Estimate posterior distribution of w"""
        # Compute posterior covariance using gamma and sigma_squared
        Sigma = np.linalg.inv(self.beta * self.Phi.T @ self.Phi + np.diag(1/self.gamma))
        # Compute posterior mean
        mu = self.beta * Sigma @ self.Phi.T @ self.t
        return mu, Sigma
    
    def maximize(self, mu, Sigma):
        """M-step: Update hyperparameters alpha and beta"""
        # Update alpha (precision of weights)
        # self.alpha = 1 / (np.square(mu) + np.diag(Sigma))
        
        # Update beta (noise precision)
        # error = self.t - self.Phi @ mu
        # gamma = np.sum(1 - self.alpha * np.diag(Sigma))
        # self.beta = (self.N - gamma) / (np.sum(np.square(error)))


        # Update gamma (variance of weights)
        self.gamma = np.square(mu) + np.diag(Sigma)
        
        # Update sigma_squared (noise variance)
        error = self.t - self.Phi @ mu
        N_eff = np.sum(self.gamma / (self.gamma + np.diag(Sigma)))
        # self.sigma_squared = (np.sum(np.square(error)) + self.sigma_squared * np.sum(np.ones(self.D)-np.diag(Sigma)/self.gamma)) / self.N
        # self.beta = 1 / self.sigma_squared

        
    def fit(self, track_iterations=np.arange(1, 1001, 100)):
        """Run EM algorithm and track weight evolution
        
        Parameters:
            track_iterations: list of iteration numbers at which to save weights
        
        Returns:
            mu: final weight estimates
            tracked_weights: dictionary with weights at specified iterations
        """
        old_mu = np.ones(self.D)
        tracked_weights = np.zeros((len(track_iterations), self.D))
        
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
            if change < self.threshold:
                print(f"Converged after {iter+1} iterations")
                break
                
            old_mu = mu.copy()
        
        return mu, tracked_weights
