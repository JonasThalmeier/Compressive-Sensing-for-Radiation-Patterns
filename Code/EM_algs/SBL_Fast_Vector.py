import numpy as np

class SBL_Fast_Vector:
    """
    Fast Sparse Bayesian Learning algorithm implementation.
    Uses efficient basis selection and pruning strategy to speed up convergence.
    Based on Tipping & Faul's "Fast Marginal Likelihood Maximisation for Sparse Bayesian Models" (2003).
    """
    def __init__(self, t, Phi, max_iter=1000, threshold=1e-4, beta_in=None):
        """
Initialize Fast SBL algorithm.
        
        Args:
            t: Target signal (N x 1 x L)
            Phi: Dictionary matrix (N x D x L)
            max_iter: Maximum number of iterations
            threshold: Convergence threshold
            beta_in: Noise precision (if None, estimated from data)
"""
        self.N, self.D, self.L = Phi.shape
        if t.shape != (self.N, self.L):
            raise ValueError("t must have shape (N, L)")
        self.Phi = Phi.transpose(0, 2, 1).reshape(-1, Phi.shape[1])
        self.t = t.reshape(-1, 1)
        self.max_iter = max_iter
        self.threshold = threshold
        self.active_basis = []
        # Initialize with complex support
        self.alpha = np.inf * np.ones(self.D, dtype=complex)
        # Use complex variance for initialization
        self.beta = beta_in if beta_in is not None else 1.0 / (0.1 * np.var(np.abs(self.t)))
        self.initialize_first_basis()

    def initialize_first_basis(self):
        """Initialize model with complex support."""
        sigma2 = 1.0 / self.beta if self.beta != 0 else 0.1 * np.var(np.abs(self.t))
        # Use conjugate transpose for complex dot products
        scores = [np.abs(Phi_i.conj() @ self.t)**2 / (Phi_i.conj() @ Phi_i).real 
                 for Phi_i in self.Phi.T]
        scores = np.array(scores)
        valid = scores > sigma2
        
        if np.any(valid):

            i = np.argmax(scores)
            Phi_i = self.Phi[:, i]
            self.active_basis = [i]
            # Use complex operations for alpha initialization
            self.alpha[i] = ((Phi_i.conj() @ Phi_i).real / (scores[i] - sigma2))
            Phi_active = self.Phi[:, self.active_basis]
            # Use conjugate transpose for complex matrices
            self.Sigma = np.linalg.inv(np.diag(self.alpha[self.active_basis]) + 
                                     self.beta * Phi_active.conj().T @ Phi_active)
            self.mu = self.beta * self.Sigma @ Phi_active.conj().T @ self.t
        else:
            self.Sigma = np.zeros((0, 0), dtype=complex)
            self.mu = np.zeros(0, dtype=complex)

    def compute_S_Q(self):
        """Compute S and Q with complex support."""
        if not self.active_basis:
            # Use conjugate transpose for complex matrices
            S = self.beta * np.sum(np.abs(self.Phi)**2, axis=0)
            Q = self.beta * (self.Phi.conj().T @ self.t)
            return S, Q

        Phi_active = self.Phi[:, self.active_basis]
        # Modify matrix operations for complex support
        term1_S = self.beta * np.sum(np.abs(self.Phi)**2, axis=0)
        term2_S = self.beta**2 * np.sum(np.abs(self.Sigma @ 
                  (Phi_active.conj().T @ self.Phi))**2, axis=0)
        S = term1_S - term2_S
        
        term1_Q = self.beta * (self.Phi.conj().T @ self.t)
        term2_Q = self.beta**2 * self.Phi.conj().T @ Phi_active @ self.Sigma @ \
                  (Phi_active.conj().T @ self.t)
        Q = term1_Q.flatten() - term2_Q.flatten()
        return S, Q

    def compute_s_q_theta(self, S, Q):
        s = np.where(np.isinf(self.alpha), S, (self.alpha * S) / (self.alpha - S))
        q = np.where(np.isinf(self.alpha), Q, (self.alpha * Q) / (self.alpha - S))
        # Should use absolute values squared for complex numbers
        theta = np.abs(q)**2 - s  # s should also use abs() for complex case
        return s, q, theta

    def select_action(self, theta, S, Q):
        """
        Select the most beneficial action: add, delete, or re-estimate basis function.
        Returns tuple of (action, index, new_alpha, change_in_likelihood).
        """
        actions = []
        # Consider adding new basis
        add_mask = (theta > 0) & np.isinf(self.alpha)
        if np.any(add_mask):
            i = np.argmax(theta * add_mask)
            # Should use absolute values for complex Q
            delta = np.abs(Q[i])**2 / S[i] - 1 + np.log(S[i]/np.abs(Q[i])**2)
            # delta = 0.5 * ((np.abs(Q[i])**2 / S[i]) - 1 - np.log(np.abs(Q[i])**2 / S[i]))
            actions.append(('add', i, S[i]**2 / (np.abs(Q[i])**2 - S[i]), delta))
        # Delete action
        del_mask = (theta <= 0) & ~np.isinf(self.alpha)
        if np.any(del_mask):
            i = self.active_basis[np.argmin(theta[self.active_basis])]
            delta = np.abs(Q[i])**2 / (S[i] - self.alpha[i]) - np.log(1 - S[i]/self.alpha[i])
            # delta = -0.5 * ((Q[i]**2 / (S[i] + self.alpha[i])) - np.log(1 + S[i]/self.alpha[i]))
            actions.append(('delete', i, None, delta))
        # Re-estimate action
        re_mask = (theta > 0) & ~np.isinf(self.alpha)
        if np.any(re_mask):
            i = self.active_basis[np.argmax(theta[self.active_basis])]
            new_alpha = (S[i]**2) / (Q[i]**2 - S[i])
            delta = np.abs(Q[i])**2/(S[i]+1/(1/new_alpha-1/self.alpha[i])) - np.log(1 + S[i] * (1/new_alpha - 1/self.alpha[i]))
            # delta = 0.5 * ((Q[i]**2 * (1/new_alpha - 1/self.alpha[i])) / (S[i] * (1/new_alpha - 1/self.alpha[i]) + 1) - np.log(1 + S[i] * (1/new_alpha - 1/self.alpha[i])))
            actions.append(('reestimate', i, new_alpha, delta))
        return max(actions, key=lambda x: x[3], default=(None, None, None, -np.inf))

    def update_model(self, action, i, new_alpha):
        if action == 'add':
            self._add_basis(i, new_alpha)
        elif action == 'delete':
            self._delete_basis(i)
        elif action == 'reestimate':
            self._reestimate_alpha(i, new_alpha)

    def _add_basis(self, i, new_alpha):
        self.active_basis.append(i)
        self.alpha[i] = new_alpha
        Phi_active = self.Phi[:, self.active_basis]
        # Missing conjugate transpose
        self.Sigma = np.linalg.inv(np.diag(self.alpha[self.active_basis]) + 
                                  self.beta * Phi_active.conj().T @ Phi_active)
        self.mu = self.beta * self.Sigma @ Phi_active.conj().T @ self.t

    def _delete_basis(self, i):
        idx = self.active_basis.index(i)
        self.active_basis.pop(idx)
        self.alpha[i] = np.inf
        if self.active_basis:
            Phi_active = self.Phi[:, self.active_basis]
            self.Sigma = np.linalg.inv(np.diag(self.alpha[self.active_basis]) + self.beta * Phi_active.T @ Phi_active)
            self.mu = self.beta * self.Sigma @ Phi_active.T @ self.t
        else:
            self.Sigma = np.zeros((0, 0))
            self.mu = np.zeros(0)

    def _reestimate_alpha(self, i, new_alpha):
        self.alpha[i] = new_alpha
        Phi_active = self.Phi[:, self.active_basis]
        self.Sigma = np.linalg.inv(np.diag(self.alpha[self.active_basis]) + self.beta * Phi_active.T @ Phi_active)
        self.mu = self.beta * self.Sigma @ Phi_active.T @ self.t

    def fit(self):
        for _ in range(self.max_iter):
            S, Q = self.compute_S_Q()
            s, q, theta = self.compute_s_q_theta(S, Q)
            action, i, new_alpha, delta = self.select_action(theta, S, Q)
            if delta < self.threshold:
                break
            self.update_model(action, i, new_alpha)
        w_est = np.zeros(self.D, dtype=complex)
        w_est[self.active_basis] = self.mu.flatten()
        return w_est, None
        # return self.mu, np.array(self.active_basis)