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
            t: Target signal (N x 1)
            Phi: Dictionary matrix (N x D)
            max_iter: Maximum number of iterations
            threshold: Convergence threshold
            beta_in: Noise precision (if None, estimated from data)
"""
        # Convert inputs to numpy arrays
        t = np.asarray(t)
        Phi = np.asarray(Phi)
        
        # For vector output mode (N, L)
        self.vector_mode = True
        self.L = t.shape[1]  # L = 3 in this case
        
        # Reshape inputs for vectorized processing
        self.t = t.reshape(-1, 1)  # Flatten (50,3) to (150,1)
        self.Phi = Phi.transpose(0, 2, 1).reshape(-1, Phi.shape[1])  # Reshape (50,150,3) to (150,150)
        
        self.N = t.shape[0]  # 50
        self.D = Phi.shape[1]  # 150
        self.max_iter = max_iter
        self.threshold = threshold
        self.active_basis = []
        self.alpha = np.inf * np.ones(self.D, dtype=complex)
        
        self._initialize_beta(beta_in)
        self.initialize_first_basis()

    def _initialize_beta(self, beta_in):
        """Initialize noise precision based on input type"""
        if beta_in is not None:
            self.beta = beta_in
        else:
            if self.vector_mode:
                # Use variance across all vector components
                self.beta = 1.0 / (0.1 * np.var(self.t))
            else:
                self.beta = 1.0 / (0.1 * np.var(self.t))

    def initialize_first_basis(self):
        """Initialize model with the most relevant basis"""
        sigma2 = 1.0 / self.beta if self.beta != 0 else 0.1 * np.var(self.t)
        
        # Compute scores using vectorized operations
        scores = []
        for Phi_i in self.Phi.T:
            Phi_i = Phi_i.reshape(-1, 1)
            numerator = (Phi_i.T @ self.t).item() ** 2
            denominator = (Phi_i.T @ Phi_i).item()
            scores.append(numerator / denominator)
            
        scores = np.array(scores)
        valid = scores > sigma2
        
        if np.any(valid):
            i = np.argmax(scores)
            self.active_basis = [i]
            Phi_active = self.Phi[:, self.active_basis]
            self.alpha[i] = (Phi_active.T @ Phi_active).item() / (scores[i] - sigma2)
            self.Sigma = np.linalg.inv(
                np.diag(self.alpha[self.active_basis]) + 
                self.beta * (Phi_active.T @ Phi_active)
            )
            self.mu = self.beta * self.Sigma @ Phi_active.T @ self.t
        else:
            self.Sigma = np.zeros((0, 0))
            self.mu = np.zeros(0)

    def compute_S_Q(self):
        """Compute basis selection metrics"""
        if not self.active_basis:
            S = self.beta * np.sum(self.Phi**2, axis=0)
            Q = self.beta * (self.Phi.T @ self.t).flatten()
            return S, Q

        Phi_active = self.Phi[:, self.active_basis]
        term1_S = self.beta * np.sum(self.Phi**2, axis=0)
        term2_S = self.beta**2 * np.sum(
            (self.Phi @ (Phi_active @ self.Sigma)) * self.Phi, 
            axis=0
        )
        S = term1_S - term2_S
        
        term1_Q = self.beta * (self.Phi.T @ self.t).flatten()
        term2_Q = self.beta**2 * self.Phi.T @ Phi_active @ self.Sigma @ (Phi_active.T @ self.t)
        Q = term1_Q - term2_Q.flatten()
        return S, Q

    # Remaining methods (compute_s_q_theta, select_action, update_model, etc.)
    # remain unchanged from the previous vector implementation

    def fit(self):
        """Run the SBL algorithm"""
        for _ in range(self.max_iter):
            S, Q = self.compute_S_Q()
            s, q, theta = self.compute_s_q_theta(S, Q)
            action, i, new_alpha, delta = self.select_action(theta, S, Q)
            if delta < self.threshold:
                break
            self.update_model(action, i, new_alpha)
            
        w_est = np.zeros(self.D_orig)
        if self.active_basis:
            w_est[self.active_basis] = self.mu.flatten()
        return w_est
    

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
            delta = 0.5 * ((np.abs(Q[i])**2 / S[i]) - 1 - np.log(np.abs(Q[i])**2 / S[i]))
            actions.append(('add', i, S[i]**2 / (np.abs(Q[i])**2 - S[i]), delta))
        # Delete action
        del_mask = (theta <= 0) & ~np.isinf(self.alpha)
        if np.any(del_mask):
            i = self.active_basis[np.argmin(theta[self.active_basis])]
            delta = -0.5 * ((Q[i]**2 / (S[i] + self.alpha[i])) - np.log(1 + S[i]/self.alpha[i]))
            actions.append(('delete', i, None, delta))
        # Re-estimate action
        re_mask = (theta > 0) & ~np.isinf(self.alpha)
        if np.any(re_mask):
            i = self.active_basis[np.argmax(theta[self.active_basis])]
            new_alpha = (S[i]**2) / (Q[i]**2 - S[i])
            delta = 0.5 * ((Q[i]**2 * (1/new_alpha - 1/self.alpha[i])) / (S[i] * (1/new_alpha - 1/self.alpha[i]) + 1) - np.log(1 + S[i] * (1/new_alpha - 1/self.alpha[i])))
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
