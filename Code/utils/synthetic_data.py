import numpy as np
from radiation.sphere_wave import F_matrix

def generate_synthetic_data_vec(N, D, L, rho, sigma, seed=42):
    """
    Generate synthetic data t = Phi * w + e for L different measurements.

    Parameters:
        N (int): Length of time-domain signal t.
        D (int): Length of frequency-domain signal w (D > N).
        L (int): Number of different measurements/matrices.
        rho (float): Sparsity factor (fraction of non-zero elements in w).
        sigma (float): Standard deviation of noise e.

    Returns:
        t (numpy.ndarray): Time-domain signals of shape (N, L).
        Phi (numpy.ndarray): Transform matrices of shape (N, D, L).
        w (numpy.ndarray): Sparse frequency-domain signal of length D.
        e (numpy.ndarray): Noise vectors of shape (N, L).
    """
    # Set random seed
    np.random.seed(seed)
    
    # Step 1: Create transform matrices (D x D x L)
    F = np.sqrt(1/N)*np.random.randn(D, D, L).astype(np.complex_) + \
        1j*np.sqrt(1/N)*np.random.randn(D, D, L).astype(np.complex_)

    # Step 2: Select N rows randomly to form Phi (N x D x L)
    row_indices = np.random.choice(D, N, replace=False)
    Phi = F[row_indices, :, :]

    # Step 3: Generate sparse vector w (length D)
    w = np.zeros(D)
    num_nonzero = int(rho * D)
    nonzero_indices = np.random.choice(D, num_nonzero, replace=False)
    w[nonzero_indices] = np.random.randn(num_nonzero)

    # Step 4: Generate noise e (N x 1 x L)
    e = np.random.normal(0, sigma, (N, L)).astype(np.complex_) + \
        1j * np.random.normal(0, sigma, (N, L)).astype(np.complex_)

    # Step 5: Compute t = Phi * w + e for each measurement
    t = np.zeros((N, L), dtype=np.complex_)
    w = w.reshape(-1, 1).flatten()
    for l in range(L):
        # Fix: Reshape w and flatten the result of matrix multiplication
        result = Phi[:, :, l] @ w  # Shape: (N, 1)
        t[:, l] = result.flatten() + e[:, l]      # Now shapes match: (N,) + (N,)

    return t, Phi, w, e


def generate_synthetic_data_spherical_wave(R=1, Theta_steps=18, Phi_steps=36, N_modes=50, c = 3, k=1, rho=0.1, sigma=0.01, seed=42):
    
    N = Theta_steps * Phi_steps
    D = 2*N_modes**2+4*N_modes
    np.random.seed(seed)
    # Step 1: Build matrix of spherical wave coefficients
    F, nms_idx, ThetaPhi_idx = F_matrix(R=R, Theta_steps=Theta_steps, Phi_steps=Phi_steps, N_modes=N_modes, c = c, k=k)
    # Step 2: Generate sparse vector w (length D)
    w = np.zeros(D)
    num_nonzero = int(rho * D)
    nonzero_indices = np.random.choice(D, num_nonzero, replace=False)
    w[nonzero_indices] = np.random.randn(num_nonzero)

    # Step 3: Generate noise e (N x 1 x L)
    e = np.random.normal(0, sigma, (N, 3)).astype(np.complex_) + \
        1j * np.random.normal(0, sigma, (N, 3)).astype(np.complex_)

    # Step 4: Compute t = F * w + e for each measurement
    t = np.zeros((N, 3), dtype=np.complex_)
    w = w.reshape(-1, 1).flatten()
    for l in range(3):
        # Fix: Reshape w and flatten the result of matrix multiplication
        result = F[:, :, l] @ w  # Shape: (N, 1)
        t[:, l] = result.flatten() + e[:, l]      # Now shapes match: (N,) + (N,)

    return t, F, w, e, nms_idx, ThetaPhi_idx
