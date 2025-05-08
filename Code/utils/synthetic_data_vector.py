import numpy as np

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
    for l in range(L):
        # Fix: Reshape w and flatten the result of matrix multiplication
        result = Phi[:, :, l] @ w.reshape(-1, 1)  # Shape: (N, 1)
        t[:, l] = result.flatten() + e[:, l]      # Now shapes match: (N,) + (N,)

    return t, Phi, w, e
