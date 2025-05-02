import numpy as np

def generate_synthetic_data(N, D, rho, sigma, seed=42, FFT=True):
    """
    Generate synthetic data t = Phi * w + e.

    Parameters:
        N (int): Length of time-domain signal t.
        D (int): Length of frequency-domain signal w (D > N).
        rho (float): Sparsity factor (fraction of non-zero elements in w).
        sigma (float): Standard deviation of noise e.

    Returns:
        t (numpy.ndarray): Time-domain signal of length N.
        Phi (numpy.ndarray): Fourier transform matrix of size N x D.
        w (numpy.ndarray): Sparse frequency-domain signal of length D.
        e (numpy.ndarray): Noise vector of length N.
    """

    # Set random seed
    np.random.seed(seed)
    
    # Step 1: Create Fourier transform matrix (D x D)
    if FFT:
        F = np.fft.fft(np.eye(D))/np.sqrt(D)  # Full Fourier transform matrix (complex-valued)
    else:
        F = np.sqrt(1/N)*np.random.randn(D, D)

    # Step 2: Select N rows randomly to form Phi (N x D)
    row_indices = np.random.choice(D, N, replace=False)
    Phi = F[row_indices, :]

    # Step 3: Generate sparse vector w (length D)
    w = np.zeros(D)
    num_nonzero = int(rho * D)  # Number of non-zero elements
    nonzero_indices = np.random.choice(D, num_nonzero, replace=False)
    w[nonzero_indices] = np.random.randn(num_nonzero)  # Assign random values to non-zero elements

    # Step 4: Generate noise e (length N)
    e = np.random.normal(0, sigma, N)

    # Step 5: Compute t = Phi * w + e
    t = Phi @ w + e

    return t, Phi, w, e
