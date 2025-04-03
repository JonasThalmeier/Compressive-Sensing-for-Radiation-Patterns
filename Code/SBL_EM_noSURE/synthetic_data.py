import numpy as np

def generate_synthetic_data(N, M, rho, sigma):
    """
    Generate synthetic data t = Phi * w + e.

    Parameters:
        N (int): Length of time-domain signal t.
        M (int): Length of frequency-domain signal w (M > N).
        rho (float): Sparsity factor (fraction of non-zero elements in w).
        sigma (float): Standard deviation of noise e.

    Returns:
        t (numpy.ndarray): Time-domain signal of length N.
        Phi (numpy.ndarray): Fourier transform matrix of size N x M.
        w (numpy.ndarray): Sparse frequency-domain signal of length M.
        e (numpy.ndarray): Noise vector of length N.
    """
    # Step 1: Create Fourier transform matrix (M x M)
    F = np.fft.fft(np.eye(M))  # Full Fourier transform matrix (complex-valued)
    F = F / np.sqrt(M)  # Normalize the Fourier transform matrix

    # Step 2: Select N rows randomly to form Phi (N x M)
    row_indices = np.random.choice(M, N, replace=False)
    Phi = F[row_indices, :].real  # Use only the real part of the selected rows

    # Step 3: Generate sparse vector w (length M)
    w = np.zeros(M)
    num_nonzero = int(rho * M)  # Number of non-zero elements
    nonzero_indices = np.random.choice(M, num_nonzero, replace=False)
    w[nonzero_indices] = np.random.randn(num_nonzero)  # Assign random values to non-zero elements

    # Step 4: Generate noise e (length N)
    e = np.random.normal(0, sigma, N)

    # Step 5: Compute t = Phi * w + e
    t = Phi @ w + e

    return t, Phi, w, e
