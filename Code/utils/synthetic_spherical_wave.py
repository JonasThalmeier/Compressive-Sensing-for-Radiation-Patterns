import numpy as np
from radiation.sphere_wave import Fmnc

def generate_spherical_wave(R=1, Theta_steps=18, Phi_steps=36, N_modes=50, c = 3, k=1, rho=0.1, sigma=0.01, seed=42):
    
    N = Theta_steps * Phi_steps
    D = 2*N_modes**2+4*N_modes
    np.random.seed(seed)
    # Step 1: Build matrix of spherical wave coefficients
    F = np.zeros((N, D, 3), dtype=np.complex_)
    nms_idx = np.zeros((D,3))
    ThetaPhi_idx = np.zeros((N,2))
    for Theta_idx, Theta in enumerate(np.linspace(1e-3, np.pi, Theta_steps)):
        for Phi_idx, Phi in enumerate(np.linspace(1e-3, 2*np.pi, Phi_steps)):
            ThetaPhi_idx[Theta_idx*Phi_steps + Phi_idx, :] = [Theta, Phi]
            for n in np.arange(1, N_modes+1):
                for m in np.arange(-n, n+1):
                    idx1 = (int(Theta_idx*Phi_steps + Phi_idx), int(n**2+n-1+m))
                    idx2 = (int(Theta_idx*Phi_steps + Phi_idx), int(n**2+n-1+m+D/2))
                    F[idx1[0], idx1[1], :], F[idx2[0], idx2[1], :] = Fmnc(m, n, c, R, Theta, Phi, k)
    for n in np.arange(1, N_modes+1):
                for m in np.arange(-n, n+1):
                    nms_idx[int(n**2+n-1+m),:] = [n, m, 1]
                    nms_idx[int(n**2+n-1+m+D/2),:] = [n, m, 2]
    # F = F/(np.max(np.abs(F), axis=(0,2))[np.newaxis, :, np.newaxis])  # Normalize F to avoid numerical issues
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
