import numpy as np
import matplotlib.pyplot as plt
from SBL_algs.EM_algs import EM_noSURE
from utils.plot_settings import get_figsize, LINE_STYLES, DPI
import os
from SBL_algs.SBL_Fast import SBL_Fast_Vector
from utils.synthetic_data import generate_synthetic_data_vec, generate_synthetic_data_spherical_wave



if __name__ == "__main__":
    seed = np.random.randint(0, 10000)
    # Generate synthetic data
    N_modes=7
    N = 20  # Length of time-domain signal
    D = 2*N_modes**2+4*N_modes  # Length of frequency-domain signal (D > N)
    L = 3
    rho = 0.1  # Sparsity factor
    sigma = 1e-1  # Standard deviation of noise
    threshold = 1e-8  # Convergence threshold
    max_iter = 200
    eta = 376.7
    f = 3e9
    lamb = 3e8 / f
    k = 2 * np.pi / lamb
    r= 0.4
    t, Phi, w_true, e,_,_ = generate_synthetic_data_spherical_wave(R=r, Phi_steps=36, Theta_steps=18, N_modes=N_modes, seed=seed, sigma=sigma, k=k)
    print(f"t.shape: {t.shape}, Phi.shape: {Phi.shape}, w_true.shape: {w_true.shape}, e.shape: {e.shape}")
    # Run both algorithms for comparison
    print("Any NaN in Phi: " + str(np.any(np.isnan(Phi))))
    # Run CoFEM
    sbl_vec = SBL_Fast_Vector(t, Phi, max_iter=max_iter, threshold=threshold)
    w_est, _ = sbl_vec.fit()

        # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot settings
    bar_width = 0.5
    index = np.arange(D)
    
    # Plot all three together
    plt.bar(index - bar_width, np.abs(w_true), width=bar_width, 
            color='black', label='True Weights')
    plt.bar(index, np.abs(w_est), width=bar_width, 
            color='red', alpha=0.7, label='EM Estimate')

    
    
    # Formatting with matrix type in title
    plt.xlabel('Weight Index')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
