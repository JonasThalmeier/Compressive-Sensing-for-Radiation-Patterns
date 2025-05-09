import numpy as np
import matplotlib.pyplot as plt
from utils.synthetic_data import generate_synthetic_data
from EM_algs.EM_wo_SURE import SBL_EM
from EM_algs.SBL_Fast import SBL_Fast
from utils.plot_settings import get_figsize, LINE_STYLES, DPI
from utils.synthetic_data_vector import generate_synthetic_data_vec
import os
from EM_algs.SBL_Fast_Vector import SBL_Fast_Vector
from utils.synthetic_spherical_wave import generate_spherical_wave



if __name__ == "__main__":
    seed = np.random.randint(0, 10000)
    # Generate synthetic data
    N_modes=7
    N = 20  # Length of time-domain signal
    D = 2*N_modes**2+4*N_modes  # Length of frequency-domain signal (D > N)
    L = 3
    rho = 0.1  # Sparsity factor
    sigma = 0.0001  # Standard deviation of noise
    threshold = 1e-6  # Convergence threshold
    max_iter = 200
    t, Phi, w_true, e = generate_spherical_wave(Phi_steps=18, Theta_steps=9, N_modes=N_modes, seed=seed, sigma=sigma)
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
