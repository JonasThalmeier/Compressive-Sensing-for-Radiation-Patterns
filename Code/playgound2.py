import numpy as np
import matplotlib.pyplot as plt
from utils.synthetic_data import generate_synthetic_data
from EM_algs.EM_wo_SURE import SBL_EM
from EM_algs.SBL_Fast import SBL_Fast
from utils.plot_settings import get_figsize, LINE_STYLES, DPI
from utils.synthetic_data_vector import generate_synthetic_data_vec
import os
from EM_algs.SBL_Fast_Vector import SBL_Fast_Vector



if __name__ == "__main__":
    # Generate synthetic data
    N = 50  # Length of time-domain signal
    D = 150  # Length of frequency-domain signal (D > N)
    L = 3
    rho = 0.1  # Sparsity factor
    sigma = 0.01  # Standard deviation of noise
    threshold = 1e-6  # Convergence threshold
    max_iter = 200
    t, Phi, w_true, e = generate_synthetic_data_vec(N, D, L, rho, sigma)
    print(f"t.shape: {t.shape}, Phi.shape: {Phi.shape}, w_true.shape: {w_true.shape}, e.shape: {e.shape}")
    # Run both algorithms for comparison
    
    # Run CoFEM
    sbl_vec = SBL_Fast_Vector(t, Phi, max_iter=max_iter, threshold=threshold)
    w_est, _ = sbl_vec.fit()