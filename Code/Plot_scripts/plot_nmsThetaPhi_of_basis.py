import numpy as np
import matplotlib.pyplot as plt
from utils.synthetic_data import generate_synthetic_data_vec, generate_synthetic_data_spherical_wave
from SBL_algs.EM_algs import EM_noSURE, CoFEM_noSURE
from SBL_algs.SBL_Fast import SBL_Fast, SBL_Fast_Vector
from utils.plot_settings import get_figsize, LINE_STYLES, DPI
from utils.synthetic_data import generate_synthetic_data_vec
import os




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
    reps = 10
    nms, ThetaPhi = np.zeros((0,3)), np.zeros((0,2))
    nms_true, ThetaPhi_true = np.zeros((0,3)), np.zeros((0,2))
    for i in range(0, reps):
        seed = np.random.randint(0, 10000)
        t, Phi, w_true, e, nms_idx, ThetaPhi_idx = generate_synthetic_data_spherical_wave(Phi_steps=36, Theta_steps=18, N_modes=N_modes, seed=seed, sigma=sigma)
        print(f"t.shape: {t.shape}, Phi.shape: {Phi.shape}, w_true.shape: {w_true.shape}, e.shape: {e.shape}")
        # Run both algorithms for comparison
        print("Any NaN in Phi: " + str(np.any(np.isnan(Phi))))
        # Run CoFEM
        sbl_vec = SBL_Fast_Vector(t, Phi, max_iter=max_iter, threshold=threshold)
        w_est, active_basis = sbl_vec.fit()
        mask = active_true = np.where(w_true != 0)[0]
        nms_true = np.concatenate((nms_true, nms_idx[mask,:]), axis=0)
        nms = np.concatenate((nms, nms_idx[active_basis,:]), axis=0)



# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# Plot 1: True n-m scatter plot with different colors for s=1 and s=2
mask_s1_true = nms_true[:, 2] == 1
mask_s2_true = nms_true[:, 2] == 2

scatter4_s1 = ax1.scatter(nms_true[mask_s1_true, 0], nms_true[mask_s1_true, 1], 
                         label='s=1', alpha=0.6, color='blue')
scatter4_s2 = ax1.scatter(nms_true[mask_s2_true, 0], nms_true[mask_s2_true, 1], 
                         label='s=2', alpha=0.6, color='red')
ax1.set_xlabel('n')
ax1.set_ylabel('m')
ax1.set_title('Distribution of True Basis Functions\nin n-m Space')

# Set axis ranges and integer grid for true basis plot
ax1.set_xlim(1, N_modes)
ax1.set_ylim(-N_modes, N_modes)
ax1.set_xticks(range(1, N_modes + 1))
ax1.set_yticks(range(-N_modes, N_modes + 1))
ax1.grid(True)
ax1.legend()
# Plot 2: n-m scatter plot with different colors for s=1 and s=2
mask_s1 = nms[:, 2] == 1
mask_s2 = nms[:, 2] == 2

scatter2_s1 = ax2.scatter(nms[mask_s1, 0], nms[mask_s1, 1], 
                         label='s=1', alpha=0.6, color='blue')
scatter2_s2 = ax2.scatter(nms[mask_s2, 0], nms[mask_s2, 1], 
                         label='s=2', alpha=0.6, color='red')
ax2.set_xlabel('n')
ax2.set_ylabel('m')
ax2.set_title('Distribution of Active Basis Functions\nin n-m Space')

# Set axis ranges and integer grid
ax2.set_xlim(1, N_modes)
ax2.set_ylim(-N_modes, N_modes)
ax2.set_xticks(range(1, N_modes + 1))
ax2.set_yticks(range(-N_modes, N_modes + 1))
ax2.grid(True)
ax2.legend()




plt.tight_layout()
plt.show()