import numpy as np
import matplotlib.pyplot as plt
from utils.plot_settings import get_figsize, LINE_STYLES, DPI
import os
from SBL_algs.SBL_Fast import SBL_Fast_Vector
from utils.Process_sim_data import load_nearfield, transform_nearfield_to_vector, transform_vector_to_nearfield
from radiation.sphere_wave import abF_matrix_alt, F_matrix_alt, Fmnc, F_expansion, nm_expansion, inverse_nm_expansion
import importlib
from utils.plot import plot_Fcoefficient_magnitudes, plot3D, plot_abcoefficient_magnitudes, vec_MSE, plot_abcoefficient_magnitudes

R = 0.2759
f = 2.4e9
k = 2*np.pi*f/(3e8)
filename = 'loop_farfield_spherical.txt'
N_modes = 15
D = 2*N_modes**2 + 4*N_modes

# filename = 'loop_farfield_spherical.txt'
Efield = load_nearfield(filename, undersampling=5)
Efield_vec = transform_nearfield_to_vector(Efield)

ab_coeffs_exp = F_expansion(Efield, N_modes, k, R, NF=True)
abF,nms_idx,thetaphi_idx = F_matrix_alt(Efield.theta, Efield.phi, R=R, N_modes=N_modes, k=k)
sbl_vec_F = SBL_Fast_Vector(Efield_vec, abF, max_iter=1000, threshold=1e-8)
w_est_F, basis = sbl_vec_F.fit()
weights_sbl = np.concatenate([nms_idx, w_est_F.reshape(-1,1)], axis=1)
weights_exp = np.concatenate([nms_idx, ab_coeffs_exp.reshape(-1,1)], axis=1)
weigh_list = [weights_sbl, weights_exp]
# Create figure
plt.figure(figsize=(12, 6))

# # Plot settings
# bar_width = 0.25
# index = np.arange(D)

# # Plot all three together
# plt.bar(index - bar_width, np.abs(ab_coeffs_exp).ravel(), width=bar_width, 
#         color='black', label='True Weights')
# plt.bar(index, np.abs(w_est_F), width=bar_width, 
#         color='red', alpha=0.7, label='EM Estimate')
# plt.show()

# Set matrix type string for title and filename

# Formatting with matrix type in title
plt.xlabel('Weight Index')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0,1])
plt.tight_layout()

plot_Fcoefficient_magnitudes(
    (weights_sbl[:,0], weights_sbl[:,1], weights_sbl[:,2], weights_sbl[:,3]),
    (weights_exp[:,0], weights_exp[:,1], weights_exp[:,2], weights_exp[:,3])
)

