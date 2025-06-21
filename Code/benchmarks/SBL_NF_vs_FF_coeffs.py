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
r = 0.0625
f = 2.4e9
k = 2*np.pi*f/(3e8)
filename_FF = 'dipole_farfield_spherical.txt'
filename_NF = 'dipole_nearfield_spherical.txt'
N_modes = 15
D = 2*N_modes**2 + 4*N_modes

# NF-----------------------------------------
Efield_NF = load_nearfield(filename_NF, undersampling=5)
Efield_NF_vec = transform_nearfield_to_vector(Efield_NF)

F_NF,nms_idx,thetaphi_idx = F_matrix_alt(Efield_NF.theta, Efield_NF.phi, R=r, N_modes=N_modes, k=k)
sbl_vec_F_NF = SBL_Fast_Vector(Efield_NF_vec, F_NF, max_iter=1000, threshold=1e-8)
w_est_F_NF, basis = sbl_vec_F_NF.fit()
weights_sbl_NF = np.concatenate([nms_idx, w_est_F_NF.reshape(-1,1)], axis=1)


# FF-----------------------------------------
Efield_FF = load_nearfield(filename_FF, undersampling=5)
Efield_FF_vec = transform_nearfield_to_vector(Efield_FF)

F_FF,nms_idx,thetaphi_idx = F_matrix_alt(Efield_FF.theta, Efield_FF.phi, R=R, N_modes=N_modes, k=k)
sbl_vec_F_FF = SBL_Fast_Vector(Efield_FF_vec, F_FF, max_iter=1000, threshold=1e-8)
w_est_F_FF, basis = sbl_vec_F_FF.fit()
weights_sbl_FF = np.concatenate([nms_idx, w_est_F_FF.reshape(-1,1)], axis=1)

weigh_list = [weights_sbl_NF, weights_sbl_FF]
# Create figure
plt.figure(figsize=(12, 6))

# Plot settings
bar_width = 0.25
index = np.arange(D)

# Plot all three together
plt.bar(index - bar_width, np.abs(w_est_F_NF).ravel(), width=bar_width, 
        color='black', label='True Weights')
plt.bar(index, np.abs(w_est_F_FF), width=bar_width, 
        color='red', alpha=0.7, label='EM Estimate')

# Set matrix type string for title and filename

# Formatting with matrix type in title
plt.xlabel('Weight Index')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0,1])
plt.tight_layout()
plt.show()

plot_Fcoefficient_magnitudes(
    (weights_sbl_NF[:,0], weights_sbl_NF[:,1], weights_sbl_NF[:,2], weights_sbl_NF[:,3]),
    (weights_sbl_FF[:,0], weights_sbl_FF[:,1], weights_sbl_FF[:,2], weights_sbl_FF[:,3])
)

