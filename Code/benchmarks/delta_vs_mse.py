import numpy as np
import matplotlib.pyplot as plt
from utils.plot_settings import get_figsize, LINE_STYLES, DPI
import os
from SBL_algs.SBL_Fast import SBL_Fast_Vector
from utils.Process_sim_data import load_nearfield, transform_nearfield_to_vector, transform_vector_to_nearfield
from radiation.sphere_wave import abF_matrix_alt, F_matrix_alt, Fmnc, F_expansion, nm_expansion, inverse_nm_expansion
import importlib
from utils.plot import plot_Fcoefficient_magnitudes, plot3D, plot_abcoefficient_magnitudes, vec_MSE
import os

R = 0.2759
f = 2.4e9
k = 2*np.pi*f/(3e8)
filename = 'dipole_farfield_spherical.txt'
N_modes = 25
D = 2*N_modes**2 + 4*N_modes

Efield_org = load_nearfield(filename, undersampling=7)
Efield_org_vec = transform_nearfield_to_vector(Efield_org)
F_org, nms_idx, _ = F_matrix_alt(Efield_org.theta, Efield_org.phi, R=R, N_modes=N_modes, k=k)

undersampling_range = range(7, 17, 1)
deltas_d = []
mses_d = []

for undersampling in undersampling_range:
    # Load and process farfield with current undersampling
    Efield_ff = load_nearfield(filename, undersampling=undersampling)
    Efield_ff_vec = transform_nearfield_to_vector(Efield_ff)
    N = len(Efield_ff.phi) * len(Efield_ff.theta)
    delta = N / D
    deltas_d.append(delta)
    
    # Forward matrix and SBL reconstruction
    F, nms_idx, _ = F_matrix_alt(Efield_ff.theta, Efield_ff.phi, R=R, N_modes=N_modes, k=k)
    sbl_vec_F = SBL_Fast_Vector(Efield_ff_vec, F, max_iter=100, threshold=1e-8)
    w_est_F, basis = sbl_vec_F.fit()
    
    # Reconstruct field
    Efield_SBLF_vec = np.zeros_like(Efield_org_vec)
    for i in range(3):
        Efield_SBLF_vec[:,i] = F_org[:,:,i] @ w_est_F
    # Compute relative MSE
    mse = vec_MSE(Efield_SBLF_vec, Efield_org_vec)
    mses_d.append(mse)


deltas_l = []
mses_l = []
filename = 'loop_farfield_spherical.txt'
Efield_org = load_nearfield(filename, undersampling=7)
Efield_org_vec = transform_nearfield_to_vector(Efield_org)
F_org, nms_idx, _ = F_matrix_alt(Efield_org.theta, Efield_org.phi, R=R, N_modes=N_modes, k=k)
for undersampling in undersampling_range:
    # Load and process farfield with current undersampling
    Efield_ff = load_nearfield(filename, undersampling=undersampling)
    Efield_ff_vec = transform_nearfield_to_vector(Efield_ff)
    N = len(Efield_ff.phi) * len(Efield_ff.theta)
    delta = N / D
    deltas_l.append(delta)
    
    # Forward matrix and SBL reconstruction
    F, nms_idx, _ = F_matrix_alt(Efield_ff.theta, Efield_ff.phi, R=R, N_modes=N_modes, k=k)
    sbl_vec_F = SBL_Fast_Vector(Efield_ff_vec, F, max_iter=100, threshold=1e-8)
    w_est_F, basis = sbl_vec_F.fit()
    
    # Reconstruct field
    Efield_SBLF_vec = np.zeros_like(Efield_org_vec)
    for i in range(3):
        Efield_SBLF_vec[:,i] = F_org[:,:,i] @ w_est_F
    # Compute relative MSE
    mse = vec_MSE(Efield_SBLF_vec, Efield_org_vec)
    mses_l.append(mse)

base_dir = os.path.dirname(os.path.dirname(__file__))
figure_dir = os.path.join(base_dir, "figures")
os.makedirs(figure_dir, exist_ok=True)
# Plotting
plt.figure(figsize=get_figsize(), dpi=DPI)
plt.plot(deltas_d, mses_d, linestyle=LINE_STYLES['EM'], label='Dipole antenna')
plt.plot(deltas_l, mses_l, linestyle=LINE_STYLES['CoFEM'], label='Loop antenna')
plt.xlabel('Undersampling $\\delta = N/D$')
plt.ylabel('Relative MSE')
plt.title('Relative MSE vs. Undersampling')
plt.grid(True)
plt.legend()
plt.gca().invert_xaxis()  # Reverse x-axis
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, "delta_vs_mse_ff.png"), dpi=DPI, bbox_inches="tight")
plt.close()

"""
R = 0.2759
r = 0.0625
f = 2.4e9
k = 2*np.pi*f/(3e8)
filename = 'dipole_farfield_spherical.txt'
N_modes = 25
D = 2*N_modes**2 + 4*N_modes

Efield_org = load_nearfield(filename, undersampling=7)
Efield_org_vec = transform_nearfield_to_vector(Efield_org)
F_org, nms_idx, _ = F_matrix_alt(Efield_org.theta, Efield_org.phi, R=R, N_modes=N_modes, k=k)

# undersampling_range = range(7, 20, 4)
deltas_d = []
mses_d = []
filename = 'dipole_nearfield_spherical.txt'

for undersampling in undersampling_range:
    # Load and process farfield with current undersampling
    Efield_nf = load_nearfield(filename, undersampling=undersampling)
    Efield_nf_vec = transform_nearfield_to_vector(Efield_nf)
    N = len(Efield_nf.phi) * len(Efield_nf.theta)
    delta = N / D
    deltas_d.append(delta)
    
    # Forward matrix and SBL reconstruction
    F, nms_idx, _ = F_matrix_alt(Efield_nf.theta, Efield_nf.phi, R=r, N_modes=N_modes, k=k)
    sbl_vec_F = SBL_Fast_Vector(Efield_nf_vec, F, max_iter=100, threshold=1e-8)
    w_est_F, basis = sbl_vec_F.fit()
    
    # Reconstruct field
    Efield_SBLF_vec = np.zeros_like(Efield_org_vec)
    for i in range(3):
        Efield_SBLF_vec[:,i] = F_org[:,:,i] @ w_est_F
    # Compute relative MSE
    mse = vec_MSE(Efield_SBLF_vec, Efield_org_vec)
    mses_d.append(mse)


deltas_l = []
mses_l = []
filename = 'loop_farfield_spherical.txt'
Efield_org = load_nearfield(filename, undersampling=7)
Efield_org_vec = transform_nearfield_to_vector(Efield_org)
F_org, nms_idx, _ = F_matrix_alt(Efield_org.theta, Efield_org.phi, R=R, N_modes=N_modes, k=k)
filename = 'loop_nearfield_spherical.txt'

for undersampling in undersampling_range:
    # Load and process farfield with current undersampling
    Efield_nf = load_nearfield(filename, undersampling=undersampling)
    Efield_nf_vec = transform_nearfield_to_vector(Efield_nf)
    N = len(Efield_nf.phi) * len(Efield_nf.theta)
    delta = N / D
    deltas_l.append(delta)
    
    # Forward matrix and SBL reconstruction
    F, nms_idx, _ = F_matrix_alt(Efield_nf.theta, Efield_nf.phi, R=r, N_modes=N_modes, k=k)
    sbl_vec_F = SBL_Fast_Vector(Efield_nf_vec, F, max_iter=100, threshold=1e-8)
    w_est_F, basis = sbl_vec_F.fit()
    
    # Reconstruct field
    Efield_SBLF_vec = np.zeros_like(Efield_org_vec)
    for i in range(3):
        Efield_SBLF_vec[:,i] = F_org[:,:,i] @ w_est_F
    # Compute relative MSE
    mse = vec_MSE(Efield_SBLF_vec, Efield_org_vec)
    mses_l.append(mse)

base_dir = os.path.dirname(os.path.dirname(__file__))
figure_dir = os.path.join(base_dir, "figures")
os.makedirs(figure_dir, exist_ok=True)
# Plotting
plt.figure(figsize=get_figsize(), dpi=DPI)
plt.plot(deltas_d, mses_d, linestyle=LINE_STYLES['EM'], label='Dipole antenna')
plt.plot(deltas_l, mses_l, linestyle=LINE_STYLES['CoFEM'], label='Loop antenna')
plt.xlabel('Undersampling $\\delta = N/D$')
plt.ylabel('Relative MSE')
plt.title('Relative MSE vs. Undersampling')
plt.grid(True)
plt.legend()
plt.gca().invert_xaxis()  # Reverse x-axis
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig(os.path.join(figure_dir, "delta_vs_mse_nffft.png"), dpi=DPI, bbox_inches="tight")
plt.close()
"""