import numpy as np
import matplotlib.pyplot as plt
from utils.plot_settings import get_figsize, LINE_STYLES, DPI
import os
from SBL_algs.SBL_Fast import SBL_Fast_Vector
from utils.Process_sim_data import load_nearfield, transform_nearfield_to_vector, transform_vector_to_nearfield
from radiation.sphere_wave import F_matrix_alt, Fmnc, calculate_spherical_coefficients, nm_expansion, inverse_nm_expansion, abF_matrix_alt
import importlib
from utils.plot import plot_coefficient_magnitudes, plot3D, plot_nm_coefficient_magnitudes

r = 50e-3
f = 2.4e9
k = 2*np.pi*f/(3e8)
filename = 'dipole_nearfield_spherical.txt'

filename = 'dipole_nearfield_spherical.txt'
Efield = load_nearfield(filename, undersampling=7)
F,nms,ThetaPhi = abF_matrix_alt(Efield.theta, Efield.phi, R=r, N_modes=20, k=k)

Efield_vec = transform_nearfield_to_vector(Efield)
F, nms_idx,_ = F_matrix_alt(Efield.theta, Efield.phi, R=r, N_modes=10, k=k)
sbl_vec = SBL_Fast_Vector(Efield_vec, F, max_iter=100, threshold=1e-8)
w_est, basis = sbl_vec.fit()
E_est = np.zeros_like(Efield_vec)
for i in range(3):
    E_est[:,i] = F[:,:,i] @ w_est
Efield_rec = transform_vector_to_nearfield(E_est, Efield.theta, Efield.phi)
plot3D(Efield_rec, Efield_rec.theta, Efield_rec.phi, r, scale_factor = 1, add_factor = 0)
weights = np.concatenate([nms_idx, w_est.reshape(-1,1)], axis=1).transpose()
plot_nm_coefficient_magnitudes(weights)
# Efield_vec = transform_nearfield_to_vector(Efield)
# ab_coeffs = nm_expansion(Efield,25, k, r)
# Efield_rec = inverse_nm_expansion(ab_coeffs, r, Efield.theta, Efield.phi, k)
# plot3D(Efield_rec, Efield_rec.theta, Efield_rec.phi, r, scale_factor = 1, add_factor = 0)
# plot3D(Efield, Efield.theta, Efield.phi, r, scale_factor = 1, add_factor = 0)