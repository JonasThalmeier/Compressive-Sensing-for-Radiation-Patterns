import numpy as np
import matplotlib.pyplot as plt
from utils.plot_settings import get_figsize, LINE_STYLES, DPI
import os
from SBL_algs.SBL_Fast import SBL_Fast_Vector
from utils.Process_sim_data import load_nearfield, transform_nearfield_to_vector, transform_vector_to_nearfield
from radiation.sphere_wave import F_matrix_alt, Fmnc, nm_expansion, inverse_nm_expansion, abF_matrix_alt
import importlib
from utils.plot import vec_MSE

r = 50e-3
f = 2.4e9
k = 2*np.pi*f/(3e8)
filename = 'dipole_nearfield_spherical.txt'

filename = 'dipole_nearfield_spherical.txt'
num_measurements = 181 * 360 #number of theta and phi steps
min_us = 10
undersampling = np.linspace(min_us,5*min_us,10).astype(int)
D = np.linspace(0.01*num_measurements/(min_us**2), 0.5*num_measurements/(min_us**2), 3, endpoint=True)
N = np.ceil(-1+np.sqrt(16+8*D)/4).astype(int)
NF_mse = np.zeros((len(N),len(undersampling)))
Efield_org = load_nearfield(filename, undersampling=min_us)
Efield_org_vec = transform_nearfield_to_vector(Efield_org)
for n_idx, n in enumerate(N):
    F_re, nms_idx,_ = F_matrix_alt(Efield_org.theta, Efield_org.phi, R=r, N_modes=n, k=k)
    for us_idx, us in enumerate(undersampling):
        print(f"Processing N={int(n)} ({n_idx+1}/{len(N)}), Undersampling={us:.1f} ({us_idx+1}/{len(undersampling)})")  # <-- Added status print
        Efield = load_nearfield(filename, undersampling=us)
        Efield_vec = transform_nearfield_to_vector(Efield)
        F, nms_idx,_ = F_matrix_alt(Efield.theta, Efield.phi, R=r, N_modes=n, k=k)
        sbl_vec_F = SBL_Fast_Vector(Efield_vec, F, max_iter=100, threshold=1e-8)
        w_est_F, basis = sbl_vec_F.fit()
        Efield_SBLF_vec = np.zeros_like(Efield_org_vec)
        for i in range(3):
            Efield_SBLF_vec[:,i] = F_re[:,:,i] @ w_est_F
        NF_mse[n_idx,us_idx] = vec_MSE(Efield_SBLF_vec,Efield_org_vec)


# Plotting
plt.figure(figsize=(10, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(N)))  # Using tab10 colormap
for n_idx, n in enumerate(N):
    plt.plot((undersampling/min_us)**2, NF_mse[n_idx, :],
             label=f'N={int(n)}',
             color=colors[n_idx])
plt.xlabel('Undersampling²')
plt.ylabel('MSE')
plt.title('NF Reconstruction Error vs Undersampling²')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()