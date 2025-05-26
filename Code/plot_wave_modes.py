import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.synthetic_data import generate_synthetic_data
from EM_algs.EM_wo_SURE import SBL_EM
from EM_algs.SBL_Fast import SBL_Fast
from utils.plot_settings import get_figsize, LINE_STYLES, DPI
from utils.synthetic_data_vector import generate_synthetic_data_vec
import os
from EM_algs.SBL_Fast_Vector import SBL_Fast_Vector
from utils.synthetic_spherical_wave import generate_spherical_wave

# Generate data with finer grid for better visualization
Theta_steps, Phi_steps = 45, 90
N_modes = 5
t, F, w, e,_,_ = generate_spherical_wave(
    Theta_steps=Theta_steps,
    Phi_steps=Phi_steps,
    N_modes=N_modes,
    sigma=0,
    R = 0.01
)

# Create meshgrid for plotting
theta = np.linspace(1e-3, np.pi, Theta_steps)
phi = np.linspace(1e-3, 2*np.pi, Phi_steps)
THETA, PHI = np.meshgrid(theta, phi, indexing='ij')  # Add indexing='ij'

# Convert spherical to Cartesian coordinates for plotting
X = np.sin(THETA) * np.cos(PHI)
Y = np.sin(THETA) * np.sin(PHI)
Z = np.cos(THETA)

# Select a specific mode (e.g., first mode)
n = 1
m = 0
s = 2
mode_idx = n**2+n-1+m+(s-1)*N_modes*(N_modes+2)  # Choose a specific index in the range [0, D-1]
w_single = np.zeros(F.shape[1])
w_single[mode_idx] = 1.0

# Calculate field components for single mode
field = np.zeros_like(t)
for l in range(3):
    field[:, l] = F[:, :, l] @ w_single

# Reshape field components to match grid
field = field.reshape(Theta_steps, Phi_steps, 3)

# Create subplots
fig = plt.figure(figsize=(15, 10))

# Plot r component
ax1 = fig.add_subplot(221, projection='3d')
scale_factor = 1  # Adjust this to control the size of distortions
add_factor = 0
field_r = np.abs(field[:,:,0])
X_r = X * (add_factor + scale_factor * field_r/np.max(field_r))
Y_r = Y * (add_factor + scale_factor * field_r/np.max(field_r))
Z_r = Z * (add_factor + scale_factor * field_r/np.max(field_r))
p1 = ax1.plot_surface(X_r, Y_r, Z_r, facecolors=plt.cm.viridis(field_r/np.max(field_r)))
ax1.set_title('Radial Component')
fig.colorbar(p1, ax=ax1)

# Plot theta component
ax2 = fig.add_subplot(222, projection='3d')
field_theta = np.abs(field[:,:,1])
X_theta = X * (add_factor + scale_factor * field_theta/np.max(field_theta))
Y_theta = Y * (add_factor + scale_factor * field_theta/np.max(field_theta))
Z_theta = Z * (add_factor + scale_factor * field_theta/np.max(field_theta))
p2 = ax2.plot_surface(X_theta, Y_theta, Z_theta, facecolors=plt.cm.viridis(field_theta/np.max(field_theta)))
ax2.set_title('Theta Component')
fig.colorbar(p2, ax=ax2)

# Plot phi component
ax3 = fig.add_subplot(223, projection='3d')
field_phi = np.abs(field[:,:,2])
X_phi = X * (add_factor + scale_factor * field_phi/np.max(field_phi))
Y_phi = Y * (add_factor + scale_factor * field_phi/np.max(field_phi))
Z_phi = Z * (add_factor + scale_factor * field_phi/np.max(field_phi))
p3 = ax3.plot_surface(X_phi, Y_phi, Z_phi, facecolors=plt.cm.viridis(field_phi/np.max(field_phi)))
ax3.set_title('Phi Component')
fig.colorbar(p3, ax=ax3)

# Plot magnitude of total field
ax4 = fig.add_subplot(224, projection='3d')
total_field = np.sqrt(np.sum(np.abs(field)**2, axis=2))
X_total = X * (add_factor + scale_factor * total_field/np.max(total_field))
Y_total = Y * (add_factor + scale_factor * total_field/np.max(total_field))
Z_total = Z * (add_factor + scale_factor * total_field/np.max(total_field))
p4 = ax4.plot_surface(X_total, Y_total, Z_total, facecolors=plt.cm.viridis(total_field/np.max(total_field)))
ax4.set_title('Total Field Magnitude')
fig.colorbar(p4, ax=ax4)

# Set equal aspect ratios and limits for all subplots
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_box_aspect([1,1,1])
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])

plt.tight_layout()
plt.show()

