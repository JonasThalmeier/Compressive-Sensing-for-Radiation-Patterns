import numpy as np
import matplotlib.pyplot as plt
from utils.plot_settings import get_figsize, LINE_STYLES, DPI
import os
from SBL_algs.SBL_Fast import SBL_Fast_Vector
from utils.Process_sim_data import load_nearfield, transform_nearfield_to_vector, transform_vector_to_nearfield
from radiation.sphere_wave import F_matrix_alt, Fmnc, calculate_spherical_coefficients
from utils.plot import plot_coefficient_magnitudes




if __name__ == "__main__":
    r = 50e-3
    f = 2.4e9
    k = 2*np.pi*f/(3e8)
    #NaN detected in Fmnc for m=2, n=5, c=3, r=50, Theta=0.0, Phi=6.26573
    # _,_ = Fmnc(2, 5, 3, 5, 0, 6.26573, k)

    filename = 'dipole_nearfield_spherical.txt'
    Efield = load_nearfield(filename, undersampling=10)
    
    Efield_vec = transform_nearfield_to_vector(Efield)
    F, nms_idx,_ = F_matrix_alt(Efield.theta, Efield.phi, R=r, N_modes=15, k=k)
    sbl_vec = SBL_Fast_Vector(Efield_vec, F, max_iter=100, threshold=1e-8)
    w_est, basis = sbl_vec.fit()
    print(f"Active basis (n,m,s): {nms_idx[basis,:]}")
    Efield_full = load_nearfield(filename, undersampling=10)
    coeffs = calculate_spherical_coefficients(Efield_full, 15, k, r, c=3)
    weights = np.concatenate([nms_idx, w_est.reshape(-1,1)], axis=1).transpose()
    plot_coefficient_magnitudes(coeffs, weights)


    E_est = np.zeros_like(Efield_vec)
    for i in range(3):
        E_est[:,i] = F[:,:,i] @ w_est
    mse = np.mean(np.abs(E_est - Efield_vec)**2)
    norm_true = np.mean(np.abs(Efield_vec)**2)
    relative_mse = mse / norm_true
    
    print(f"Relative MSE: {relative_mse}")

    for i, component in enumerate(['r', 'theta', 'phi']):
        mse_component = np.mean(np.abs(E_est[:, i] - Efield_vec[:, i])**2)
        norm_component = np.mean(np.abs(Efield_vec[:, i])**2)
        relative_mse_component = mse_component / norm_component
        print(f"Relative MSE ({component}): {relative_mse_component}")

  

    theta_mesh, phi_mesh = np.meshgrid(Efield.theta, Efield.phi, indexing='ij')
    E_est_struct = transform_vector_to_nearfield(E_est, Efield.theta, Efield.phi)
        # Calculate field magnitudes
    Er_mag = E_est_struct.Mag_Er
    Etheta_mag = E_est_struct.Mag_Etheta
    Ephi_mag = E_est_struct.Mag_Ephi
    total_mag = np.sqrt(Er_mag**2 + Etheta_mag**2 + Ephi_mag**2)

    # Convert to Cartesian coordinates with radius = field magnitude
    X = r * np.sin(theta_mesh) * np.cos(phi_mesh)
    Y = r * np.sin(theta_mesh) * np.sin(phi_mesh)
    Z = r * np.cos(theta_mesh)
    


    scale_factor = 1
    add_factor = 0
    fig = plt.figure(figsize=(15, 12))
    cmap = plt.cm.viridis
    
    # Plot Er component
    ax1 = fig.add_subplot(221, projection='3d')
    X_total = X * (add_factor + scale_factor * Er_mag/np.max(Er_mag))
    Y_total = Y * (add_factor + scale_factor * Er_mag/np.max(Er_mag))
    Z_total = Z * (add_factor + scale_factor * Er_mag/np.max(Er_mag))
    surf1 = ax1.plot_surface(X_total,Y_total,Z_total, facecolors=cmap(Er_mag/np.max(Er_mag)),
                            rstride=1, cstride=1, antialiased=False)
    ax1.set_title('Radial Component (Er)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # Plot Etheta component
    ax2 = fig.add_subplot(222, projection='3d')
    X_total = X * (add_factor + scale_factor * Etheta_mag/np.max(Etheta_mag))
    Y_total = Y * (add_factor + scale_factor * Etheta_mag/np.max(Etheta_mag))
    Z_total = Z * (add_factor + scale_factor * Etheta_mag/np.max(Etheta_mag))
    surf2 = ax2.plot_surface(X_total,Y_total,Z_total, facecolors=cmap(Etheta_mag/np.max(Etheta_mag)),
                            rstride=1, cstride=1, antialiased=False)
    ax2.set_title('Theta Component (Eθ)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # Plot Ephi component
    ax3 = fig.add_subplot(223, projection='3d')
    X_total = X * (add_factor + scale_factor * Ephi_mag/np.max(Ephi_mag))
    Y_total = Y * (add_factor + scale_factor * Ephi_mag/np.max(Ephi_mag))
    Z_total = Z * (add_factor + scale_factor * Ephi_mag/np.max(Ephi_mag))
    surf3 = ax3.plot_surface(X_total,Y_total,Z_total, facecolors=cmap(Ephi_mag/np.max(Ephi_mag)),
                            rstride=1, cstride=1, antialiased=False)
    ax3.set_title('Phi Component (Eφ)')
    fig.colorbar(surf3, ax=ax3, shrink=0.5)
    
    # Plot total magnitude
    ax4 = fig.add_subplot(224, projection='3d')
    X_total = X * (add_factor + scale_factor * total_mag/np.max(total_mag))
    Y_total = Y * (add_factor + scale_factor * total_mag/np.max(total_mag))
    Z_total = Z * (add_factor + scale_factor * total_mag/np.max(total_mag))
    surf4 = ax4.plot_surface(X_total,Y_total,Z_total, facecolors=cmap(total_mag/np.max(total_mag)),
                            rstride=1, cstride=1, antialiased=False)
    ax4.set_title('Total Field Magnitude')
    fig.colorbar(surf4, ax=ax4, shrink=0.5)
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_box_aspect([1,1,1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()