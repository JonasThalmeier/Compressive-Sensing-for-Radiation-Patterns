import numpy as np
import matplotlib.pyplot as plt
from utils.Process_sim_data import load_nearfield, transform_nearfield_to_vector, transform_vector_to_nearfield
from utils.plot_settings import get_figsize, LINE_STYLES, DPI



def plot3D(E_est_struct, theta, phi, r, scale_factor = 1, add_factor = 0):
    theta_mesh, phi_mesh = np.meshgrid(theta, phi, indexing='ij')
    Er_mag = E_est_struct.Mag_Er
    Etheta_mag = E_est_struct.Mag_Etheta
    Ephi_mag = E_est_struct.Mag_Ephi
    total_mag = np.sqrt(Er_mag**2 + Etheta_mag**2 + Ephi_mag**2)

    # Convert to Cartesian coordinates with radius = field magnitude
    X = r * np.sin(theta_mesh) * np.cos(phi_mesh)
    Y = r * np.sin(theta_mesh) * np.sin(phi_mesh)
    Z = r * np.cos(theta_mesh)
    fig = plt.figure(figsize=(15, 12))
    cmap = plt.cm.viridis
    
    # Plot Er component
    ax1 = fig.add_subplot(221, projection='3d')
    X_total = X * (add_factor + scale_factor * Er_mag)
    Y_total = Y * (add_factor + scale_factor * Er_mag)
    Z_total = Z * (add_factor + scale_factor * Er_mag)
    surf1 = ax1.plot_surface(X_total,Y_total,Z_total, facecolors=cmap(Er_mag/np.max(Er_mag)),
                            rstride=1, cstride=1, antialiased=False)
    ax1.set_title('Radial Component (Er)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # Plot Etheta component
    ax2 = fig.add_subplot(222, projection='3d')
    X_total = X * (add_factor + scale_factor * Etheta_mag)
    Y_total = Y * (add_factor + scale_factor * Etheta_mag)
    Z_total = Z * (add_factor + scale_factor * Etheta_mag)
    surf2 = ax2.plot_surface(X_total,Y_total,Z_total, facecolors=cmap(Etheta_mag/np.max(Etheta_mag)),
                            rstride=1, cstride=1, antialiased=False)
    ax2.set_title('Theta Component (Eθ)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # Plot Ephi component
    ax3 = fig.add_subplot(223, projection='3d')
    X_total = X * (add_factor + scale_factor * Ephi_mag)
    Y_total = Y * (add_factor + scale_factor * Ephi_mag)
    Z_total = Z * (add_factor + scale_factor * Ephi_mag)
    surf3 = ax3.plot_surface(X_total,Y_total,Z_total, facecolors=cmap(Ephi_mag/np.max(Ephi_mag)),
                            rstride=1, cstride=1, antialiased=False)
    ax3.set_title('Phi Component (Eφ)')
    fig.colorbar(surf3, ax=ax3, shrink=0.5)
    
    # Plot total magnitude
    ax4 = fig.add_subplot(224, projection='3d')
    X_total = X * (add_factor + scale_factor * total_mag)
    Y_total = Y * (add_factor + scale_factor * total_mag)
    Z_total = Z * (add_factor + scale_factor * total_mag)
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

def plot_coefficient_magnitudes(*coeff_arrays):
    """Visualize coefficient magnitudes with optional comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Collect magnitude ranges for consistent color scaling
    s_magnitudes = {1: [], 2: []}
    for arr in coeff_arrays:
        n, m, s, vals = arr
        magnitudes = np.abs(vals)
        non_zero = magnitudes != 0
        for st in [1, 2]:
            s_magnitudes[st].extend(magnitudes[(s == st) & non_zero])
    
    vmin_vmax = {st: (min(mags), max(mags)) if mags else (0, 1) 
                for st, mags in s_magnitudes.items()}
    
    # Plot each dataset
    for i, arr in enumerate(coeff_arrays):
        n, m, s, vals = arr
        magnitudes = np.abs(vals)
        non_zero = magnitudes != 0
        
        # Apply filtering
        n = n[non_zero]
        m = m[non_zero]
        s = s[non_zero]
        magnitudes = magnitudes[non_zero]
        
        # Offset subsequent datasets
        if i > 0:
            n = n + 0.1  # Horizontal shift for visual separation
            
        # Plot for each s category
        for st, ax in zip([1, 2], [ax1, ax2]):
            mask = s == st
            ax.scatter(n[mask], m[mask], c=magnitudes[mask], 
                      cmap='viridis', s=50, alpha=0.7,
                      vmin=vmin_vmax[st][0], vmax=vmin_vmax[st][1])
    
    # Configure plots
    for ax, st in zip([ax1, ax2], [1, 2]):
        ax.set_xlabel('n')
        ax.set_ylabel('m')
        ax.set_title(f's = {st}')
        if ax.collections:
            fig.colorbar(ax.collections[0], ax=ax, label='Coefficient Magnitude')
    
    plt.suptitle('Spherical Harmonics Coefficient Magnitudes Comparison' 
                if len(coeff_arrays) > 1 
                else 'Spherical Harmonics Coefficient Magnitudes')
    plt.tight_layout()
    plt.show()

def plot_Fcoefficient_magnitudes(*coeff_arrays):
    """Visualize coefficient magnitudes with optional comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Collect all magnitudes for global color scaling
    all_magnitudes = []
    for arr in coeff_arrays:
        n, m, s, vals = arr
        all_magnitudes.extend(np.abs(vals).flatten())
    vmin = min(all_magnitudes)
    vmax = max(all_magnitudes)
    import os
    base_dir = os.path.dirname(os.path.dirname(__file__))
    figure_dir = os.path.join(base_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    # Plot each dataset
    for i, arr in enumerate(coeff_arrays):
        n, m, s, vals = arr
        magnitudes = np.abs(vals)
        non_zero = magnitudes != 0
        
        # Apply filtering
        n = n[non_zero]
        m = m[non_zero]
        s = s[non_zero]
        magnitudes = magnitudes[non_zero]
        
        # Offset subsequent datasets
        if i > 0:
            n = n + 0.3  # Horizontal shift for visual separation
            
        # Plot for each s category
        for st, ax in zip([1, 2], [ax1, ax2]):
            mask = s == st
            ax.scatter(n[mask], m[mask], c=magnitudes[mask], 
                      cmap='plasma', s=50, alpha=0.7,
                      vmin=vmin, vmax=vmax+2)
    
    # Configure plots
    for ax, st in zip([ax1, ax2], [1, 2]):
        ax.set_xlabel('n')
        ax.set_ylabel('m')
        ax.set_title(f's = {st}')
        if ax.collections:
            fig.colorbar(ax.collections[0], ax=ax, label='Coefficient Magnitude')
    
    plt.suptitle('Spherical Harmonics Coefficient Magnitudes Comparison' 
                if len(coeff_arrays) > 1 
                else 'Spherical Harmonics Coefficient Magnitudes')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "loop_ff_weights.png"), dpi=DPI, bbox_inches="tight")

    plt.show()



def plot_abcoefficient_magnitudes(coeff_array):
    """Visualize coefficient magnitudes for ae/ao/be/bo types from a single dataset.
    Accepts either tuple of (n, m, s, vals) or numpy array of shape (N, 4)."""
    # Process input
    if isinstance(coeff_array, np.ndarray) and coeff_array.ndim == 2 and coeff_array.shape[1] >= 4:
        n, m, s, vals = coeff_array[:, 0], coeff_array[:, 1], coeff_array[:, 2], coeff_array[:, 3]
    else:
        n, m, s, vals = coeff_array  # Unpack tuple directly

    # Filter out zero values
    non_zero = np.abs(vals) != 0
    n = n[non_zero]
    m = m[non_zero]
    s = s[non_zero]
    magnitudes = np.abs(vals[non_zero])
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    s_labels = {1: 'ae', 2: 'ao', 3: 'be', 4: 'bo'}
    
    # Plot each coefficient type
    for st, ax in zip([1, 2, 3, 4], axes.flatten()):
        mask = s == st
        if not np.any(mask):
            continue  # Skip empty categories
            
        sc = ax.scatter(n[mask], m[mask], c=magnitudes[mask], 
                        cmap='viridis', s=50, alpha=0.7)
        
        ax.set_xlabel('n')
        ax.set_ylabel('m')
        ax.set_title(f'Type = {s_labels[st]}')
        fig.colorbar(sc, ax=ax, label='Coefficient Magnitude')

    plt.suptitle('Spherical Harmonics Coefficient Magnitudes')
    plt.tight_layout()
    plt.show()

def vec_MSE(vec_est,vec_true):
    mse = np.mean(np.abs(vec_est - vec_true)**2)
    norm_true = np.mean(np.abs(vec_true)**2)
    relative_mse = mse / norm_true

    print(f"Relative MSE: {relative_mse}")

    for i, component in enumerate(['r', 'theta', 'phi']):
        mse_component = np.mean(np.abs(vec_est[:, i] - vec_true[:, i])**2)
        norm_component = np.mean(np.abs(vec_true[:, i])**2)
        relative_mse_component = mse_component / norm_component
        print(f"Relative MSE ({component}): {relative_mse_component}")
    return relative_mse