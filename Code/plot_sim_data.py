import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_spherical_field_magnitude(theta, phi, Er, Etheta, Ephi):
    """Plot spherical field as a surface where radius represents field magnitude"""
    
    theta = np.squeeze(theta)
    phi = np.squeeze(phi)
    
    # Create meshgrid with correct indexing
    theta_mesh, phi_mesh = np.meshgrid(theta, phi, indexing='ij')
    
    # Calculate total field magnitude
    field_magnitude = np.sqrt(Er**2 + Etheta**2 + Ephi**2)
    
    # Normalize magnitude to be visible (optional scaling factor)
    scaling_factor = 1.0  # Adjust this if needed
    r = scaling_factor * field_magnitude

     # Calculate field magnitudes
    Er_mag = np.abs(Er)
    Etheta_mag = np.abs(Etheta)
    Ephi_mag = np.abs(Ephi)
    total_mag = np.sqrt(Er_mag**2 + Etheta_mag**2 + Ephi_mag**2)

    # Convert to Cartesian coordinates with radius = field magnitude
    X = r * np.sin(theta_mesh) * np.cos(phi_mesh)
    Y = r * np.sin(theta_mesh) * np.sin(phi_mesh)
    Z = r * np.cos(theta_mesh)
    
    # # Create figure
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    
    # # Plot the surface
    # surf = ax.plot_surface(X, Y, Z, 
    #                       rstride=1, cstride=1, 
    #                       cmap='viridis',
    #                       linewidth=0, 
    #                       antialiased=False)
    
    # # Add colorbar
    # cbar = fig.colorbar(surf, ax=ax, shrink=0.5)
    # cbar.set_label('Field Magnitude (V/m)')
    
    # # Set labels and title
    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_zlabel('Z (mm)')
    # ax.set_title('Field Magnitude Represented as Radial Distance')
    
    # # Equal aspect ratio
    # ax.set_box_aspect([1, 1, 1])
    
    # plt.tight_layout()
    # plt.show()

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

if __name__ == "__main__":
    import os
    sim_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Code/Simulations')
    txt_file = os.path.join(sim_dir, 'dipole_nearfield_spherical.txt')

    if not os.path.exists(txt_file):
        print(f"Error: File not found at {txt_file}")
    else:
        data = np.loadtxt(txt_file, skiprows=1)
        theta_all = data[:, 0]
        phi_all = data[:, 1]
        Re_Er = data[:, 2]
        Im_Er = data[:, 3]
        Re_Etheta = data[:, 4]
        Im_Etheta = data[:, 5]
        Re_Ephi = data[:, 6]
        Im_Ephi = data[:, 7]

        # Correct magnitude calculation
        Er = np.sqrt(Re_Er**2 + Im_Er**2)
        Etheta = np.sqrt(Re_Etheta**2 + Im_Etheta**2)
        Ephi = np.sqrt(Re_Ephi**2 + Im_Ephi**2)

        # Get unique theta and phi values
        unique_theta = np.unique(theta_all)
        unique_phi = np.unique(phi_all)
        n_theta = len(unique_theta)
        n_phi = len(unique_phi)

        # Check data consistency
        if len(theta_all) != n_theta * n_phi:
            raise ValueError("Data does not form a regular theta-phi grid.")

        # Reshape fields into 2D grids
        Er_grid = Er.reshape(n_theta, n_phi)
        Etheta_grid = Etheta.reshape(n_theta, n_phi)
        Ephi_grid = Ephi.reshape(n_theta, n_phi)

        plot_spherical_field_magnitude(unique_theta, unique_phi, Er_grid, Etheta_grid, Ephi_grid)