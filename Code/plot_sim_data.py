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
    
    # Convert to Cartesian coordinates with radius = field magnitude
    X = r * np.sin(theta_mesh) * np.cos(phi_mesh)
    Y = r * np.sin(theta_mesh) * np.sin(phi_mesh)
    Z = r * np.cos(theta_mesh)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, 
                          rstride=1, cstride=1, 
                          cmap='viridis',
                          linewidth=0, 
                          antialiased=False)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5)
    cbar.set_label('Field Magnitude (V/m)')
    
    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Field Magnitude Represented as Radial Distance')
    
    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
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