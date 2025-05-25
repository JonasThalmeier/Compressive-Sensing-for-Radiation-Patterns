import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata

def cartesian_to_spherical(x, y, z, Ex, Ey, Ez):
    """Convert Cartesian coordinates and fields to spherical coordinates."""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    
    # Convert fields to spherical
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    Er = Ex * sin_theta * cos_phi + Ey * sin_theta * sin_phi + Ez * cos_theta
    Etheta = Ex * cos_theta * cos_phi + Ey * cos_theta * sin_phi - Ez * sin_theta
    Ephi = -Ex * sin_phi + Ey * cos_phi
    
    return r, theta, phi, Er, Etheta, Ephi

def get_field_at_radius(data_file, target_r, method='cubic'):
    # Load data
    data = np.loadtxt(data_file, skiprows=2)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    
    # Extract real and imaginary parts of the field components
    Ex = data[:, 3] + 1j * data[:, 4]
    Ey = data[:, 5] + 1j * data[:, 6]
    Ez = data[:, 7] + 1j * data[:, 8]
    
    # Convert to spherical coordinates for all points
    r, theta, phi, Er, Etheta, Ephi = cartesian_to_spherical(x, y, z, Ex, Ey, Ez)
    
    # Find points near the target radius
    tolerance = 3 #0.1  # mm tolerance for considering points at the same radius
    mask = np.abs(r - target_r) < tolerance
    
    if np.sum(mask) == 0:
        raise ValueError(f"No data points found near radius {target_r} mm")
    
    # Create grid of theta and phi for interpolation
    theta_grid = np.linspace(0, np.pi, 180)
    phi_grid = np.linspace(-np.pi, np.pi, 360)
    
    # Create meshgrid for interpolation
    theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    
    # Interpolate each spherical component
    points = np.column_stack((theta[mask], phi[mask]))
    
    # Interpolate real and imaginary parts separately
    Er_real = griddata(points, Er[mask].real, (theta_mesh, phi_mesh), method=method)
    Er_imag = griddata(points, Er[mask].imag, (theta_mesh, phi_mesh), method=method)
    Etheta_real = griddata(points, Etheta[mask].real, (theta_mesh, phi_mesh), method=method)
    Etheta_imag = griddata(points, Etheta[mask].imag, (theta_mesh, phi_mesh), method=method)
    Ephi_real = griddata(points, Ephi[mask].real, (theta_mesh, phi_mesh), method=method)
    Ephi_imag = griddata(points, Ephi[mask].imag, (theta_mesh, phi_mesh), method=method)
    
    # Combine real and imaginary parts
    Er_interp = Er_real + 1j * Er_imag
    Etheta_interp = Etheta_real + 1j * Etheta_imag
    Ephi_interp = Ephi_real + 1j * Ephi_imag
    
    return theta_grid, phi_grid, Er_interp, Etheta_interp, Ephi_interp

# Example usage
if __name__ == "__main__":
    import os
    data_file = os.path.join(os.path.dirname(__file__), 'dipol1.txt')  # Robust path    
    target_r = 25.0  # Target radius in mm
    
    try:
        theta, phi, Er, Etheta, Ephi = get_field_at_radius(data_file, target_r, method='nearest')
        output_dir = os.path.dirname(data_file)
        output_file = os.path.join(output_dir, f'fields_r_{target_r:.1f}mm.npz')
        np.savez(output_file, theta=theta, phi=phi, Er=Er, Etheta=Etheta, Ephi=Ephi)
        theta_idx = np.argmin(np.abs(theta - np.pi/2))
        phi_idx = np.argmin(np.abs(phi - 0))
        
        print(f"Field at r={target_r} mm, theta={theta[theta_idx]:.2f}, phi={phi[phi_idx]:.2f}:")
        print(f"Er: {Er[theta_idx, phi_idx]} V/m")
        print(f"Etheta: {Etheta[theta_idx, phi_idx]} V/m")
        print(f"Ephi: {Ephi[theta_idx, phi_idx]} V/m")
        
    except ValueError as e:
        print(e)