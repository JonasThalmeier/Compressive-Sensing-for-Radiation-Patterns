import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from collections import namedtuple
import os


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

def load_nearfield(filename, undersampling=1):
    sim_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Simulations')
    txt_file = os.path.join(sim_dir, filename)
    data = np.loadtxt(txt_file, skiprows=1)
    theta_all = data[:, 0]
    phi_all = data[:, 1]
    Re_Er = data[:, 2]
    Im_Er = data[:, 3]
    Re_Etheta = data[:, 4]
    Im_Etheta = data[:, 5]
    Re_Ephi = data[:, 6]
    Im_Ephi = data[:, 7]

    # Get unique theta and phi values with undersampling
    unique_theta = np.unique(theta_all)[::undersampling]
    unique_phi = np.unique(phi_all)[::undersampling]
    
    # Create mask for undersampled points
    mask = np.isin(theta_all, unique_theta) & np.isin(phi_all, unique_phi)
    
    # Apply mask to all fields
    theta_all = theta_all[mask]
    phi_all = phi_all[mask]
    Re_Er = Re_Er[mask]
    Im_Er = Im_Er[mask]
    Re_Etheta = Re_Etheta[mask]
    Im_Etheta = Im_Etheta[mask]
    Re_Ephi = Re_Ephi[mask]
    Im_Ephi = Im_Ephi[mask]

    # Update dimensions
    n_theta = len(unique_theta)
    n_phi = len(unique_phi)

    # Validate grid completeness
    if len(theta_all) != n_theta * n_phi:
        raise ValueError(f"Undersampling resulted in incomplete grid. Expected {n_theta * n_phi} points, got {len(theta_all)}")

    # Correct magnitude calculation
    Er = np.sqrt(Re_Er**2 + Im_Er**2)
    Etheta = np.sqrt(Re_Etheta**2 + Im_Etheta**2)
    Ephi = np.sqrt(Re_Ephi**2 + Im_Ephi**2)

    Efield = namedtuple('Efield', ['theta', 'phi', 'Re_Er', 'Re_Etheta', 'Re_Ephi', 
                              'Im_Er', 'Im_Etheta', 'Im_Ephi',
                              'Mag_Er', 'Mag_Etheta', 'Mag_Ephi'])
    
    return Efield(
        theta=unique_theta,
        phi=unique_phi,
        Re_Er=Re_Er.reshape(n_theta, n_phi),
        Re_Etheta=Re_Etheta.reshape(n_theta, n_phi),
        Re_Ephi=Re_Ephi.reshape(n_theta, n_phi),
        Im_Er=Im_Er.reshape(n_theta, n_phi),
        Im_Etheta=Im_Etheta.reshape(n_theta, n_phi),
        Im_Ephi=Im_Ephi.reshape(n_theta, n_phi),
        Mag_Er=Er.reshape(n_theta, n_phi),
        Mag_Etheta=Etheta.reshape(n_theta, n_phi),
        Mag_Ephi=Ephi.reshape(n_theta, n_phi)
    )

def transform_nearfield_to_vector(Efield):
    """Transform Efield data into vector format (N, 1, 3) with [Er, Etheta, Ephi] order"""
    N = len(Efield.theta) * len(Efield.phi)
    E_vector = np.zeros((N, 3), dtype=np.complex_)
    
    # Combine real and imaginary parts into complex numbers
    Er = Efield.Re_Er + 1j * Efield.Im_Er
    Etheta = Efield.Re_Etheta + 1j * Efield.Im_Etheta
    Ephi = Efield.Re_Ephi + 1j * Efield.Im_Ephi
    
    # Flatten and reshape into desired format
    for idx, (i, j) in enumerate(np.ndindex(Er.shape)):
        E_vector[idx, 0] = Er[i, j]    # Er component
        E_vector[idx, 1] = Etheta[i, j]  # Etheta component
        E_vector[idx, 2] = Ephi[i, j]    # Ephi component
    
    return E_vector

def transform_vector_to_nearfield(E_vector, theta, phi):
    theta_len, phi_len = len(theta), len(phi)
    """Convert vector format (N,3) back to Efield components with shape (theta_len, phi_len)"""
    # Reshape vector components
    Er = E_vector[:, 0].reshape(theta_len, phi_len)
    Etheta = E_vector[:, 1].reshape(theta_len, phi_len)
    Ephi = E_vector[:, 2].reshape(theta_len, phi_len)
    
    # Create Efield-like structure (adjust attributes based on your actual Efield class)
    class Efield:
        pass
    
    Efield_out = Efield()
    Efield_out.Re_Er = np.real(Er)
    Efield_out.Im_Er = np.imag(Er)
    Efield_out.Re_Etheta = np.real(Etheta)
    Efield_out.Im_Etheta = np.imag(Etheta)
    Efield_out.Re_Ephi = np.real(Ephi)
    Efield_out.Im_Ephi = np.imag(Ephi)
    Efield_out.theta = theta  # Adjust if needed
    Efield_out.phi = phi
    Efield_out.Mag_Er = np.abs(Er)
    Efield_out.Mag_Etheta = np.abs(Etheta)
    Efield_out.Mag_Ephi = np.abs(Ephi)
    return Efield_out

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