import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


def barPnm(n, m, x):
    return np.sqrt(0.5*(2*n+1)*np.math.factorial(n-m)/np.math.factorial(n+m)) * sp.special.lpmv(m, n, x)

def dbarPnmcosTheta_dTheta(n, m, Theta):
    """
    Calculate the derivative of the normalized associated Legendre function
    with respect to theta.
    
    Parameters:
        n (int): Degree
        m (int): Order
        Theta (float): Polar angle in radians
    
    Returns:
        float: Derivative value
    """
    eps = 1e-10  # Small value to prevent division by zero
    cos_theta = np.cos(Theta)
    sin_theta = np.sin(Theta)
    
    # Handle edge cases at poles (Theta = 0 or π)
    if np.abs(sin_theta) < eps:
        return 0.0
    
    # Safe division for the derivative
    denominator = cos_theta**2 - 1
    denominator = np.where(np.abs(denominator) < eps, np.sign(denominator)*eps, denominator)
    
    Pnm_derivate = (1/denominator) * (
        -(n+1)*cos_theta*sp.special.lpmv(m, n, cos_theta) + 
        (n-m+1)*sp.special.lpmv(m, n+1, cos_theta)
    )
    
    normalization = np.sqrt((2*n+1)*(np.math.factorial(n-m)/(2*np.math.factorial(n+m))))
    return -normalization * Pnm_derivate * sin_theta


def Fmnc(m, n, c, r, Theta, Phi, k):
    """
    Calculate the vector spherical wave functions F_mn^(c) for electromagnetic field expansion.
    
    This function implements the vector spherical harmonics using normalized associated 
    Legendre functions and spherical Bessel/Hankel functions for different radial dependencies.
    
    Parameters:
        m (int): Azimuthal mode number
        n (int): Polar mode number (n >= |m|)
        c (int): Type of spherical wave:
                1: Regular wave (spherical Bessel function j_n)
                2: Irregular wave (spherical Neumann function y_n)
                3: Outgoing wave (spherical Hankel function h_n^(1))
                4: Incoming wave (spherical Hankel function h_n^(2))
        r (float or array): Radial coordinate
        Theta (float or array): Polar angle in radians
        Phi (float or array): Azimuthal angle in radians
        k (float): Wave number
    
    Returns:
        tuple: Two arrays containing the components of the vector spherical waves:
              (F_mn^(c)_1, F_mn^(c)_2), each with shape (3,) for (r,θ,φ) components
    """
    factor = 1/np.sqrt(2*np.pi*n*(n+1))
    if m != 0:
        factor *= (-m/np.abs(m))**m
    barPnm_val = barPnm(n, np.abs(m), np.cos(Theta))
    dbarPnm_val = dbarPnmcosTheta_dTheta(n, np.abs(m), Theta)
    if c == 1:
        z = sp.special.spherical_jn(n, k*r)
        z_prime = sp.special.spherical_jn(n, k*r, derivative=True)
    elif c == 2:
        z = sp.special.spherical_yn(n, k*r)
        z_prime = sp.special.spherical_yn(n, k*r, derivative=True)
    elif c == 3:
        z = sp.special.spherical_jn(n, k*r) + 1j*sp.special.spherical_yn(n, k*r)
        z_prime = sp.special.spherical_jn(n, k*r, derivative=True) + 1j*sp.special.spherical_yn(n, k*r, derivative=True)
    else:
        z = sp.special.spherical_jn(n, k*r) - 1j*sp.special.spherical_yn(n, k*r)
        z_prime = sp.special.spherical_jn(n, k*r, derivative=True) - 1j*sp.special.spherical_yn(n, k*r, derivative=True)
    sinTheta = np.sin(Theta)
    if sinTheta == 0:
        sinTheta = 1e-3
    Theta1 = z * 1j * m * barPnm_val * np.exp(1j*m*Phi) / sinTheta
    Phi1 = -z * dbarPnm_val * np.exp(1j*m*Phi)
    r2 = (n*(n+1)/(k*r))*z*barPnm_val*np.exp(1j*m*Phi)
    Theta2 = 1/(k*r) * (z + k*r*z_prime) * dbarPnm_val * np.exp(1j*m*Phi)
    Phi2 = 1/(k*r) * (z + k*r*z_prime) * 1j * m * barPnm_val * np.exp(1j*m*Phi) / sinTheta
    if np.any(np.isnan(np.array([Theta1, Phi1, r2, Theta2, Phi2]))):
        print(f"NaN detected in Fmnc for m={m}, n={n}, c={c}, r={r}, Theta={Theta}, Phi={Phi}")
        print(f"Values: {Theta1}, {Phi1}, {r2}, {Theta2}, {Phi2}")
    return factor*np.array([0,Theta1,Phi1]), factor*np.array([r2,Theta2,Phi2])

def F_matrix(R=1, Theta_steps=18, Phi_steps=36, N_modes=50, c = 3, k=1):
    N = Theta_steps * Phi_steps
    D = 2*N_modes**2+4*N_modes
    # Step 1: Build matrix of spherical wave coefficients
    F = np.zeros((N, D, 3), dtype=np.complex_)
    nms_idx = np.zeros((D,3))
    ThetaPhi_idx = np.zeros((N,2))
    for Theta_idx, Theta in enumerate(np.linspace(1e-3, np.pi, Theta_steps)):
        for Phi_idx, Phi in enumerate(np.linspace(1e-3, 2*np.pi, Phi_steps)):
            ThetaPhi_idx[Theta_idx*Phi_steps + Phi_idx, :] = [Theta, Phi]
            for n in np.arange(1, N_modes+1):
                for m in np.arange(-n, n+1):
                    idx1 = (int(Theta_idx*Phi_steps + Phi_idx), int(n**2+n-1+m))
                    idx2 = (int(Theta_idx*Phi_steps + Phi_idx), int(n**2+n-1+m+D/2))
                    F[idx1[0], idx1[1], :], F[idx2[0], idx2[1], :] = Fmnc(m, n, c, R, Theta, Phi, k)
    for n in np.arange(1, N_modes+1):
                for m in np.arange(-n, n+1):
                    nms_idx[int(n**2+n-1+m),:] = [n, m, 1]
                    nms_idx[int(n**2+n-1+m+D/2),:] = [n, m, 2]
    # F = F/(np.max(np.abs(F), axis=(0,2))[np.newaxis, :, np.newaxis])  # Normalize F to avoid numerical issues
    return F, nms_idx, ThetaPhi_idx

def F_matrix_alt(theta, phi, R=1, N_modes=50, c=3, k=1):
    """Build spherical wave coefficient matrix for given theta/phi arrays"""
    N = len(theta) * len(phi)
    D = 2*N_modes**2 + 4*N_modes
    Phi_steps = len(phi)
    F = np.zeros((N, D, 3), dtype=np.complex_)
    nms_idx = np.zeros((D, 3))
    ThetaPhi_idx = np.zeros((N, 2))
    for Theta_idx, Theta in enumerate(theta):
        for Phi_idx, Phi in enumerate(phi):
            ThetaPhi_idx[int(Theta_idx*Phi_steps + Phi_idx), :] = [Theta, Phi]
            for n in range(1, N_modes+1):
                for m in range(-n, n+1):
                    idx1 = (int(Theta_idx*Phi_steps + Phi_idx), n**2 + n - 1 + m)
                    idx2 = (int(Theta_idx*Phi_steps + Phi_idx), n**2 + n - 1 + m + D//2)
                    F[idx1[0], idx1[1], :], F[idx2[0], idx2[1], :] = Fmnc(m, n, c, R, Theta, Phi, k)
    for n in range(1, N_modes+1):
        for m in range(-n, n+1):
            nms_idx[n**2 + n - 1 + m] = [n, m, 1]
            nms_idx[n**2 + n - 1 + m + D//2] = [n, m, 2]
    
    return F, nms_idx, ThetaPhi_idx

def calculate_spherical_coefficients(Efield, max_n, k, r, c=3):
    """
    Calculate spherical harmonics coefficients for E-field components.
    
    Args:
        E_real: Tuple of (Er_real, Eth_real, Ephi_real) matrices
        E_imag: Tuple of (Er_imag, Eth_imag, Ephi_imag) matrices
        theta: 1D array of theta values
        phi: 1D array of phi values
        max_n: Maximum n value for harmonic expansion
    
    Returns:
        Dictionary of complex coefficients {(s, m, n): coefficient}
    """
    # Convert E-field to complex representation
    Er = Efield.Re_Er + 1j*Efield.Im_Er
    Eth = Efield.Re_Etheta + 1j*Efield.Im_Etheta
    Ephi = Efield.Re_Ephi + 1j*Efield.Im_Ephi
    theta = Efield.theta
    phi = Efield.phi

    # Calculate differential elements
    dtheta = np.abs(theta[1] - theta[0])
    dphi = np.abs(phi[1] - phi[0])
    
    coefficients1 = []
    coefficients2 = []
    
    # Loop through all possible modes
    for n in range(1, max_n+1):
        for m in range(-n, n+1):
            integral1 = 0j
            integral2 = 0j
            
            # Perform surface integration
            for i, th in enumerate(theta):
                for j, ph in enumerate(phi):
                    # Get vector spherical harmonic
                    F1,F2 = Fmnc(m, n, c, r, th, ph, k)
                    # Dot product with E-field conjugate
                    dot_product1 = (Er[i,j]*np.conj(F1[0]) + 
                                    Eth[i,j]*np.conj(F1[1]) + 
                                    Ephi[i,j]*np.conj(F1[2]))
                    dot_product2 = (Er[i,j]*np.conj(F2[0]) + 
                                    Eth[i,j]*np.conj(F2[1]) + 
                                    Ephi[i,j]*np.conj(F2[2]))
                    
                    # Jacobian factor for spherical coordinates
                    integral1 += dot_product1 * np.sin(th) * dtheta * dphi
                    integral2 += dot_product2 * np.sin(th) * dtheta * dphi
            

            coefficients1.append([n,m,1,integral1])
            coefficients2.append([n,m,2,integral2])
                
    return np.concatenate([coefficients1,coefficients2], axis=0).transpose()