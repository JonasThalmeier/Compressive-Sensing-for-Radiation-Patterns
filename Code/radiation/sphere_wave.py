import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from collections import namedtuple



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
    normalization = np.sqrt((2*n+1)*(np.math.factorial(n-m)/(2*np.math.factorial(n+m))))
    return normalization * dPnmcosTheta_dTheta(n, m, Theta)

def dPnmcosTheta_dTheta(n, m, Theta):
    """
    Calculate the derivative of the associated Legendre function
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
        return 0
        # if m == 1 and n >= 1:
        #     return -0.5 * n * (n+1)  # Example correction for m=1 case
        # # Add other special cases as needed for your application
    
    # Safe division for the derivative
    denominator = cos_theta**2-1
    denominator = np.where(np.abs(denominator) < eps, np.sign(denominator)*eps, denominator)
    
    Pnm_derivate = (1/denominator) * (
        -(n+1)*cos_theta*sp.special.lpmv(m, n, cos_theta) + 
        (n-m+1)*sp.special.lpmv(m, n+1, cos_theta)
    )
    
    return  -Pnm_derivate * sin_theta

def z_n(n,k,r,c):
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
    return z,z_prime

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
    z,z_prime = z_n(n,k,r,c)
    sinTheta = np.sin(Theta)
    if sinTheta <= 1e-3:
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

def nm_eo(m, n, r, Theta, Phi, k, c=3):
    z,z_prime = z_n(n,k,r,c)
    P = sp.special.lpmv(m, n, np.cos(Theta))
    dP = dPnmcosTheta_dTheta(n, m, Theta)
    sinTheta = np.sin(Theta)
    if sinTheta <= 1e-3:
        sinTheta = 1e-3
    sinmPhi = np.sin(m*Phi)
    cosmPhi = np.cos(m*Phi)
    me = np.array([0,-z*m*P*sinmPhi/sinTheta, -z*dP*cosmPhi])
    mo = np.array([0,z*m*P*cosmPhi/sinTheta, -z*dP*sinmPhi])
    ne = np.array([n*(n+1)*z*P*sinmPhi/(k*r), (z+k*r*z_prime)*dP*sinmPhi, (z+k*r*z_prime)*m*P*cosmPhi/(k*r*sinTheta)]) 
    no = np.array([n*(n+1)*z*P*cosmPhi/(k*r), (z+k*r*z_prime)*dP*cosmPhi, -(z+k*r*z_prime)*m*P*sinmPhi/(k*r*sinTheta)])
    return me,mo,ne,no

def nm_expansion(Efield, max_n, k, r, c=3):
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
    
    coefficients_ae = []
    coefficients_ao = []
    coefficients_be = []
    coefficients_bo = []
    
    # Loop through all possible modes
    for n in range(1, max_n+1):
        for m in range(-n, n+1):
            integralae = 0j
            integralao = 0j
            integralbe = 0j
            integralbo = 0j
            
            # Perform surface integration
            for i, th in enumerate(theta):
                for j, ph in enumerate(phi):
                    # Get vector spherical harmonic
                    me,mo,ne,no = nm_eo(m, n, r, th, ph, k)
                    # Dot product with E-field conjugate
                    dot_product_ae = (Eth[i,j]*np.conj(me[1]) + 
                                    Ephi[i,j]*np.conj(me[2]))
                    dot_product_ao =  (Eth[i,j]*np.conj(mo[1]) + 
                                    Ephi[i,j]*np.conj(mo[2]))
                    dot_product_be =  (Er[i,j]*np.conj(ne[0])+
                                       Eth[i,j]*np.conj(ne[1]) + 
                                    Ephi[i,j]*np.conj(ne[2]))
                    dot_product_bo =  (Er[i,j]*np.conj(no[0])+
                                       Eth[i,j]*np.conj(no[1]) + 
                                    Ephi[i,j]*np.conj(no[2]))

                    # dot_product_be =  (Eth[i,j]*ne[1] + 
                    #                 Ephi[i,j]*ne[2])
                    # dot_product_bo =  (Eth[i,j]*no[1] + 
                    #                 Ephi[i,j]*no[2])
                    
                    # Jacobian factor for spherical coordinates
                    integralae += -dot_product_ae * np.sin(th) * dtheta * dphi
                    integralao += -dot_product_ao * np.sin(th) * dtheta * dphi
                    integralbe += -dot_product_be * np.sin(th) * dtheta * dphi
                    integralbo += -dot_product_bo * np.sin(th) * dtheta * dphi
            
            z,z_prime = z_n(n,k,r,c)
            factora = (2*n+1)*np.math.factorial(n-m)/(z**2*2*np.pi*(n+1)*np.math.factorial(n+m))
            factorb = (2*n+1)*np.math.factorial(n-m)/((z/(k*r)+z_prime)**2*2*np.pi*n*(n+1)*np.math.factorial(n+m))

            coefficients_ae.append([factora*integralae])
            coefficients_ao.append([factora*integralao])
            coefficients_be.append([factorb*integralbe])
            coefficients_bo.append([factorb*integralbo])
    coefficients_ae = np.array([coefficients_ae])
    coefficients_ao = np.array([coefficients_ao])
    coefficients_be = np.array([coefficients_be])
    coefficients_bo = np.array([coefficients_bo])
    return np.concatenate([coefficients_ae.reshape(-1,1),coefficients_ao.reshape(-1,1),coefficients_be.reshape(-1,1),coefficients_bo.reshape(-1,1)], axis=0)

def inverse_nm_expansion(coefficients, r, theta, phi, k):
    """
    Reconstruct E-field from spherical harmonics coefficients.
    Args:
        coefficients: Array of [n, m, ae, ao, be, bo] coefficients
        r: Radial distance (scalar)
        theta: 1D array of theta values
        phi: 1D array of phi values
        k: Wavenumber
    Returns:
        Tuple of (Er, Eth, Ephi) complex field components
    """
    # Initialize field arrays
    Er = np.zeros((len(theta), len(phi)), dtype=np.complex128)
    Eth = np.zeros_like(Er)
    Ephi = np.zeros_like(Er)
    
    # Process each coefficient set
    for n, m, ae, ao, be, bo in coefficients:
        # Calculate harmonics for all theta, phi points
        n = int(n)
        m = int(m)
        for i, th in enumerate(theta):
            for j, ph in enumerate(phi):
                me, mo, ne, no = nm_eo(m, n, r, th, ph, k)
                Er[i,j] += ae*me[0] + ao*mo[0] + be*ne[0] + bo*no[0]
                Eth[i,j] += ae*me[1] + ao*mo[1] + be*ne[1] + bo*no[1]
                Ephi[i,j] += ae*me[2] + ao*mo[2] + be*ne[2] + bo*no[2]
    
    # Calculate all required components
    Re_Er = np.real(-Er)
    Re_Etheta = np.real(-Eth)
    Re_Ephi = np.real(-Ephi)
    Im_Er = np.imag(-Er)
    Im_Etheta = np.imag(-Eth)
    Im_Ephi = np.imag(-Ephi)
    Mag_Er = np.abs(Er)
    Mag_Etheta = np.abs(Eth)
    Mag_Ephi = np.abs(Ephi)

    Efield = namedtuple('Efield', ['theta', 'phi', 'Re_Er', 'Re_Etheta', 'Re_Ephi', 
                              'Im_Er', 'Im_Etheta', 'Im_Ephi',
                              'Mag_Er', 'Mag_Etheta', 'Mag_Ephi'])
    return Efield(
        theta=theta,
        phi=phi,
        Re_Er=Re_Er,
        Re_Etheta=Re_Etheta,
        Re_Ephi=Re_Ephi,
        Im_Er=Im_Er,
        Im_Etheta=Im_Etheta,
        Im_Ephi=Im_Ephi,
        Mag_Er=Mag_Er,
        Mag_Etheta=Mag_Etheta,
        Mag_Ephi=Mag_Ephi
    )

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

def abF_matrix_alt(theta, phi, R=1, N_modes=50, k=1):
    """Build spherical wave coefficient matrix for given theta/phi arrays"""
    N = len(theta) * len(phi)
    D = (N_modes**2 + 2*N_modes)
    Phi_steps = len(phi)
    Fme = np.zeros((N, D, 3), dtype=np.complex_)
    Fmo = np.zeros((N, D, 3), dtype=np.complex_)
    Fne = np.zeros((N, D, 3), dtype=np.complex_)
    Fno = np.zeros((N, D, 3), dtype=np.complex_)    
    nms_idx = np.zeros((D, 2))
    ThetaPhi_idx = np.zeros((N, 2))
    for Theta_idx, Theta in enumerate(theta):
        for Phi_idx, Phi in enumerate(phi):
            ThetaPhi_idx[int(Theta_idx*Phi_steps + Phi_idx), :] = [Theta, Phi]
            for n in range(1, N_modes+1):
                for m in range(-n, n+1):
                    idx = (int(Theta_idx*Phi_steps + Phi_idx), n**2 + n - 1 + m)
                    Fme[idx[0], idx[1], :],Fmo[idx[0], idx[1], :],Fne[idx[0], idx[1], :],Fno[idx[0], idx[1], :] = nm_eo(m, n, R, Theta, Phi, k)
    for n in range(1, N_modes+1):
        for m in range(-n, n+1):
            nms_idx[n**2 + n - 1 + m] = [n, m]
    modes = np.concatenate([np.ones((D,1)),2*np.ones((D,1)),3*np.ones((D,1)),4*np.ones((D,1))], axis=0)
    nms_idx = np.concatenate([nms_idx,nms_idx,nms_idx,nms_idx], axis=0)
    nms_idx = np.concatenate([nms_idx,modes],axis=1)
    F = np.concatenate([Fme,Fmo,Fne,Fno],axis=1)
    return F, nms_idx, ThetaPhi_idx

def F_expansion(Efield, max_n, k, r, c=3):
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
    
    coefficients_F1 = []
    coefficients_F2 = []

    
    # Loop through all possible modes
    for n in range(1, max_n+1):
        for m in range(-n, n+1):
            integralF1 = 0j
            integralF2 = 0j
            
            # Perform surface integration
            for i, th in enumerate(theta):
                for j, ph in enumerate(phi):
                    # Get vector spherical harmonic
                    F1,F2 = Fmnc(-m, n, c, r, th, ph, k)
                    # Dot product with E-field conjugate
                    dot_product_F1 = (Eth[i,j]*F1[1] + 
                                      Ephi[i,j]*F1[2])
                                        #Er[i,j]*F1[0])
                    dot_product_F2 =  (Eth[i,j]*F2[1] + 
                                       Ephi[i,j]*F2[2])
                                        #Er[i,j]*F2[0])
                    # Jacobian factor for spherical coordinates
                    integralF1 += dot_product_F1 * np.sin(th) * dtheta * dphi
                    integralF2 += dot_product_F2 * np.sin(th) * dtheta * dphi
            
            z,z_prime = z_n(n,k,r,c)
            factorF1 = 1/(z**2)
            factorF2 = 1/(((1/r)*(z+k*r*z_prime))**2)

            coefficients_F1.append([factorF1*integralF1])
            coefficients_F2.append([factorF2*integralF2])
    coefficients_F1 = np.array([coefficients_F1])
    coefficients_F2 = np.array([coefficients_F2])

    return np.concatenate([coefficients_F1.reshape(-1,1),coefficients_F2.reshape(-1,1)], axis=0)
