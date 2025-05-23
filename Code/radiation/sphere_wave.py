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
        sinTheta = 1e-10
    Theta1 = z * 1j * m * barPnm_val * np.exp(1j*m*Phi) / sinTheta
    Phi1 = -z * dbarPnm_val * np.exp(1j*m*Phi)
    r2 = (n*(n+1)/(k*r))*z*barPnm_val*np.exp(1j*m*Phi)
    Theta2 = 1/(k*r) * (z + k*r*z_prime) * dbarPnm_val * np.exp(1j*m*Phi)
    Phi2 = 1/(k*r) * (z + k*r*z_prime) * 1j * m * barPnm_val * np.exp(1j*m*Phi) / np.sin(Theta)
    if np.any(np.isnan(np.array([Theta1, Phi1, r2, Theta2, Phi2]))):
        print(f"NaN detected in Fmnc for m={m}, n={n}, c={c}, r={r}, Theta={Theta}, Phi={Phi}")
        print(f"Values: {Theta1}, {Phi1}, {r2}, {Theta2}, {Phi2}")
    return factor*np.array([0,Theta1,Phi1]), factor*np.array([r2,Theta2,Phi2])