import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


def barPnm(n, m, x):
    return np.sqrt(0.5*(2*n+1)*np.math.factorial(n-m)/np.math.factorial(n+m)) * sp.special.lpmv(m, n, x)

def dbarPnmcosTheta_dTheta(n, m, Theta):
    Pnm_derivate = 1/(np.cos(Theta)**2-1)*(-(n+1)*np.cos(Theta)*sp.special.lpmv(m, n, np.cos(Theta)) + (n-m+1)*sp.special.lpmv(m, n+1, np.cos(Theta)))
    return -np.sqrt((2*n+1)*(np.math.factorial(n-m)/(2*np.math.factorial(n+m)))) * Pnm_derivate *np.sin(Theta)


def F1mnc(m, n, c, r, Theta, Phi, k):
    factor = 1/np.sqrt(2*np.pi*n*(n+1))
    if m != 0:
        factor *= (m/np.abs(m))**m
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
    Theta1 = z * 1j * m * barPnm_val * np.exp(1j*m*Phi) / np.sin(Theta)
    Phi1 = -z * dbarPnm_val * np.exp(1j*m*Phi)
    r2 = (n*(n+1)/(k*r))*z*barPnm*np.exp(1j*m*Phi)
    Theta2 = 1/(k*r) * (z + k*r*z_prime) * dbarPnm_val * np.exp(1j*m*Phi)
    Phi2 = 1/(k*r) * (z + k*r*z_prime) * 1j * m * barPnm_val * np.exp(1j*m*Phi) / np.sin(Theta)
    return factor*np.array([0,Theta1,Phi1]), factor*np.array([r2,Theta2,Phi2])
    