import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


def barPnm(n, m, x):
    return np.sqrt(0.5*(2*n+1)*np.math.factorial(n-m)/np.math.factorial(n+m)) * sp.special.lpmv(m, n, x)

def F1mnc(m, n, r, Theta, Phi, k):
    factor = 1/np.sqrt(2*pi*n*(n+1))
    if m != 0:
        factor *= (m/np.abs(m))**m
    barPnm_val = barPnm(n, m, np.cos(Theta))
    if c == 1:
        z = sp.besselj(n, k*r)
    elif c == 2:
        z = sp.neumann(n, k*r)
    elif c == 3:
        z = sp.besselj(n, k*r) + 1j*sp.neumann(n, k*r)
    else:
        z = sp.besselj(n, k*r) - 1j*sp.neumann(n, k*r)
    Theta = z * 1j * m * barPnm_val * np.exp(1j*m*Phi) / np.sin(Theta)
    Phi = -z