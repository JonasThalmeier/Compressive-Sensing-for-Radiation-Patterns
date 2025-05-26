import numpy as np
import matplotlib.pyplot as plt
from utils.plot_settings import get_figsize, LINE_STYLES, DPI
import os
from radiation.sphere_wave import F_matrix



if __name__ == "__main__":
    seed = np.random.randint(0, 10000)
    # Generate synthetic data
    N_modes=7
    N = 20  # Length of time-domain signal
    D = 2*N_modes**2+4*N_modes  # Length of frequency-domain signal (D > N)
    L = 3
    rho = 0.1  # Sparsity factor
    sigma = 1e-8  # Standard deviation of noise
    threshold = 1e-8  # Convergence threshold
    eta = 376.7
    f = 3e9
    lamb = 3e8 / f
    k = 2 * np.pi / lamb
    r= 0.4 
    max_iter = 200
    Phi, nms_idx, ThetaPhi_idx = F_matrix(R=r, Theta_steps=180, Phi_steps=360, N_modes=N_modes, c = 3, k=k)
    # print(f"t.shape: {t.shape}, Phi.shape: {Phi.shape}, w_true.shape: {w_true.shape}, e.shape: {e.shape}")

    E = (k/np.sqrt(eta)) * (Phi[:,0:int(D/2)]+Phi[:,int(D/2):D]) # E field
    H = -1*1j*k*np.sqrt(eta) * (Phi[:,int(D/2):D]+Phi[:,0:int(D/2)]) # H field
    S_r = 0.5 * np.real(E[:,:,1]*H[:,:,2].conj()-E[:,:,2]*H[:,:,1].conj()) # Power
    Power = np.sum(S_r*r**2, axis=0)  # shape: (D,)
    # Run both algorithms for comparison

   

    # Energy of each mode: sum over all points and field components
    mode_energies = np.sum(np.abs(Phi)**2, axis=(0, 2))  # shape: (D,)

    # # Plot energy vs mode index
    # plt.figure(figsize=(12, 4))
    # plt.plot(mode_energies)
    # labels = [f"n={int(n)} m={int(m)} s={int(s)}" for n, m, s in nms_idx]
    # plt.xticks(range(len(labels)), labels, rotation=90)
    # plt.title("Energy per Mode")
    # plt.xlabel("Mode Index")
    # plt.ylabel("Energy")
    # plt.grid(True)
    # plt.show()

    # Create figure with three subplots
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Create colormap for different n values
    colors = plt.cm.viridis(np.linspace(0, 1, N_modes))
    nms_idx = nms_idx[0:int(D/2),:]

    # For each field component        # For each n and s
    for n in range(1, N_modes+1):
    
        # Get indices for this n and s
        mask = (nms_idx[:, 0] == n)
        m_values = nms_idx[mask, 1]
        # Sort by m values to get continuous lines
        sort_idx = np.argsort(m_values)
        m_values = m_values[sort_idx]
        # field_values = np.sum(np.abs(Phi[:, mask, :]**2), axis=(0,2))[sort_idx]  # Using first point for visualization
        power_vals = Power[mask]
        
        # Plot with different line styles for s=1 and s=2
        linestyle = '-'
        ax.plot(m_values, power_vals, 
                linestyle=linestyle, 
                color=colors[n-1], 
                label=f'n={n}')

    ax.set_xlabel('m')
    ax.legend()
    ax.set_ylabel(f'|F^2|')
    ax.grid(True)


    plt.tight_layout()
    plt.show()

    mask = (nms_idx[:, 0] == 3) & (nms_idx[:, 2] == 1) & (nms_idx[:, 1] == 0)
    print("------x------- ")
    print(Phi[:,mask,0])
    print("------y------- ")
    print(Phi[:,mask,1])
    print("------z------- ")
    print(Phi[:,mask,2])