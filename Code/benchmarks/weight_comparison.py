from utils.synthetic_data import generate_synthetic_data
from EM_algs.EM_wo_SURE import SBL_EM
from EM_algs.CoFEM_wo_SURE import SBL_CoFEM
from EM_algs.SBL_Fast import SBL_Fast
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_weight_comparison(N=50, D=150, rho=0.2, sigma=0.01, max_iter=200, FFT=True):
    """Plot true weights vs EM vs CoFEM estimates in one figure"""
    # Generate data and run algorithms
    t, Phi, w_true, e = generate_synthetic_data(N, D, rho, sigma, FFT=FFT, seed=3)
    
    sbl_em = SBL_EM(t, Phi, max_iter=max_iter)
    w_em, _ = sbl_em.fit()
    
    sbl_fast = SBL_Fast(t, Phi, max_iter=max_iter)
    w_fast, _ = sbl_fast.fit()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot settings
    bar_width = 0.25
    index = np.arange(D)
    
    # Plot all three together
    plt.bar(index - bar_width, np.abs(w_true), width=bar_width, 
            color='black', label='True Weights')
    plt.bar(index, np.abs(w_em), width=bar_width, 
            color='red', alpha=0.7, label='EM Estimate')
    plt.bar(index + bar_width, np.abs(w_fast), width=bar_width, 
            color='blue', alpha=0.7, label='SBL Fast Estimate')
    
    # Set matrix type string for title and filename
    matrix_type = "FFT" if FFT else "Gaussian"
    
    # Formatting with matrix type in title
    plt.xlabel('Weight Index')
    plt.ylabel('Magnitude')
    plt.title(f'Weight Comparison with {matrix_type} Matrix (ρ={rho}, σ={sigma}, δ={N/D})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure with matrix type in filename
    figure_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(os.path.join(figure_dir, f'weight_comparison_{matrix_type}_simga{sigma}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print MSE
    print(f"\nNormalized MSE:")
    print(f"EM: {np.mean((np.abs(w_em-w_true))**2)/np.mean(np.abs(w_true)**2):.2e}")
    print(f"SBL Fast: {np.mean((np.abs(w_fast-w_true))**2)/np.mean(np.abs(w_true)**2):.2e}")

if __name__ == "__main__":
    plot_weight_comparison(N=50, D=100, rho=0.05, sigma=0.01, max_iter=1000, FFT=True)
    plot_weight_comparison(N=50, D=100, rho=0.05, sigma=0.01, max_iter=1000, FFT=False)