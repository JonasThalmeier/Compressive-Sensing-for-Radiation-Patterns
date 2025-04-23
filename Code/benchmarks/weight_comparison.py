from utils.synthetic_data import generate_synthetic_data
from EM_algs.EM_wo_SURE import SBL_EM
from EM_algs.CoFEM_wo_SURE import SBL_CoFEM
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_weight_comparison(N=50, D=150, rho=0.2, sigma=0.01, max_iter=200):
    """Plot true weights vs EM vs CoFEM estimates in one figure"""
    # Generate data and run algorithms
    t, Phi, w_true, e = generate_synthetic_data(N, D, rho, sigma)
    
    sbl_em = SBL_EM(t, Phi, max_iter=max_iter, beta=1/sigma**2)
    w_em, _ = sbl_em.fit()
    
    sbl_cofem = SBL_CoFEM(t, Phi, num_probes=1000, max_iter=max_iter, beta=1/sigma**2)
    w_cofem, _ = sbl_cofem.fit()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot settings
    bar_width = 0.25
    index = np.arange(D)
    
    # Plot all three together
    plt.bar(index - bar_width, w_true, width=bar_width, 
            color='black', label='True Weights')
    plt.bar(index, w_em, width=bar_width, 
            color='red', alpha=0.7, label='EM Estimate')
    plt.bar(index + bar_width, w_cofem, width=bar_width, 
            color='blue', alpha=0.7, label='CoFEM Estimate')
    
    # Formatting
    plt.xlabel('Weight Index')
    plt.ylabel('Magnitude')
    plt.title(f'Weight Comparison (ρ={rho})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    figure_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(os.path.join(figure_dir, 'weight_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print MSE
    print(f"\nNormalized MSE:")
    print(f"EM: {np.mean((w_em-w_true)**2)/np.mean(w_true**2):.2e}")
    print(f"CoFEM: {np.mean((w_cofem-w_true)**2)/np.mean(w_true**2):.2e}")

if __name__ == "__main__":
    plot_weight_comparison(N=50, D=100, rho=0.03, sigma=0.001, max_iter=1000)