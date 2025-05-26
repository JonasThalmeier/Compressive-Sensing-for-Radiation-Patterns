from utils.synthetic_data import generate_synthetic_data
from utils.plot_settings import *
from SBL_algs.EM_wo_SURE import SBL_EM
from SBL_algs.CoFEM_wo_SURE import SBL_CoFEM
import numpy as np
import matplotlib.pyplot as plt
import os

def run_mse_comparison(N, D, rho_values, sigma=0.01, max_iter=200, repetitions=5):
    """Compare MSE across different sparsity levels"""
    mse_cofem = np.zeros(len(rho_values))
    mse_em = np.zeros(len(rho_values))
    
    for i, rho in enumerate(rho_values):
        temp_mse_cofem = 0
        temp_mse_em = 0
        
        for _ in range(repetitions):
            # Generate new data for each repetition
            t, Phi, w_true, e = generate_synthetic_data(N, D, rho, sigma)
            
            # Run CoFEM
            sbl_cofem = SBL_CoFEM(t, Phi, num_probes=10, max_iter=max_iter)
            w_cofem, _ = sbl_cofem.fit()
            temp_mse_cofem += np.mean((w_cofem - w_true)**2) / np.mean(w_true**2)
            
            # Run EM
            sbl_em = SBL_EM(t, Phi, max_iter=max_iter)
            w_em, _ = sbl_em.fit()
            temp_mse_em += np.mean((w_em - w_true)**2) / np.mean(w_true**2)
        
        # Average over repetitions
        mse_cofem[i] = temp_mse_cofem / repetitions
        mse_em[i] = temp_mse_em / repetitions
    
    return mse_cofem, mse_em

if __name__ == "__main__":
    # Parameters
    N = 50       # Fixed signal length
    D = 3 * N     # D = 3N as in your original
    sigma = 0.01   # Noise level
    
    # Sparsity levels to test (0.01 to 0.5)
    rho_values = np.linspace(0.01, 1, 15)
    
    # Run comparison
    mse_cofem, mse_em = run_mse_comparison(N, D, rho_values, sigma, repetitions=1)
    
    # Plotting
    plt.figure(figsize=get_figsize(1.5))
    
    plt.plot(rho_values, mse_em, 
             linestyle=LINE_STYLES['EM'], 
             label='EM')
    plt.plot(rho_values, mse_cofem, 
             linestyle=LINE_STYLES['CoFEM'], 
             label='CoFEM')
    
    plt.xlabel('Sparsity Level (ρ)')
    plt.ylabel('Normalized MSE')
    plt.title('MSE Comparison vs Sparsity')
    plt.grid(True)
    plt.yscale('log')  # Often helpful for MSE
    plt.legend()
    
    # Save figure
    figure_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(os.path.join(figure_dir, 'MSEw_vs_rho.png'), 
                dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # Print results table
    print("\nMSE Comparison Summary:")
    print(f"{'ρ':<8}{'EM MSE':<12}{'CoFEM MSE':<12}")
    for rho, em, cf in zip(rho_values, mse_em, mse_cofem):
        print(f"{rho:.3f}:  {em:.2e}  {cf:.2e}")