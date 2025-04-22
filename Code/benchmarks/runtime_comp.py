from utils.synthetic_data import generate_synthetic_data
from utils.plot_settings import *   
from EM_algs.EM_wo_SURE import SBL_EM
from EM_algs.CoFEM_wo_SURE import SBL_CoFEM
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def run_comparison(N, D, rho=0.1, sigma=0.01, threshold=1e-6, max_iter=200, repetitions=3):
    cofem_time = 0
    em_time = 0
    for i in range(repetitions):
        # Generate data
        t, Phi, w_true, e = generate_synthetic_data(N, D, rho, sigma)
        # Time CoFEM
        start_time = time.time()
        sbl_cofem = SBL_CoFEM(t, Phi, num_probes=10, max_iter=max_iter, 
                            threshold=threshold, beta=1/sigma**2)
        w_cofem, _ = sbl_cofem.fit(np.array([max_iter]))
        cofem_time += (time.time() - start_time)/repetitions
        
        # Time EM
        start_time = time.time()
        sbl_em = SBL_EM(t, Phi, max_iter=max_iter, threshold=threshold)
        w_em, _ = sbl_em.fit(np.array([max_iter]))
        em_time += (time.time() - start_time)/repetitions
    
    return cofem_time, em_time

if __name__ == "__main__":
    # Parameters
    N_min, N_max = 10, 10000
    num_points = 10
    ratio = 3  # D = ratio * N
    rho = 0.1
    sigma = 0.01
    
    # Create N values
    N_values = np.logspace(np.log10(N_min), np.log10(N_max), num_points, dtype=int)
    cofem_times = []
    em_times = []
    
    # Run comparison
    for N in N_values:
        D = ratio * N
        print(f"Running comparison for N={N}, D={D}")
        cofem_time, em_time = run_comparison(N, D, rho, sigma, repetitions=3)
        cofem_times.append(cofem_time)
        em_times.append(em_time)
    
    # Create figure directory if it doesn't exist
    figure_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    
    # Create preconfigured figure
    plt.figure(figsize=get_figsize(1.5))  # 1.5x width

    # Plot with black-only styles
    plt.plot(N_values, em_times, 
            linestyle=LINE_STYLES['EM'], 
            label='EM')
    plt.plot(N_values, cofem_times, 
            linestyle=LINE_STYLES['CoFEM'], 
            label='CoFEM')

    # Apply standardized formatting
    plt.xlabel('Signal Length (N)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime Comparison: EM vs CoFEM')
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('linear')
    plt.legend()

    # Save with consistent settings
    plt.savefig(os.path.join(figure_dir, 'runtime_comp.png'), 
                dpi=DPI, bbox_inches='tight')
    plt.close()

    # Print summary (unchanged)
    print("\nRuntime Summary:")
    print(f"Maximum N tested: {N_max}")
    print(f"EM max runtime: {max(em_times):.2f} seconds")
    print(f"CoFEM max runtime: {max(cofem_times):.2f} seconds")