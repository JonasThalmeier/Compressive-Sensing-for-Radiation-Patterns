from utils.synthetic_data import generate_synthetic_data
from SBL_algs.EM_wo_SURE import SBL_EM
from SBL_algs.CoFEM_wo_SURE import SBL_CoFEM
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Generate synthetic data
    N = 100  # Length of time-domain signal
    D = 300  # Length of frequency-domain signal (D > N)
    rho = 0.1  # Sparsity factor
    sigma = 0.01  # Standard deviation of noise
    threshold = 1e-6  # Convergence threshold
    max_iter = 200
    t, Phi, w_true, e = generate_synthetic_data(N, D, rho, sigma)
    
    # Run both algorithms for comparison
    track_iterations = np.arange(1, max_iter+1, 10)
    
    # Run CoFEM
    sbl_cofem = SBL_CoFEM(t, Phi, num_probes=1000, max_iter=max_iter, threshold=threshold, beta=1/sigma**2)
    w_cofem, tracked_weights_cofem = sbl_cofem.fit(track_iterations)
    
    # Run EM for comparison
    sbl_em = SBL_EM(t, Phi, max_iter=max_iter, threshold=threshold, beta=1/sigma**2)
    w_em, tracked_weights_em = sbl_em.fit(track_iterations)
    
    # Calculate MSE evolution for both methods
    mse_cofem = []
    mse_em = []
    
    # Calculate MSE against true weights w_true
    for weights_cofem, weights_em in zip(tracked_weights_cofem, tracked_weights_em):
        # Use absolute value for complex numbers
        mse_cofem.append(np.linalg.norm(np.abs(w_true - weights_cofem)))
        mse_em.append(np.linalg.norm(np.abs(w_true - weights_em)))
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(track_iterations, mse_cofem, 'b-o', label='CoFEM', alpha=0.7)
    plt.plot(track_iterations, mse_em, 'r-o', label='EM', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('MSE (log scale)')
    plt.title('Evolution of Weight Recovery MSE')
    plt.grid(True)
    plt.yscale('log')
    plt.legend()
    plt.show()

    # Print final results
    print(f"Final MSE (CoFEM): {mse_cofem[-1]:.6e}")
    print(f"Final MSE (EM): {mse_em[-1]:.6e}")

