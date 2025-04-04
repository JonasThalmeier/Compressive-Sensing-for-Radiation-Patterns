from synthetic_data import generate_synthetic_data  # Import the function
from EM_wo_SURE import SBL_EM  # Import the SBL_EM class
import numpy as np
import matplotlib.pyplot as plt


# Example usage
if __name__ == "__main__":

    delta = np.arange(0.1, 1.1, 0.1)  # Undersampling factors
    rho = np.arange(0.1, 1.1, 0.1)  # Sparsity factors
    M = 10  # Length of frequency-domain signal
    sigma = 0.1  # Standard deviation of noise
    runs = 3  # Number of runs for averaging
    mse_values = np.zeros((len(delta), len(rho)))  # Initialize MSE values
    for i, d in enumerate(delta):   # Loop over undersampling factors
        for j, r in enumerate(rho):  # Loop over sparsity factors
            for run in range(runs):
                N = int(M * d)  # Length of time-domain signal
                # Generate synthetic data
                t, Phi, w, e = generate_synthetic_data(N, M, r, sigma, seed=run)
                # Run SBL
                track_iterations = 1  # Define tracking iterations
                sbl = SBL_EM(t, Phi, 10000, 1e-8)
                w_estimated, _ = sbl.fit()
                # Calculate MSE
                mse = np.mean(np.square(Phi @ w - Phi @ w_estimated))
                mse_values[i, j] += mse / runs
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.imshow(np.log(mse_values), aspect='auto', cmap='hot', interpolation='nearest')
    plt.colorbar(label='log(MSE)')
    plt.xticks(np.round(np.arange(len(rho)),1), rho)
    plt.yticks(np.round(np.arange(len(delta)),1), delta)
    plt.xlabel('Sparsity factor (rho)')
    plt.ylabel('Undersampling factor (delta)')
    plt.title('MSE Heatmap')
    plt.grid(True)
    plt.show()


    