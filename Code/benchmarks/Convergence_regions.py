from utils.synthetic_data import generate_synthetic_data  # Import the function
from EM_algs.EM_wo_SURE import SBL_EM  # Import the SBL_EM class
import numpy as np
import matplotlib.pyplot as plt
import os


# Example usage
if __name__ == "__main__":

    delta = np.arange(0.1, 1.1, 0.1)  # Undersampling factors
    rho = np.arange(0.1, 1.1, 0.1)  # Sparsity factors
    M = 50  # Length of frequency-domain signal
    sigma = 0.1  # Standard deviation of noise
    runs = 3  # Number of runs for averaging
    mse_values = np.zeros((len(delta), len(rho)))  # Initialize MSE values
    for i, d in enumerate(delta):   # Loop over undersampling factors
        for j, r in enumerate(rho):  # Loop over sparsity factors
            for run in range(runs):
                N = int(M * d)  # Length of time-domain signal
                # Generate synthetic data
                t, Phi, w, e = generate_synthetic_data(N, M, r, sigma, seed=run, FFT=False)
                # Run SBL
                track_iterations = 1  # Define tracking iterations
                sbl = SBL_EM(t, Phi, max_iter=1000, threshold=1e-6, beta_in=1/sigma**2)
                w_estimated, _ = sbl.fit()
                # Calculate MSE
                mse = np.linalg.norm(Phi @ w - Phi @ w_estimated)
                mse_values[i, j] += mse / runs
    # Create the plot
        # Create figure directory if it doesn't exist
    figure_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figure_dir, exist_ok=True)


    # Flip the mse_values array vertically to have low delta at the bottom
    mse_values_flipped = np.flipud(mse_values)
    
    # Create the heatmap with flipped values
    plt.imshow(np.log(mse_values_flipped), aspect='auto', cmap='hot', interpolation='nearest')
    plt.colorbar(label='log(MSE)')
    
    # Adjust ticks - delta is now in reverse order
    plt.xticks(np.arange(len(rho)), np.round(rho, 1))
    plt.yticks(np.arange(len(delta)), np.round(np.flip(delta), 1))
    plt.xlabel('Sparsity factor (rho)')
    plt.ylabel('Undersampling factor (delta)')
    plt.title('MSE Heatmap')
    plt.grid(True)
    plt.savefig(os.path.join(figure_dir, 'Convergance_regions.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


    
