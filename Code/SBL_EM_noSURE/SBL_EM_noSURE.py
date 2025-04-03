from synthetic_data import generate_synthetic_data  # Import the function
from EM import SBL_EM  # Import the SBL_EM class
import numpy as np
import matplotlib.pyplot as plt


# Example usage
if __name__ == "__main__":
    N = 10  # Length of time-domain signal
    M = 30  # Length of frequency-domain signal (M > N)
    rho = 0.1  # Sparsity factor
    sigma = 0.01  # Standard deviation of noise

    t, Phi, w, e = generate_synthetic_data(N, M, rho, sigma)

    print("Generated synthetic data:")
    print("t (time-domain signal):", t)
    print("Phi (Fourier transform matrix):", Phi)
    print("w (sparse frequency-domain signal):", w)
    print("e (noise):", e)

    # Run SBL
    track_iterations = np.arange(1, 1001, 10)  # Define tracking iterations
    sbl = SBL_EM(t, Phi, 1000000, 1e-8)  # Initialize SBL with EM algorithm
    w_estimated, tracked_weights = sbl.fit(track_iterations)
    
    # Calculate and plot MSE evolution
    mse_values = []
    for weights in tracked_weights:
        # Calculate reconstructed signal
        t_reconstructed = Phi @ weights
        # Calculate MSE
        mse = np.mean(np.square(t - t_reconstructed))
        mse_values.append(mse)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(track_iterations, mse_values, 'b-o')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('Evolution of Signal Reconstruction MSE')
    plt.grid(True)
    plt.yscale('log')  # Use log scale for better visualization
    plt.show()


    # Print results
    print("True weights:", w)
    print("Estimated weights:", w_estimated)
    print("MSE:", np.mean((w - w_estimated)**2))

    