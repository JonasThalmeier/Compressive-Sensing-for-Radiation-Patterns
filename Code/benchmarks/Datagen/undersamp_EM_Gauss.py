from utils.synthetic_data import generate_synthetic_data
from SBL_algs.EM_wo_SURE import SBL_EM
from SBL_algs.CoFEM_wo_SURE import SBL_CoFEM
from SBL_algs.SBL_Fast import SBL_Fast
import numpy as np
import matplotlib.pyplot as plt
from utils.plot_settings import get_figsize, LINE_STYLES, DPI
import os
import sys


def relative_error(w_est, w_true):
    return 100 * np.linalg.norm(np.abs(w_est - w_true)) / np.linalg.norm(np.abs(w_true))

# Generate synthetic data
N = 100  # Length of time-domain signal
D=256
rho = 0.06  # Sparsity factor
sigma = 0.01  # Standard deviation of noise
threshold = 1e-6  # Convergence threshold
max_iter = 200
delta_values=np.linspace(1, 0.1, 20)
repetitions = 5
errors = np.zeros((len(delta_values), repetitions))
for r in range(repetitions):
    for d, delta in enumerate(delta_values):
        N = int(np.floor(D*delta))
        t, Phi, w_true, e = generate_synthetic_data(N, D, rho, sigma,FFT=False)
        
        SBL_alg = SBL_EM(t, Phi, max_iter=max_iter, threshold=threshold)
        w_cofem, _ = SBL_alg.fit(np.array([10]))
        errors[d, r] = relative_error(w_cofem, w_true)
        print(f"Delta={delta}; Iteration {d+1}/{len(delta_values)}, Repetition {r+1}/{repetitions}, Error: {errors[d, r]:.4f}")

print(errors)
base_dir = os.path.dirname(os.path.dirname(__file__))
figure_dir = os.path.join(base_dir, "figures")
os.makedirs(figure_dir, exist_ok=True)
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)
data_file = os.path.join(data_dir, "undersampling_EM_Gauss.npz")
np.savez(data_file, delta_values=delta_values, errors=errors)
data = np.load(data_file)
delta_values = data['delta_values']
errors = data['errors']
mean_errors = np.mean(errors, axis=1)
# Plot
plt.figure(figsize=get_figsize(1.5))
plt.plot(delta_values, mean_errors, linestyle=LINE_STYLES["EM"])


plt.xlabel(r"$\delta$")
plt.ylabel(r"NRMSE [%]")
plt.title("Accuracy vs. Undersampling (EM Gauss)")
plt.grid(True)
plt.tight_layout()
plt.gca().invert_xaxis()
