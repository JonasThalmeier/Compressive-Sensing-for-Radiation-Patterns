import numpy as np
import matplotlib.pyplot as plt
from utils.synthetic_data import generate_synthetic_data
from EM_algs.EM_wo_SURE import SBL_EM
from EM_algs.CoFEM_wo_SURE import SBL_CoFEM
from utils.plot_settings import get_figsize, LINE_STYLES, DPI
import os

def relative_error(w_est, w_true):
    return 100 * np.linalg.norm(w_est - w_true) / np.linalg.norm(w_true)

def run_accuracy_vs_sparsity(D=1024, rho_values=np.linspace(0.01, 0.5, 10), delta=0.25, sigma=0.01, max_iter=50, threshold=1e-6, repetitions=5):
    FFT_errors = np.zeros(len(rho_values))
    gauss_errors = np.zeros(len(rho_values))

    for i in range(repetitions):
        for r, rho in enumerate(rho_values):
            N = int(np.floor(D*delta))

            # CoFEM (FFT)
            t, Phi, w_true, _ = generate_synthetic_data(N, D, rho, sigma, FFT=True)
            sbl_cofem = SBL_CoFEM(t, Phi, num_probes=10, max_iter=max_iter,
                                threshold=threshold, beta_in=1/sigma**2)
            w_cofem_known, _ = sbl_cofem.fit(np.array([max_iter]))
            FFT_errors[r] = relative_error(w_cofem_known, w_true)/repetitions

            # CoFEM (Gaussian)
            t, Phi, w_true, _ = generate_synthetic_data(N, D, rho, sigma, FFT=False)
            sbl_cofem_free = SBL_CoFEM(t, Phi, num_probes=10, max_iter=max_iter,
                                    threshold=threshold)
            w_cofem_learned, _ = sbl_cofem_free.fit(np.array([max_iter]))
            gauss_errors[r] = relative_error(w_cofem_learned, w_true)/repetitions

    return rho_values, FFT_errors, gauss_errors

def run_accuracy_vs_undersampling(D=1024, delta_values=np.linspace(1, 0.1, 10), rho=0.06, sigma=0.01, max_iter=50, threshold=1e-6, repetitions=5):
    FFT_errors = np.zeros(len(delta_values))
    gauss_errors = np.zeros(len(delta_values))

    for i in range(repetitions):
        for d, delta in enumerate(delta_values):
            N = int(np.floor(D*delta))
            # CoFEM (FFT)
            t, Phi, w_true, _ = generate_synthetic_data(N, D, rho, sigma, FFT=True)
            sbl_cofem = SBL_CoFEM(t, Phi, num_probes=10, max_iter=max_iter,
                                threshold=threshold, beta_in=1/sigma**2)
            w_cofem_known, _ = sbl_cofem.fit(np.array([max_iter]))
            FFT_errors[d] = relative_error(w_cofem_known, w_true)/repetitions

            # CoFEM (Gaussian)
            t, Phi, w_true, _ = generate_synthetic_data(N, D, rho, sigma, FFT=False)
            sbl_cofem_free = SBL_CoFEM(t, Phi, num_probes=10, max_iter=max_iter,
                                    threshold=threshold)
            w_cofem_learned, _ = sbl_cofem_free.fit(np.array([max_iter]))
            gauss_errors[d] = relative_error(w_cofem_learned, w_true)/repetitions

    return delta_values, FFT_errors, gauss_errors

if __name__ == "__main__":
    rho_vals, FFT_errors, gauss_errors = run_accuracy_vs_sparsity(rho_values=np.linspace(0.01, 0.25, 5),repetitions=1)
    print(FFT_errors)
    # Prepare figure path
    base_dir = os.path.dirname(os.path.dirname(__file__))
    figure_dir = os.path.join(base_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    # Plot
    plt.figure(figsize=get_figsize(1.5))
    plt.plot(rho_vals, FFT_errors, label="Fourier matrix", linestyle=LINE_STYLES["EM"])
    plt.plot(rho_vals, gauss_errors, label="Gaussian", linestyle=LINE_STYLES["CoFEM"])


    plt.xlabel(r"Sparsity factor $\rho$")
    plt.ylabel(r"NRMSE [%]")
    plt.title("Accuracy vs. Sparsity (CoFEM)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(figure_dir, "accuracy_vs_sparsity_FFTGauss.png"), dpi=DPI, bbox_inches="tight")
    plt.close()

    delta_vals, FFT_errors, gauss_errors = run_accuracy_vs_undersampling(delta_values=np.linspace(1, 0.1, 5),repetitions=1)
    print(FFT_errors)
    
    # Plot
    plt.figure(figsize=get_figsize(1.5))
    plt.plot(delta_vals, FFT_errors, label="Fourier matrix", linestyle=LINE_STYLES["EM"])
    plt.plot(delta_vals, gauss_errors, label="Gaussian", linestyle=LINE_STYLES["CoFEM"])


    plt.xlabel(r"Undersampling factor $\delta$")
    plt.ylabel(r"NRMSE [%]")
    plt.title("Accuracy vs. Undersampling (CoFEM)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(figure_dir, "accuracy_vs_undersampling_FFTGauss.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
