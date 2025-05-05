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
    em_known_errors = np.zeros(len(rho_values))
    em_learned_errors = np.zeros(len(rho_values))
    cofem_known_errors = np.zeros(len(rho_values))
    cofem_learned_errors = np.zeros(len(rho_values))

    for i in range(repetitions):
        for r, rho in enumerate(rho_values):
            N = int(np.floor(D*delta))
            t, Phi, w_true, _ = generate_synthetic_data(N, D, rho, sigma, FFT=False)

            # EM (known noise)
            sbl_em = SBL_EM(t, Phi, max_iter=max_iter, threshold=threshold, beta_in=1/sigma**2)
            w_em_known, _ = sbl_em.fit(np.array([max_iter]))
            em_known_errors[r] = relative_error(w_em_known, w_true)/repetitions

            # EM (unknown noise)
            sbl_em = SBL_EM(t, Phi, max_iter=max_iter, threshold=threshold)
            w_em_unknown, _ = sbl_em.fit(np.array([max_iter]))
            em_learned_errors[r] = relative_error(w_em_unknown, w_true)/repetitions

            # CoFEM (known noise)
            sbl_cofem = SBL_CoFEM(t, Phi, num_probes=10, max_iter=max_iter,
                                threshold=threshold, beta_in=1/sigma**2)
            w_cofem_known, _ = sbl_cofem.fit(np.array([max_iter]))
            cofem_known_errors[r] = relative_error(w_cofem_known, w_true)/repetitions

            # CoFEM (unknown noise: omit beta)
            sbl_cofem_free = SBL_CoFEM(t, Phi, num_probes=10, max_iter=max_iter,
                                    threshold=threshold)
            w_cofem_learned, _ = sbl_cofem_free.fit(np.array([max_iter]))
            cofem_learned_errors[r] = relative_error(w_cofem_learned, w_true)/repetitions

    return rho_values, em_known_errors, em_learned_errors, cofem_known_errors, cofem_learned_errors

def run_accuracy_vs_undersampling(D=1024, delta_values=np.linspace(1, 0.1, 10), rho=0.06, sigma=0.01, max_iter=50, threshold=1e-6, repetitions=5):
    em_known_errors = np.zeros(len(delta_values))
    em_learned_errors = np.zeros(len(delta_values))
    cofem_known_errors = np.zeros(len(delta_values))
    cofem_learned_errors = np.zeros(len(delta_values))

    for i in range(repetitions):
        for d, delta in enumerate(delta_values):
            N = int(np.floor(D*delta))
            t, Phi, w_true, _ = generate_synthetic_data(N, D, rho, sigma, FFT=False)

            # EM (known noise)
            sbl_em = SBL_EM(t, Phi, max_iter=max_iter, threshold=threshold, beta_in=1/sigma**2)
            w_em_known, _ = sbl_em.fit(np.array([max_iter]))
            em_known_errors[d] = relative_error(w_em_known, w_true)/repetitions

            # EM (unknown noise)
            sbl_em = SBL_EM(t, Phi, max_iter=max_iter, threshold=threshold)
            w_em_unknown, _ = sbl_em.fit(np.array([max_iter]))
            em_learned_errors[d] = relative_error(w_em_unknown, w_true)/repetitions

            # CoFEM (known noise)
            sbl_cofem = SBL_CoFEM(t, Phi, num_probes=10, max_iter=max_iter,
                                threshold=threshold, beta_in=1/sigma**2)
            w_cofem_known, _ = sbl_cofem.fit(np.array([max_iter]))
            cofem_known_errors[d] = relative_error(w_cofem_known, w_true)/repetitions

            # CoFEM (unknown noise: omit beta)
            sbl_cofem_free = SBL_CoFEM(t, Phi, num_probes=10, max_iter=max_iter,
                                    threshold=threshold)
            w_cofem_learned, _ = sbl_cofem_free.fit(np.array([max_iter]))
            cofem_learned_errors[d] = relative_error(w_cofem_learned, w_true)/repetitions

    return delta_values, em_known_errors, em_learned_errors, cofem_known_errors, cofem_learned_errors

if __name__ == "__main__":
    rho_vals, em_known_err, em_learned_err, cofem_known_err, cofem_learned_err = run_accuracy_vs_sparsity(rho_values=np.linspace(0.01, 0.28, 20),repetitions=5)

    # Prepare figure path
    base_dir = os.path.dirname(os.path.dirname(__file__))
    figure_dir = os.path.join(base_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    # Plot
    plt.figure(figsize=get_figsize(1.5))
    plt.plot(rho_vals, em_known_err, label="EM (σ² known)", linestyle=LINE_STYLES["EM"])
    plt.plot(rho_vals, em_learned_err, label="EM (σ² unknown)", linestyle=LINE_STYLES["Baseline"])
    plt.plot(rho_vals, cofem_known_err, label="CoFEM (σ² known)", linestyle=LINE_STYLES["CoFEM"])
    plt.plot(rho_vals, cofem_learned_err, label="CoFEM (σ² unknown)", linestyle=LINE_STYLES["Oracle"])

    plt.xlabel(r"Sparsity factor $\rho$")
    plt.ylabel(r"NRMSE [%]")
    plt.title("Accuracy vs. Sparsity (Gaussian Dictionary)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(figure_dir, "accuracy_vs_sparsity_EMCoFEM.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    
    delta_vals, em_known_err, em_learned_err, cofem_known_err, cofem_learned_err = run_accuracy_vs_undersampling(delta_values=np.linspace(0.1, 1, 20),repetitions=1)

    # Prepare figure path
    base_dir = os.path.dirname(os.path.dirname(__file__))
    figure_dir = os.path.join(base_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    # Plot
    plt.figure(figsize=get_figsize(1.5))
    plt.plot(delta_vals, em_known_err, label="EM (σ² known)", linestyle=LINE_STYLES["EM"])
    plt.plot(delta_vals, em_learned_err, label="EM (σ² unknown)", linestyle=LINE_STYLES["Baseline"])
    plt.plot(delta_vals, cofem_known_err, label="CoFEM (σ² known)", linestyle=LINE_STYLES["CoFEM"])
    plt.plot(delta_vals, cofem_learned_err, label="CoFEM (σ² unknown)", linestyle=LINE_STYLES["Oracle"])

    plt.xlabel(r"Undersampling factor $\delta$")
    plt.ylabel(r"NRMSE [%]")
    plt.title("Accuracy vs. Undersampling (Gaussian Dictionary)")
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(figure_dir, "accuracy_vs_undersampling_EMCoFEM.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
