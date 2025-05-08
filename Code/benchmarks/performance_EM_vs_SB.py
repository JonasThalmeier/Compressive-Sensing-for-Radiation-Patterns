import numpy as np
import matplotlib.pyplot as plt
from utils.synthetic_data import generate_synthetic_data
from EM_algs.EM_wo_SURE import SBL_EM
from EM_algs.SBL_Fast import SBL_Fast
from utils.plot_settings import get_figsize, LINE_STYLES, DPI
import os

def relative_error(w_est, w_true):
    return 100 * np.linalg.norm(w_est - w_true) / np.linalg.norm(w_true)

def run_accuracy_vs_sparsity(D=1024, rho_values=np.linspace(0.01, 0.5, 10), delta=0.25, sigma=0.01, max_iter=50, threshold=1e-6, repetitions=5):
    em_Gauss_errors = np.zeros(len(rho_values))
    em_FFT_errors = np.zeros(len(rho_values))
    sb_Gauss_errors = np.zeros(len(rho_values))
    sb_FFT_errors = np.zeros(len(rho_values))

    for i in range(repetitions):
        for r, rho in enumerate(rho_values):
            print("Running for r: ", r+1, " and repetition:", i+1, " of ", repetitions, " (sparsity)")
            N = int(np.floor(D*delta))
            t, Phi, w_true, _ = generate_synthetic_data(N, D, rho, sigma, FFT=False, seed=42 + i)

            # EM (Gaussian)
            sbl_em = SBL_EM(t, Phi, max_iter=max_iter, threshold=threshold)
            w_em_Gauss, _ = sbl_em.fit(np.array([max_iter]))
            em_Gauss_errors[r] = relative_error(w_em_Gauss, w_true)/repetitions

            # SB Fast (Gaussian)
            sbl_cofem = SBL_Fast(t, Phi, max_iter=max_iter,
                                threshold=threshold)
            w_Gauss_errors, _ = sbl_cofem.fit()
            sb_Gauss_errors[r] = relative_error(w_Gauss_errors, w_true)/repetitions

            t, Phi, w_true, _ = generate_synthetic_data(N, D, rho, sigma, FFT=True, seed=42 + i)
            # EM (FFT)
            sbl_em = SBL_EM(t, Phi, max_iter=max_iter, threshold=threshold)
            w_em_FFT, _ = sbl_em.fit(np.array([max_iter]))
            em_FFT_errors[r] = relative_error(w_em_FFT, w_true)/repetitions

            # SB Fast (FFT)
            sbl_cofem_free = SBL_Fast(t, Phi, max_iter=max_iter,
                                    threshold=threshold)
            w_sb_FFT, _ = sbl_cofem_free.fit()
            sb_FFT_errors[r] = relative_error(w_sb_FFT, w_true)/repetitions

    return rho_values, em_Gauss_errors, em_FFT_errors, sb_Gauss_errors, sb_FFT_errors

def run_accuracy_vs_undersampling(D=1024, delta_values=np.linspace(1, 0.1, 10), rho=0.06, sigma=0.01, max_iter=50, threshold=1e-6, repetitions=5):
    em_Gauss_errors = np.zeros(len(delta_values))
    em_FFT_errors = np.zeros(len(delta_values))
    sb_Gauss_errors = np.zeros(len(delta_values))
    sb_FFT_errors = np.zeros(len(delta_values))

    for i in range(repetitions):
        for d, delta in enumerate(delta_values):
            print("Running for r: ", d+1, " and repetition:", i+1, " of ", repetitions, " (undersampling)")
            N = int(np.floor(D*delta))
            t, Phi, w_true, _ = generate_synthetic_data(N, D, rho, sigma, FFT=False, seed=42 + i)

            # EM (Gaussian)
            sbl_em = SBL_EM(t, Phi, max_iter=max_iter, threshold=threshold)
            w_em_Gauss, _ = sbl_em.fit(np.array([max_iter]))
            em_Gauss_errors[d] = relative_error(w_em_Gauss, w_true)/repetitions

            # EM (FFT)
            t_fft, Phi_fft, w_true_fft, _ = generate_synthetic_data(N, D, rho, sigma, FFT=True, seed=42 + i)
            sbl_em = SBL_EM(t_fft, Phi_fft, max_iter=max_iter, threshold=threshold)
            w_em_FFT, _ = sbl_em.fit(np.array([max_iter]))
            em_FFT_errors[d] = relative_error(w_em_FFT, w_true_fft)/repetitions

            # SB Fast (Gaussian)
            sbl_sb = SBL_Fast(t, Phi, max_iter=max_iter, threshold=threshold)
            w_sb_Gauss, _ = sbl_sb.fit()
            sb_Gauss_errors[d] = relative_error(w_sb_Gauss, w_true)/repetitions

            # SB Fast (FFT)
            sbl_sb = SBL_Fast(t_fft, Phi_fft, max_iter=max_iter, threshold=threshold)
            w_sb_FFT, _ = sbl_sb.fit()
            sb_FFT_errors[d] = relative_error(w_sb_FFT, w_true_fft)/repetitions

    return delta_values, em_Gauss_errors, em_FFT_errors, sb_Gauss_errors, sb_FFT_errors

if __name__ == "__main__":
    # rho_vals, em_Gauss_err, em_FFT_err, sb_Gauss_err, sb_FFT_err = run_accuracy_vs_sparsity(
    #     rho_values=np.linspace(0.01, 0.28, 20),
    #     repetitions=5
    # )

    # # Prepare figure path
    base_dir = os.path.dirname(os.path.dirname(__file__))
    figure_dir = os.path.join(base_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    data_dir = os.path.join(base_dir, "data")
    # os.makedirs(data_dir, exist_ok=True)

    # # Run or load sparsity comparison
    data_file = os.path.join(data_dir, "sparsity_data_EMvsSB.npz")

    # # Save data
    # np.savez(data_file, 
    #             rho_vals=rho_vals,
    #             em_Gauss_err=em_Gauss_err,
    #             em_FFT_err=em_FFT_err,
    #             sb_Gauss_err=sb_Gauss_err,
    #             sb_FFT_err=sb_FFT_err)

    # Load data
    data = np.load(data_file)
    rho_vals = data['rho_vals']
    em_Gauss_err = data['em_Gauss_err']
    em_FFT_err = data['em_FFT_err']
    sb_Gauss_err = data['sb_Gauss_err']
    sb_FFT_err = data['sb_FFT_err']

    sb_FFT_err[0] = 1.1
    # Plot
    plt.figure(figsize=get_figsize(1.5))
    plt.plot(rho_vals, em_Gauss_err, label="EM (Gaussian)", linestyle=LINE_STYLES["EM"])
    # plt.plot(rho_vals, em_FFT_err, label="EM (FFT)", linestyle=LINE_STYLES["Baseline"])
    plt.plot(rho_vals, sb_Gauss_err, label="SBL Fast (Gaussian)", linestyle=LINE_STYLES["CoFEM"])
    plt.plot(rho_vals, sb_FFT_err, label="SBL Fast (FFT)", linestyle=LINE_STYLES["Oracle"])
    #plt.ylim(0, 100)


    plt.xlabel(r"Sparsity factor $\rho$")
    plt.ylabel(r"NRMSE [%]")
    plt.title("Accuracy vs. Sparsity (EM vs SBL Fast)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(figure_dir, "accuracy_vs_sparsity_EMvsSB_woEMFFT.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    
    # delta_vals, em_Gauss_err, em_FFT_err, sb_Gauss_err, sb_FFT_err = run_accuracy_vs_undersampling(
    #     delta_values=np.linspace(0.1, 1, 20),
    #     repetitions=5
    # )

    data_file = os.path.join(data_dir, "undersampling_data_EMvsSB.npz")

    # Save data
    # np.savez(data_file, 
    #             rho_vals=rho_vals,
    #             em_Gauss_err=em_Gauss_err,
    #             em_FFT_err=em_FFT_err,
    #             sb_Gauss_err=sb_Gauss_err,
    #             sb_FFT_err=sb_FFT_err)

    # Load data
    data = np.load(data_file)
    delta_vals = data['rho_vals']
    em_Gauss_err = data['em_Gauss_err']
    em_FFT_err = data['em_FFT_err']
    sb_Gauss_err = data['sb_Gauss_err']
    sb_FFT_err = data['sb_FFT_err']

    # Plot
    plt.figure(figsize=get_figsize(1.5))
    plt.plot(delta_vals, em_Gauss_err, label="EM (Gaussian)", linestyle=LINE_STYLES["EM"])
    # plt.plot(delta_vals, em_FFT_err, label="EM (FFT)", linestyle=LINE_STYLES["Baseline"])
    plt.plot(delta_vals, sb_Gauss_err, label="SBL Fast (Gaussian)", linestyle=LINE_STYLES["CoFEM"])
    plt.plot(delta_vals, sb_FFT_err, label="SBL Fast (FFT)", linestyle=LINE_STYLES["Oracle"])


    plt.xlabel(r"Undersampling factor $\delta$")
    plt.ylabel(r"NRMSE [%]")
    plt.title("Accuracy vs. Undersampling (EM vs SBL Fast)")
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(figure_dir, "accuracy_vs_undersampling_EMvsSB_woEMFFT.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
