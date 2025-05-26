import numpy as np
import matplotlib.pyplot as plt
from utils.synthetic_data import generate_synthetic_data
from SBL_algs.EM_wo_SURE import SBL_EM
from SBL_algs.CoFEM_wo_SURE import SBL_CoFEM
from SBL_algs.SBL_Fast import SBL_Fast
from SBL_algs.SBL_Fast_Vector import SBL_Fast_Vector
from utils.plot_settings import get_figsize, LINE_STYLES, DPI
import os

def relative_error(w_est, w_true):
    return 100 * np.linalg.norm(np.abs(w_est - w_true)) / np.linalg.norm(np.abs(w_true))

def run_accuracy_vs_sparsity(D=1024, rho_values=np.linspace(0.01, 0.5, 10), delta=0.25, sigma=0.01, max_iter=50, threshold=1e-6, repetitions=5):
    FFT_errors = np.zeros(len(rho_values))
    gauss_errors = np.zeros(len(rho_values))

    for i in range(repetitions):
        for r, rho in enumerate(rho_values):
            print("Running for r: ", r+1, " and repetition:", i+1, " of ", repetitions, " (sparsity)")

            N = int(np.floor(D*delta))

            # CoFEM (FFT)
            t, Phi, w_true, _ = generate_synthetic_data(N, D, rho, sigma, FFT=True, seed=42 + i)
            sbl_cofem = SBL_Fast(t, Phi,  max_iter=max_iter,
                                threshold=threshold)
            w_cofem_known, _ = sbl_cofem.fit()
            FFT_errors[r] = relative_error(w_cofem_known, w_true)/repetitions

            # CoFEM (Gaussian)
            t, Phi, w_true, _ = generate_synthetic_data(N, D, rho, sigma, FFT=True, seed=42 + i)
            sbl_cofem_free = SBL_Fast_Vector(t, Phi,  max_iter=max_iter,
                                    threshold=threshold)
            w_cofem_learned, _ = sbl_cofem_free.fit()
            gauss_errors[r] = relative_error(w_cofem_learned, w_true)/repetitions

    return rho_values, FFT_errors, gauss_errors

def run_accuracy_vs_undersampling(D=1024, delta_values=np.linspace(1, 0.1, 10), rho=0.06, sigma=0.01, max_iter=50, threshold=1e-6, repetitions=5):
    FFT_errors = np.zeros(len(delta_values))
    gauss_errors = np.zeros(len(delta_values))

    for i in range(repetitions):
        for d, delta in enumerate(delta_values):
            print("Running for d: ", d+1, " and repetition:", i+1, " of ", repetitions, " (undersampling)")
            N = int(np.floor(D*delta))
            # CoFEM (FFT)
            t, Phi, w_true, _ = generate_synthetic_data(N, D, rho, sigma, FFT=True, seed=42 + i)
            sbl_cofem = SBL_Fast(t, Phi,  max_iter=max_iter,
                                threshold=threshold)
            w_cofem_known, _ = sbl_cofem.fit()
            FFT_errors[d] = relative_error(w_cofem_known, w_true)/repetitions

            # CoFEM (Gaussian)
            t, Phi, w_true, _ = generate_synthetic_data(N, D, rho, sigma, FFT=True, seed=42 + i)
            sbl_cofem_free = SBL_Fast_Vector(t, Phi,  max_iter=max_iter,
                                    threshold=threshold)
            w_cofem_learned, _ = sbl_cofem_free.fit()
            gauss_errors[d] = relative_error(w_cofem_learned, w_true)/repetitions

    return delta_values, FFT_errors, gauss_errors

if __name__ == "__main__":
    
    rho_vals, FFT_errors, gauss_errors = run_accuracy_vs_sparsity(D=512, rho_values=np.linspace(0.01, 0.28, 10),repetitions=1, max_iter=50)
    print(FFT_errors)
    print(gauss_errors)
    # if np.all(np.isnan(FFT_errors)):
    #     y_limit = 70  # Fallback value
    # else:
    #     y_limit = min(1.2 * np.nanmax(FFT_errors), 70)
    # Prepare figure path
    base_dir = os.path.dirname(os.path.dirname(__file__))
    figure_dir = os.path.join(base_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    data_file = os.path.join(data_dir, "sparsity_data_GausFFT.npz")
    np.savez(data_file, 
                 rho_vals=rho_vals,
                 FFT_errors=FFT_errors,
                 gauss_errors=gauss_errors)
    
    data = np.load(data_file)
    rho_vals = data['rho_vals']
    FFT_errors = data['FFT_errors']
    gauss_errors = data['gauss_errors']

    # Plot
    plt.figure(figsize=get_figsize(1.5))
    plt.plot(rho_vals, FFT_errors, label="old", linestyle=LINE_STYLES["EM"])
    plt.plot(rho_vals, gauss_errors, label="new", linestyle=LINE_STYLES["CoFEM"])
    # plt.ylim(0, 30)


    plt.xlabel(r"Sparsity factor $\rho$")
    plt.ylabel(r"NRMSE [%]")
    plt.title("Accuracy vs. Sparsity (old SBL vs new SBL)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(figure_dir, "SBL_old_vs_new_spars.png"), dpi=DPI, bbox_inches="tight")
    plt.close()

    delta_vals, FFT_errors, gauss_errors = run_accuracy_vs_undersampling(D=512, delta_values=np.linspace(0.1, 1, 10),repetitions=1,max_iter=50)
    print(FFT_errors)
    print(gauss_errors)
    # if np.all(np.isnan(FFT_errors)):
    #     y_limit = 70  # Fallback value
    # else:
    #     y_limit = min(1.2 * np.nanmax(FFT_errors), 70)

    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    data_file = os.path.join(data_dir, "undersampling_data_GausFFT.npz")
    np.savez(data_file, 
                 rho_vals=rho_vals,
                 FFT_errors=FFT_errors,
                 gauss_errors=gauss_errors)
    
    data = np.load(data_file)
    rho_vals = data['rho_vals']
    FFT_errors = data['FFT_errors']
    gauss_errors = data['gauss_errors']

    # Plot
    plt.figure(figsize=get_figsize(1.5))
    plt.plot(delta_vals, FFT_errors, label="old", linestyle=LINE_STYLES["EM"])
    plt.plot(delta_vals, gauss_errors, label="new", linestyle=LINE_STYLES["CoFEM"])
    # plt.ylim(0, y_limit)

    plt.xlabel(r"Undersampling factor $\delta$")
    plt.ylabel(r"NRMSE [%]")
    plt.title("Accuracy vs. Undersampling (old SBL vs new SBL)")
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(figure_dir, "SBL_old_vs_new_under.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
