import numpy as np
import matplotlib.pyplot as plt
from utils.synthetic_data import generate_synthetic_data
from EM_algs.EM_wo_SURE import SBL_EM
from EM_algs.SBL_Fast import SBL_Fast
from utils.plot_settings import get_figsize, LINE_STYLES, DPI
import os

base_dir = os.path.dirname(__file__)  # Only go up once since we're already in /Code
data_dir = os.path.join(base_dir, "data")
data_file = os.path.join(data_dir, "sparsity_data_EMvsSB.npz")
data = np.load(data_file)
rho_vals = data['rho_vals']
em_Gauss_err = data['em_Gauss_err']
em_FFT_err = data['em_FFT_err']
sb_Gauss_err = data['sb_Gauss_err']
sb_FFT_err = data['sb_FFT_err']
print(em_FFT_err)
print(em_Gauss_err)
print(sb_FFT_err)
print(sb_Gauss_err)