import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# --- Core Settings ---
plt.style.use('default')  # Start with clean slate
DPI = 300
WIDTH_INCHES = 6.0
GOLDEN_RATIO = 1.618

# --- Black-Only Styling ---
rcParams.update({
    'axes.prop_cycle': plt.cycler('color', ['black']),  # ← Critical line
    'lines.color': 'black',
    'patch.edgecolor': 'black',
    'text.color': 'black',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'grid.color': 'black',
    
    # Font settings
    'font.size': 12,
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    
    # Line styles
    'lines.linewidth': 2,
    'grid.alpha': 0.5,
    'grid.linestyle': '--'
})

# --- Style Presets ---
LINE_STYLES = {
    'EM': '-',
    'CoFEM': '--',
    'Oracle': ':',
    'Baseline': '-.'
}

def get_figsize(columns=1):
    width = WIDTH_INCHES * columns
    return (width, width / GOLDEN_RATIO)

def ensure_figure_dir():
    """Create figures directory if needed"""
    figdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figdir, exist_ok=True)
    return figdir