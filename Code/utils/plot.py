import numpy as np
import matplotlib.pyplot as plt

def plot_coefficient_magnitudes(*coeff_arrays):
    """Visualize coefficient magnitudes with optional comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Collect magnitude ranges for consistent color scaling
    s_magnitudes = {1: [], 2: []}
    for arr in coeff_arrays:
        n, m, s, vals = arr
        magnitudes = np.abs(vals)
        non_zero = magnitudes != 0
        for st in [1, 2]:
            s_magnitudes[st].extend(magnitudes[(s == st) & non_zero])
    
    vmin_vmax = {st: (min(mags), max(mags)) if mags else (0, 1) 
                for st, mags in s_magnitudes.items()}
    
    # Plot each dataset
    for i, arr in enumerate(coeff_arrays):
        n, m, s, vals = arr
        magnitudes = np.abs(vals)
        non_zero = magnitudes != 0
        
        # Apply filtering
        n = n[non_zero]
        m = m[non_zero]
        s = s[non_zero]
        magnitudes = magnitudes[non_zero]
        
        # Offset subsequent datasets
        if i > 0:
            n = n + 0.1  # Horizontal shift for visual separation
            
        # Plot for each s category
        for st, ax in zip([1, 2], [ax1, ax2]):
            mask = s == st
            ax.scatter(n[mask], m[mask], c=magnitudes[mask], 
                      cmap='viridis', s=50, alpha=0.7,
                      vmin=vmin_vmax[st][0], vmax=vmin_vmax[st][1],
                      label=f'Dataset {i+1}' if len(coeff_arrays) > 1 else None)
    
    # Configure plots
    for ax, st in zip([ax1, ax2], [1, 2]):
        ax.set_xlabel('n')
        ax.set_ylabel('m')
        ax.set_title(f's = {st}')
        if ax.collections:
            fig.colorbar(ax.collections[0], ax=ax, label='Coefficient Magnitude')
        ax.legend()
    
    plt.suptitle('Spherical Harmonics Coefficient Magnitudes Comparison' 
                if len(coeff_arrays) > 1 
                else 'Spherical Harmonics Coefficient Magnitudes')
    plt.tight_layout()
    plt.show()