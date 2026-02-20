
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

class PlotSettings():
    """
    Class to set global plot styles and provide a function for consistent plot formatting.
    """

    def __init__(self):
        pass

    def set_global_style(self):
        # Global plot style settings
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],      # Use DejaVu Serif for text
            "mathtext.fontset": "cm",            # Use Computer Modern for math
            "font.size": 14,                     # Base font size
            "axes.labelsize": 16,                # Axis label font size
            "axes.titlesize": 16,                # Title font size
            "legend.fontsize": 13,               # Legend font size
            "xtick.labelsize": 13,               # X tick label size
            "ytick.labelsize": 13,               # Y tick label size
            "axes.linewidth": 1.2,               # Thicker axis lines
            "xtick.direction": "in",             # x-tick direction
            "ytick.direction": "in",             # y-tick direction
            "text.usetex": False,                # Enable LaTeX if needed
            "figure.dpi": 150,                   # Good resolution for screens
            "savefig.dpi": 300                   # High resolution for saving
        })


    # Function to set consistent styles for individual axes
    def set_style_ax(self, ax, style='default', minor=True, gridlines=False):
        if minor:
            # Minor ticks
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        if style == 'default':
            # Ticks on all sides, pointing inwards, with specific lengths and widths
            ax.tick_params(which='both', direction='in', top=True, right=True)
            ax.tick_params(which='major', length=6, width=1)
            ax.tick_params(which='minor', length=3, width=1)
            # Change spine widths
            for spine in ax.spines.values():
                spine.set_linewidth(1)

        if style == 'minimalist':
            # Minimalist style
            # Hide top and right spines (borders)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Ticks only on bottom and left
            ax.tick_params(which='both', top=False, right=False)
        
        if style == 'bands':
            # Ticks on all sides, pointing outwards, with specific lengths and widths
            ax.tick_params(which='both', direction='out', labelsize = 16, pad = 4)
            ax.tick_params(which='major', length=6)
            ax.tick_params(which='minor', length=3)

        # Optional grid
        if gridlines:
            ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.7)
            ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.4)