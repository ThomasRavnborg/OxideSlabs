
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
            "font.serif": ["DejaVu Serif"],
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

    def set_style_ax(self, ax, gridlines=False, minimalist=False):
        # Minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        # Minimalist style
        if minimalist:
            # Hide top and right spines (borders)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Ticks only on bottom and left
            ax.tick_params(which='both', top=False, right=False)
        else:
            # Tick parameters
            ax.tick_params(which='both', direction='in', top=True, right=True)
            ax.tick_params(which='major', length=7, width=1.2)
            ax.tick_params(which='minor', length=4, width=1)
        # Optional grid
        if gridlines:
            ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.7)
            ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.4)