
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

class PlotSettings():
    """
    Class to set global plot styles and provide a function for consistent plot formatting.
    """

    def __init__(self):
        pass

    def set_global_style(self, latex=False):
        """Function to set global plot styles for consistent and publication-quality figures.
        Parameters:
        - latex: Boolean indicating whether to use LaTeX for text rendering.
        Returns:
        - None: Updates Matplotlib rcParams for consistent styling across all plots.
        """
        # Global plot style settings
        if latex:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "font.size": 10,
                "axes.labelsize": 10,
                "axes.titlesize": 10,
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "axes.linewidth": 0.8,
                "figure.dpi": 150,
                "savefig.dpi": 300,
            })

        else:
            plt.rcParams.update({
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": ["DejaVu Serif"],
                "mathtext.fontset": "cm",
                "font.size": 14,
                "axes.labelsize": 16,
                "axes.titlesize": 16,
                "legend.fontsize": 13,
                "xtick.labelsize": 13,
                "ytick.labelsize": 13,
                "axes.linewidth": 1.2,
                "figure.dpi": 150,
                "savefig.dpi": 300
            })
        """
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
        """


    def set_style_ax(self, ax, style='default', minor=True, gridlines=False):
        """Function to set consistent styles for axes, including ticks, spines, and gridlines.
        Parameters:
        - ax: Matplotlib axis object to style.
        - style: String specifying the style to apply ('default', 'minimalist', 'bands').
        - minor: Boolean indicating whether to show minor ticks.
        - gridlines: Boolean indicating whether to show gridlines.
        Returns:
        - None: Modifies the provided axis object in place to apply the specified style.
        """
        if minor:
            # Minor ticks
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        
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
            ax.tick_params(which='major', length=6, width=1)
            ax.tick_params(which='minor', length=3, width=1)

        # Optional grid
        if gridlines:
            ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.7)
            ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.4)


    def set_size(self, fig, width_pt=369.0, fraction=1, aspect_ratio=None):
        """Function to set figure dimensions to avoid scaling in LaTeX.
        Parameters:
        - fig: Matplotlib figure object to set the size of.
        - width_pt: Document textwidth or columnwidth in pt.
        - fraction: Fraction of the width to occupy.
        - aspect_ratio: The aspect ratio of the figure.
        Returns:
        - None: Modifies the provided figure object in place to set the size.
        """
        inches_per_pt = 1 / 72.27
        # Target width
        width_in = width_pt * fraction * inches_per_pt

        # Preserve original AR if not specified
        if aspect_ratio is None:
            w0, h0 = fig.get_size_inches()
            aspect_ratio = h0 / w0

        height_in = width_in * aspect_ratio
        # Set figure size
        fig.set_size_inches(width_in, height_in)
    
    def save_figure(self, fig, filename="figure", dir="figures"):
        """Function to save figure with consistent formatting (pdf).
        Parameters:
        - fig: Matplotlib figure object to save.
        - filename: Name of the file to save the figure as (without extension).
        - dir: Directory to save the figure in.
        Returns:
        - None: Saves the figure to disk with the specified filename and consistent formatting.
        """
        # Check if the filename already has an extension, and remove it
        parts = filename.split('.')
        filename = parts[0]
        # Save the figure as a PDF with the specified filename
        fig.savefig(os.path.join(dir, f'{parts[0]}.pdf'), bbox_inches="tight")