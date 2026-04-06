
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

class PlotSettings():
    """
    Class to set global plot styles and provide a function for consistent plot formatting.
    """

    def __init__(self):
        pass

    def set_global_style(self, latex=False, base_font_size=10):
        """Function to set global plot styles for consistent and publication-quality figures.
        Parameters:
        - latex: Boolean indicating whether to use LaTeX for text rendering.
        - base_font_size: Base font size for all text elements in the plot.
        Returns:
        - None: Updates Matplotlib rcParams for consistent styling across all plots.
        """
        # Update Matplotlib rcParams for consistent styling
        plt.rcParams.update({
            "text.usetex": latex,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": base_font_size,
            "axes.labelsize": base_font_size,
            "axes.titlesize": base_font_size,
            "legend.fontsize": base_font_size * 0.8,
            "xtick.labelsize": base_font_size * 0.8,
            "ytick.labelsize": base_font_size * 0.8,
            "axes.linewidth": base_font_size * 0.08,
            "figure.dpi": 200,
            "savefig.dpi": 300
        })


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
            ax.tick_params(which='major', length=6, width=0.8)
            ax.tick_params(which='minor', length=3, width=0.8)
            # Change spine widths
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

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
            ax.tick_params(which='major', length=6, width=0.8)
            ax.tick_params(which='minor', length=3, width=0.8)

        # Optional grid
        if gridlines:
            ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.7)
            ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.4)

    
    def set_size(self, fig, width=1, aspect_ratio=None):

        width_pt=369.0
        inches_per_pt = 1 / 72.27

        # Total figure width (fixed)
        fig_width_in = width_pt * inches_per_pt

        # Graphics width
        graphics_width_in = fig_width_in * width

        # Preserve original AR if not specified
        if aspect_ratio is None:
            w0, h0 = fig.get_size_inches()
            aspect_ratio = h0 / w0

        # Graphics height from aspect ratio
        graphics_height_in = graphics_width_in * aspect_ratio

        # Figure height equals graphics height (no vertical whitespace)
        fig_height_in = graphics_height_in

        fig.set_size_inches(fig_width_in, fig_height_in)

        # Center graphics horizontally
        margin = (1 - width) / 2
        fig.tight_layout(rect=[margin, 0, 1 - margin, 1])
    


    def create_figure(self, width=0.7, style='default', minor=True, gridlines=False):
        """Function to create a Matplotlib figure and axis with consistent styling.
        Parameters:
        - width: Fraction of the target width for the figure (default is 0.7).
        - style: String specifying the style to apply to the axis ('default', 'minimalist', 'bands').
        - minor: Boolean indicating whether to show minor ticks.
        - gridlines: Boolean indicating whether to show gridlines.
        Returns:
        - fig: Matplotlib figure object with the specified size and styling.
        - ax: Matplotlib axis object with the specified styling.
        """
        fig, ax = plt.subplots()
        self.set_size(fig, width=width)
        self.set_style_ax(ax, style=style, minor=minor, gridlines=gridlines)
        return fig, ax


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