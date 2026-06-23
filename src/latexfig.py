import os
import string
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

class LatexFigure:
    """
    Helper class for publication-ready matplotlib figures.

    Workflow:
    - Figures generated at full LaTeX textwidth
    - LaTeX controls scaling
    - Fonts remain consistent
    """

    def __init__(self, textwidth_pt=369, font_size=9, render_latex=False):

        self.textwidth_pt = textwidth_pt
        self.font_size = font_size
        self.latex = render_latex

        self._set_style()

    def _set_style(self):

        plt.rcParams.update({
            "text.usetex": self.latex,

            "font.family": "serif",
            "mathtext.fontset": "cm",

            "font.size": self.font_size,
            "axes.labelsize": self.font_size,
            "axes.titlesize": self.font_size,

            "legend.fontsize": self.font_size * 0.9,
            "xtick.labelsize": self.font_size * 0.9,
            "ytick.labelsize": self.font_size * 0.9,

            "axes.linewidth": 0.8,

            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,

            "figure.dpi": 200,
            "savefig.dpi": 300,
        })

    def _figsize(self, width, nrows, ncols, aspect_ratio):

        inches_per_pt = 1 / 72.27
        width_in = self.textwidth_pt * inches_per_pt * width
        height_in = width_in * aspect_ratio * nrows / ncols

        return width_in, height_in

    def _style_axes(self, axes, style, grid, minor):

        for ax in axes:

            if style == 'default':
                # Ticks on all sides, pointing inwards, with specific lengths and widths
                ax.tick_params(which='both', direction='in', top=True, right=True)
                #ax.tick_params(which='major', length=6, width=0.8)
                #ax.tick_params(which='minor', length=3, width=0.8)
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
                ax.tick_params(which='both', direction='out', top=False, right=False)
                #ax.tick_params(which='major', length=6, width=0.8)
                #ax.tick_params(which='minor', length=3, width=0.8)

            if minor:
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))

            if grid:
                ax.grid(True, which="major", ls="--", lw=0.6, alpha=0.7)
                ax.grid(True, which="minor", ls=":", lw=0.5, alpha=0.5)
            
        

    def _add_panel_labels(self, axes, offset=(0.02, 0.95)):

        for i, ax in enumerate(axes):

            label = f"({string.ascii_lowercase[i]})"

            ax.text(
                offset[0],
                offset[1],
                label,
                transform=ax.transAxes,
                fontweight="bold",
                va="top",
                ha="left",
            )

    def create(self, width=1.0, AR=0.62, subplots=(1, 1), style='default', grid=False, minor=True, panel_labels=False, **kwargs):
        """Function to create a figure and style the axes.
        Arguments:
        - width: fraction of LaTeX textwidth to use for figure width (e.g. 0.8 for 80% of textwidth)
        - AR: the aspect ratio of the figure
        - subplots: tuple of (nrows, ncols) for the number of subplots
        - style: the style to apply to the axes
        - grid: whether to show grid lines
        - minor: whether to show minor ticks
        - panel_labels: whether to add panel labels (a), (b), etc.
        - kwargs: additional keyword arguments to pass to plt.subplots()
        Returns:
        - fig: the created figure object
        - axes: the created axes object(s)
        """

        if width <= 0 or width > 1:
            raise ValueError("Width fraction must be between 0 and 1")

        fig, axes = plt.subplots(*subplots,
                                 figsize=self._figsize(width, *subplots, aspect_ratio=AR),
                                 constrained_layout=True, **kwargs)
        
        axes = np.atleast_1d(axes).ravel()

        self._style_axes(axes, style, grid, minor)

        if panel_labels:
            self._add_panel_labels(axes)

        return fig, axes

    def save(self, fig, filename="figure", dir="figures"):

        os.makedirs(dir, exist_ok=True)

        path = os.path.join(dir, f"{filename}.pdf")

        fig.savefig(path, bbox_inches="tight")

