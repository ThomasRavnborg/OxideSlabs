import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import string

class LatexFigure:
    """
    Publication-ready figure class for LaTeX:
    - Handles LaTeX column width scaling
    - Supports internal axes fraction (white margins)
    - Automatic panel labels (a), (b), ...
    - Subplots with sharex/sharey, width_ratios, etc.
    """

    def __init__(self,
                 width_pt=369,       # LaTeX column width in points
                 base_font_size=9,
                 latex=False,
                 aspect_ratio=0.62):
        self.width_pt = width_pt
        self.base_font_size = base_font_size
        self.latex = latex
        self.aspect_ratio = aspect_ratio
        self._set_global_style()

    def _set_global_style(self):
        plt.rcParams.update({
            "text.usetex": self.latex,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": self.base_font_size,
            "axes.titlesize": self.base_font_size,
            "axes.labelsize": self.base_font_size * 0.9,
            "legend.fontsize": self.base_font_size * 0.8,
            "xtick.labelsize": self.base_font_size * 0.8,
            "ytick.labelsize": self.base_font_size * 0.8,
            "axes.linewidth": 0.8,
            "figure.dpi": 200,
            "savefig.dpi": 300
        })

    def _figsize(self):
        """Figure size in inches for LaTeX width."""
        inches_per_pt = 1 / 72.27
        width_in = self.width_pt * inches_per_pt
        height_in = width_in * self.aspect_ratio
        return width_in, height_in

    def create(self,
               nrows=1,
               ncols=1,
               fraction=1.0,            # fraction of figure width for axes
               height_fraction=None,     # fraction of figure height for axes
               style="default",
               minor=True,
               grid=False,
               add_labels=False,
               label_fontsize=None,
               **subplot_kwargs):
        """
        Create a figure with consistent LaTeX sizing.
        fraction < 1 adds horizontal white margins
        height_fraction < 1 adds vertical white margins
        """
        fig_width, fig_height = self._figsize()
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(fig_width, fig_height),
            constrained_layout=False,  # Turn off constrained_layout
            **subplot_kwargs
        )

        # Apply horizontal/vertical margins
        left = (1 - fraction) / 2
        right = 1 - left
        if height_fraction is not None:
            bottom = (1 - height_fraction) / 2
            top = 1 - bottom
        else:
            bottom, top = 0.15, 0.95

        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

        # Style axes
        self._style_all_axes(axes, style, minor, grid)

        # Add panel labels
        if add_labels:
            self._add_panel_labels(axes, fontsize=label_fontsize)

        return fig, axes

    def _style_all_axes(self, axes, style, minor, grid):
        if not hasattr(axes, "flat"):
            axes = [axes]
        else:
            axes = axes.flat
        for ax in axes:
            self.style_axis(ax, style=style, minor=minor, grid=grid)

    def style_axis(self, ax, style="default", minor=True, grid=False):
        if minor:
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        if style == "default":
            ax.tick_params(which='both', direction='in', top=True, right=True)
            ax.tick_params(which='major', length=6, width=0.8)
            ax.tick_params(which='minor', length=3, width=0.8)
        elif style == "minimalist":
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(which='both', top=False, right=False)

        if grid:
            ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.7)
            ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.4)

    def _add_panel_labels(self, axes, fontsize=None, offset=(0.02, 0.95)):
        if not hasattr(axes, "flat"):
            axes = [axes]
        else:
            axes = axes.flat

        if fontsize is None:
            fontsize = self.base_font_size * 1.1

        for i, ax in enumerate(axes):
            label = f"({string.ascii_lowercase[i]})"
            ax.text(
                offset[0], offset[1],
                label,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight='bold',
                va='top',
                ha='left'
            )

    def save(self, fig, filename="figure", dir="figures"):
        os.makedirs(dir, exist_ok=True)
        filename = os.path.splitext(filename)[0]
        fig.savefig(os.path.join(dir, f"{filename}.pdf"))