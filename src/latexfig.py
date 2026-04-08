import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import string
import os
import string
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

    def _figsize(self, aspect_ratio=0.62):

        inches_per_pt = 1 / 72.27
        width = self.textwidth_pt * inches_per_pt
        height = width * aspect_ratio

        return width, height

    def _style_axes(self, axes, grid, minor):

        if not hasattr(axes, "flat"):
            axes = [axes]
        else:
            axes = axes.flat

        for ax in axes:

            if minor:
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())

            if grid:
                ax.grid(True, which="major", ls="--", lw=0.6, alpha=0.7)
                ax.grid(True, which="minor", ls=":", lw=0.5, alpha=0.5)

    def _add_panel_labels(self, axes, offset=(0.02, 0.95)):

        if not hasattr(axes, "flat"):
            axes = [axes]
        else:
            axes = axes.flat

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

    def create(self, subplots=(1, 1), grid=False, minor=True, panel_labels=False, **kwargs):
        """Function to create a figure and style the axes.
        Arguments:
        - subplots: tuple of (nrows, ncols) for the number of subplots
        - grid: whether to show grid lines
        - minor: whether to show minor ticks
        - panel_labels: whether to add panel labels (a), (b), etc.
        - kwargs: additional keyword arguments to pass to plt.subplots()
        Returns:
        - fig: the created figure object
        - axes: the created axes object(s)
        """

        fig, axes = plt.subplots(*subplots, figsize=self._figsize(), constrained_layout=True, **kwargs)

        self._style_axes(axes, grid, minor)

        if panel_labels:
            self._add_panel_labels(axes)

        return fig, axes

    def save(self, fig, filename="figure", dir="figures"):

        os.makedirs(dir, exist_ok=True)

        path = os.path.join(dir, f"{filename}.pdf")

        fig.savefig(path, bbox_inches="tight")




import os
import string
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


class LatexFigure2:

    def __init__(
        self,
        textwidth_pt=369,
        font_size=9,
        latex=False
    ):

        self.textwidth_pt = textwidth_pt
        self.font_size = font_size
        self.latex = latex

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

    def _figsize(self, aspect_ratio=0.62):

        inches_per_pt = 1 / 72.27

        width = self.textwidth_pt * inches_per_pt
        height = width * aspect_ratio

        return width, height

    def create(
        self,
        subplots=(1, 1),
        width=1.0, aspect_ratio=0.62,
        sharex=False,
        sharey=False,
        grid=False,
        minor=True,
        panel_labels=False,
        **kwargs
    ):

        fig, axes = plt.subplots(*subplots, figsize=self._figsize(aspect_ratio),
            sharex=sharex,
            sharey=sharey,
            constrained_layout=False,
            **kwargs
        )

        # compute margins so axes shrink but figure stays full width
        left = (1 - width) / 2
        right = 1 - left

        bottom = (1 - width) / 2
        top = 1 - bottom

        fig.subplots_adjust(
            left=left,
            right=right,
            bottom=bottom,
            top=top
        )

        self._style_axes(axes, grid, minor)

        if panel_labels:
            self._add_panel_labels(axes)

        return fig, axes

    def _style_axes(self, axes, grid, minor):

        if not hasattr(axes, "flat"):
            axes = [axes]
        else:
            axes = axes.flat

        for ax in axes:

            if minor:
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())

            if grid:
                ax.grid(True, which="major", ls="--", lw=0.6, alpha=0.7)
                ax.grid(True, which="minor", ls=":", lw=0.5, alpha=0.5)

    def _add_panel_labels(self, axes, offset=(0.02, 0.95)):

        if not hasattr(axes, "flat"):
            axes = [axes]
        else:
            axes = axes.flat

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

    def save(self, fig, filename="figure", dir="figures"):

        os.makedirs(dir, exist_ok=True)

        fig.savefig(
            os.path.join(dir, f"{filename}.pdf"),
            bbox_inches=None
        )


class LatexFigure3:

    def __init__(
        self,
        textwidth_pt=369,
        aspect_ratio=0.62,
        font_size=9,
        latex=False
    ):

        self.textwidth_pt = textwidth_pt
        self.aspect_ratio = aspect_ratio
        self.font_size = font_size
        self.latex = latex

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

    def _figsize(self):

        inches_per_pt = 1 / 72.27

        width = self.textwidth_pt * inches_per_pt
        height = width * self.aspect_ratio

        return width, height

    def create(
        self,
        nrows=1,
        ncols=1,
        scale=1.0,
        sharex=False,
        sharey=False,
        grid=False,
        minor=True,
        panel_labels=False,
        **kwargs
    ):

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=self._figsize(),
            sharex=sharex,
            sharey=sharey,
            constrained_layout=False,
            **kwargs
        )

        self._style_axes(axes, grid, minor)

        if panel_labels:
            self._add_panel_labels(axes)

        # --- Step 1: compute natural margins ---
        fig.canvas.draw()
        fig.tight_layout()

        left = fig.subplotpars.left
        right = fig.subplotpars.right
        bottom = fig.subplotpars.bottom
        top = fig.subplotpars.top

        # --- Step 2: shrink axes horizontally ---
        if scale < 1:

            width = right - left
            new_width = width * scale

            center = (left + right) / 2

            left = center - new_width / 2
            right = center + new_width / 2

        fig.subplots_adjust(
            left=left,
            right=right,
            bottom=bottom,
            top=top
        )

        return fig, axes

    def _style_axes(self, axes, grid, minor):

        if not hasattr(axes, "flat"):
            axes = [axes]
        else:
            axes = axes.flat

        for ax in axes:

            if minor:
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())

            if grid:
                ax.grid(True, which="major", ls="--", lw=0.6, alpha=0.7)
                ax.grid(True, which="minor", ls=":", lw=0.5, alpha=0.5)

    def _add_panel_labels(self, axes, offset=(0.02, 0.95)):

        if not hasattr(axes, "flat"):
            axes = [axes]
        else:
            axes = axes.flat

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

    def save(self, fig, name="figure", folder="figures"):

        os.makedirs(folder, exist_ok=True)

        fig.savefig(
            os.path.join(folder, f"{name}.pdf")
        )