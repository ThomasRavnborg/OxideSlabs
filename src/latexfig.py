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
                ax.tick_params(which='both', direction='out', pad = 4)
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
    def __init__(self, textwidth_pt=369, aspect_ratio=0.62, font_size=9, latex=False):
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

    def create(self, nrows=1, ncols=1, scale=1.0, sharex=False, sharey=False,
               grid=False, minor=True, panel_labels=False, **kwargs):

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=self._figsize(),
            sharex=sharex,
            sharey=sharey,
            constrained_layout=False,
            **kwargs
        )

        self._style_axes(axes, grid, minor)
        if panel_labels:
            self._add_panel_labels(axes)

        # --- force draw to compute text sizes ---
        fig.canvas.draw()

        # --- compute axes bounding boxes ---
        axes_list = [axes] if not hasattr(axes, "flat") else list(axes.flat)

        # find tight bounding box of all axes including labels/ticks
        x0 = min(ax.get_position().x0 for ax in axes_list)
        y0 = min(ax.get_position().y0 for ax in axes_list)
        x1 = max(ax.get_position().x1 for ax in axes_list)
        y1 = max(ax.get_position().y1 for ax in axes_list)

        width = x1 - x0
        height = y1 - y0
        cx = x0 + width / 2
        cy = y0 + height / 2

        # --- scale axes box uniformly ---
        new_width = width * scale
        new_height = height * scale
        new_x0 = cx - new_width / 2
        new_y0 = cy - new_height / 2

        # reposition each axis proportionally
        for ax in axes_list:
            pos = ax.get_position()
            rel_x = (pos.x0 - x0) / width
            rel_y = (pos.y0 - y0) / height
            rel_w = pos.width / width
            rel_h = pos.height / height
            ax.set_position([
                new_x0 + rel_x * new_width,
                new_y0 + rel_y * new_height,
                rel_w * new_width,
                rel_h * new_height
            ])

        return fig, axes

    def _style_axes(self, axes, grid, minor):
        axes_list = [axes] if not hasattr(axes, "flat") else axes.flat
        for ax in axes_list:
            if minor:
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
            if grid:
                ax.grid(True, which="major", ls="--", lw=0.6, alpha=0.7)
                ax.grid(True, which="minor", ls=":", lw=0.5, alpha=0.5)

    def _add_panel_labels(self, axes, offset=(0.02, 0.95)):
        axes_list = [axes] if not hasattr(axes, "flat") else axes.flat
        for i, ax in enumerate(axes_list):
            label = f"({string.ascii_lowercase[i]})"
            ax.text(offset[0], offset[1], label,
                    transform=ax.transAxes,
                    fontweight="bold",
                    va="top", ha="left")

    def save(self, fig, name="figure", folder="figures"):
        os.makedirs(folder, exist_ok=True)
        fig.savefig(os.path.join(folder, f"{name}.pdf"))



import os
import string
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


class LatexFigure4:

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

        # --- compute natural layout ---
        fig.canvas.draw()
        fig.tight_layout(pad=0.3)

        if scale < 1:

            axes_list = [axes] if not hasattr(axes, "flat") else list(axes.flat)

            # --- find bounding box of entire axes grid ---
            x0 = min(ax.get_position().x0 for ax in axes_list)
            y0 = min(ax.get_position().y0 for ax in axes_list)

            x1 = max(ax.get_position().x1 for ax in axes_list)
            y1 = max(ax.get_position().y1 for ax in axes_list)

            width = x1 - x0
            height = y1 - y0

            # --- scaled box ---
            new_width = width * scale
            new_height = height * scale

            cx = x0 + width / 2
            cy = y0 + height / 2

            new_x0 = cx - new_width / 2
            new_y0 = cy - new_height / 2

            # --- reposition each axis relative to scaled box ---
            for ax in axes_list:

                pos = ax.get_position()

                rel_x = (pos.x0 - x0) / width
                rel_y = (pos.y0 - y0) / height

                rel_w = pos.width / width
                rel_h = pos.height / height

                ax.set_position([
                    new_x0 + rel_x * new_width,
                    new_y0 + rel_y * new_height,
                    rel_w * new_width,
                    rel_h * new_height
                ])

        return fig, axes

    def _style_axes(self, axes, grid, minor):

        axes_list = [axes] if not hasattr(axes, "flat") else axes.flat

        for ax in axes_list:

            if minor:
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())

            if grid:
                ax.grid(True, which="major", ls="--", lw=0.6, alpha=0.7)
                ax.grid(True, which="minor", ls=":", lw=0.5, alpha=0.5)

    def _add_panel_labels(self, axes, offset=(0.02, 0.95)):

        axes_list = [axes] if not hasattr(axes, "flat") else axes.flat

        for i, ax in enumerate(axes_list):

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


class LatexFigureSimple:
    """
    Simple, robust LaTeX-ready figure class.

    - Width always matches LaTeX \textwidth
    - Preserves aspect ratio
    - Axes + labels scale with 'scale'
    - Optional horizontal padding
    """

    def __init__(self, textwidth_pt=369, aspect_ratio=0.62, font_size=9, latex=False):
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

    def create(self, nrows=1, ncols=1, scale=1.0, sharex=False, sharey=False,
               grid=False, minor=True, panel_labels=False, hpad=0.0, **kwargs):
        """
        Create a figure with proper LaTeX sizing.

        scale: float <1 to shrink axes content uniformly
        hpad: fraction of width to add as horizontal padding (white space)
        """

        fig, axes = plt.subplots(nrows, ncols, figsize=self._figsize(),
                                 sharex=sharex, sharey=sharey,
                                 constrained_layout=False, **kwargs)

        # Flatten axes
        axes_list = [axes] if not hasattr(axes, "flat") else list(axes.flat)

        # Style axes
        for ax in axes_list:
            if minor:
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
            if grid:
                ax.grid(True, which="major", ls="--", lw=0.6, alpha=0.7)
                ax.grid(True, which="minor", ls=":", lw=0.5, alpha=0.5)

        # Panel labels
        if panel_labels:
            for i, ax in enumerate(axes_list):
                ax.text(0.02, 0.95, f"({string.ascii_lowercase[i]})",
                        transform=ax.transAxes,
                        fontweight="bold", va="top", ha="left")

        # --- Draw canvas to compute positions ---
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # Compute the tight bounding box of the entire figure including all axes, labels, ticks
        tight_boxes = [ax.get_tightbbox(renderer) for ax in axes_list]
        x0 = min(bbox.x0 for bbox in tight_boxes)
        x1 = max(bbox.x1 for bbox in tight_boxes)
        y0 = min(bbox.y0 for bbox in tight_boxes)
        y1 = max(bbox.y1 for bbox in tight_boxes)

        bbox_width = x1 - x0
        bbox_height = y1 - y0
        center_x = x0 + bbox_width / 2
        center_y = y0 + bbox_height / 2

        # Scale the tight box
        new_width = bbox_width * scale
        new_height = bbox_height * scale

        new_x0 = center_x - new_width / 2
        new_y0 = center_y - new_height / 2

        # Figure dimensions in pixels
        fig_width_px, fig_height_px = fig.get_size_inches() * fig.dpi

        # Reposition each axis proportionally inside the scaled tight box
        for ax, bbox in zip(axes_list, tight_boxes):
            rel_x0 = (bbox.x0 - x0) / bbox_width
            rel_y0 = (bbox.y0 - y0) / bbox_height
            rel_w = bbox.width / bbox_width
            rel_h = bbox.height / bbox_height

            ax.set_position([
                (new_x0 + rel_x0 * new_width) / fig_width_px,
                (new_y0 + rel_y0 * new_height) / fig_height_px,
                rel_w * new_width / fig_width_px,
                rel_h * new_height / fig_height_px
            ])

        # Add horizontal padding if requested
        if hpad > 0:
            for ax in axes_list:
                pos = ax.get_position()
                pos.x0 += hpad / 2
                pos.x1 -= hpad / 2
                ax.set_position(pos)

        return fig, axes

    def save(self, fig, name="figure", folder="figures"):
        os.makedirs(folder, exist_ok=True)
        fig.savefig(os.path.join(folder, f"{name}.pdf"))