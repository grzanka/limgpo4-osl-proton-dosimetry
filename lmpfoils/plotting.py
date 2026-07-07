"""Diagnostic and article-style plotting helpers.

Ported from ``ifj_lmp_foils/src/visualisation/plot.py`` and
``ifj_lmp_foils/src/visualisation/visualize.py``.
"""
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle as CirclePatch
from scipy import ndimage

from lmpfoils.geometry import Circle, create_circular_mask, get_mean_std


def apply_mpl_settings():
    """Apply the matplotlib rcParams used throughout the article figures."""
    mpl.rcParams['axes.grid.which'] = 'minor'
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['figure.figsize'] = (10, 12)
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['ytick.minor.visible'] = True


def apply_grid(ax):
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    ax.grid(which='major', linestyle='-', linewidth='0.7', color='black', alpha=0.7)


def apply_lim(ax):
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)


def apply_defaults(ax):
    apply_grid(ax)
    apply_lim(ax)


def label_text(data: npt.NDArray, title: str) -> str:
    result = '\n'.join((
        title,
        f'mean = {np.nanmean(data):.3f}',
        f'median = {np.nanmedian(data):.3f}',
        '\n',
        f'stddev = {np.nanstd(data):.3f}',
        f'stddev / mean = {100. * np.nanstd(data) / np.nanmean(data):.3f} %',
        '\n',
        f'min = {np.nanmin(data):.3f}',
        f'max = {np.nanmax(data):.3f}',
    ))
    return result


def min_max_area_loc(data: npt.NDArray, circle_px: Circle, window_size: int = 10) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    N, M = window_size, window_size
    P, Q = data.shape
    mask_for_circle = create_circular_mask(img=data, circle_px=circle_px)

    meds = ndimage.median_filter(data, size=(M, N))
    meds[~mask_for_circle] = np.nan
    data_medfilt = meds[M // 2:(M // 2) + P - M + 1, N // 2:(N // 2) + Q - N + 1]

    max_idx = np.unravel_index(np.nanargmax(data_medfilt), data_medfilt.shape)
    max_center = max_idx[0] + window_size, max_idx[1] + window_size

    min_idx = np.unravel_index(np.nanargmin(data_medfilt), data_medfilt.shape)
    min_center = min_idx[0] + window_size, min_idx[1] + window_size

    return (min_center, max_center)


def plot_2d(data: npt.NDArray,
            circle: Optional[Circle] = None,
            angle_deg: Optional[float] = None,
            ax: Optional[plt.Axes] = None,
            a_min: Optional[float] = None,
            a_max: Optional[float] = None,
            cbar: bool = True,
            **kwargs) -> plt.Axes:
    """Quick 2D image view with an optional overlaid detector circle,
    orientation line and image-center marker.

    Used across the pipeline notebooks to inspect the effect of each
    processing step (raw -> background-subtracted -> flat-field corrected ->
    centered -> rotated). Ported from ``4-from_raw_to_signal.ipynb``.
    """
    if ax is None:
        _, ax = plt.subplots()
    current_a_min = a_min
    if a_min is None:
        current_a_min = np.nanpercentile(a=data, q=3)
    current_a_max = a_max
    if a_max is None:
        current_a_max = np.nanpercentile(a=data, q=97)
    im = ax.imshow(np.clip(data, a_min=current_a_min, a_max=current_a_max), cmap="terrain", **kwargs)
    if cbar:
        ax.figure.colorbar(im, ax=ax)
    if circle is not None:
        ax.add_artist(CirclePatch((circle.x, circle.y), circle.r, fill=False, color="red"))
        ax.plot(circle.x, circle.y, "ro")
        if angle_deg is not None:
            angle_rad = np.deg2rad(angle_deg)
            radius_line = Line2D([circle.x, circle.x - circle.r * np.sin(angle_rad)],
                                  [circle.y, circle.y - circle.r * np.cos(angle_rad)],
                                  color="red")
            ax.add_artist(radius_line)
    ax.plot(data.shape[1] / 2, data.shape[0] / 2, "bo")
    ax.grid()
    return ax


def plot_data(data: npt.NDArray, path: str, circle_px: Optional[Circle] = None, details: bool = False):
    """Diagnostic plot of a single foil image: 2D map plus (optionally)
    orthogonal profile sections and a histogram of pixel values inside the
    detected circle."""
    circle = circle_px
    if not circle:
        circle = Circle(x=data.shape[0] / 2, y=data.shape[0] / 2, r=250)

    axes = [[]]
    if details:
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(16, 10), constrained_layout=True)
        ax = axes[0][0]
    else:
        fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=True)

    # don't plot top 5% to avoid hot pixels
    data_for_plotting = np.clip(data, a_min=None, a_max=np.nanpercentile(a=data, q=95))
    pos0 = ax.imshow(data_for_plotting, cmap='terrain', interpolation='None')
    plt.colorbar(pos0, ax=ax, shrink=0.4)

    mask_for_circle = create_circular_mask(img=data, circle_px=circle)
    title_for_circle = f'circle at {circle.x},{circle.y}\n     radius {circle.r:.1f} : \n'
    text_for_circle = label_text(data=data[mask_for_circle], title=title_for_circle)

    title_for_image = 'full image : \n\n'
    text_for_image = label_text(data=data, title=title_for_image)

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    if not details:
        # place a text box for circular area
        ax.text(0.15, 0.95, text_for_circle, transform=ax.transAxes, fontsize=14,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=props)

        # place a text box for image area
        ax.text(0.65, 0.95, text_for_image, transform=ax.transAxes, fontsize=14,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=props)

    ax.add_artist(plt.Circle(xy=(circle.x, circle.y),
                              radius=circle.r,
                              color='black',
                              fill=False,
                              transform=ax.transData))

    mean_in_circle = np.nanmean(data[mask_for_circle])
    std_in_circle = np.nanstd(data[mask_for_circle])

    bins = np.arange(start=int(mean_in_circle - 5 * std_in_circle), stop=int(mean_in_circle + 5 * std_in_circle), step=1)
    if bins.size < 30:
        bins = np.linspace(start=mean_in_circle - 5 * std_in_circle, stop=mean_in_circle + 5 * std_in_circle, num=100)

    if details:
        axes[1][1].hist(data[mask_for_circle].flatten(), bins=bins, histtype='step', label='original')
        axes[1][1].grid()

        section_cut_x = circle.x + circle.r / 2
        section_cut_y = circle.y + circle.r / 2
        window_size = 10
        min_center, max_center = min_max_area_loc(data, circle_px=circle, window_size=window_size)
        axes[0][0].set_xlabel("X")
        axes[0][0].set_ylabel("Y")

        axes[0][0].axvline(section_cut_x, color='blue')
        axes[0][0].axvline(min_center[0], color='green')
        axes[0][0].axvline(max_center[0], color='red')

        axes[0][0].axhline(section_cut_y, color='blue')
        axes[0][0].axhline(min_center[1], color='green')
        axes[0][0].axhline(max_center[1], color='red')

        def section(data, location, window_size, axis: int, circle_px: Optional[Circle] = None):
            start = int(location - window_size / 2)
            stop = int(location + window_size / 2)
            if axis == 1:  # along Y, fixed X
                result = np.average(data[:, start:stop], axis=1)
                if circle_px:
                    lower_y, upper_y = circle.section_x(x=0.5 * (start + stop))
                    result[0:int(lower_y)] = np.nan
                    result[int(upper_y):] = np.nan
            else:  # along X, fixed Y
                result = np.average(data[start:stop, :], axis=0)
                if circle_px:
                    lower_x, upper_x = circle.section_y(y=0.5 * (start + stop))
                    result[0:int(lower_x)] = np.nan
                    result[int(upper_x):] = np.nan
            return result

        def abs2perc(x):
            return 100. - 100. * x / mean_in_circle

        def perc2abs(x):
            return mean_in_circle * (100. - x) / 100.

        axes[0][1].plot(section(data, section_cut_x, window_size, 1, circle), label='ref', color='blue')
        axes[0][1].plot(section(data, min_center[0], window_size, 1, circle), label='min', color='green')
        axes[0][1].plot(section(data, max_center[0], window_size, 1, circle), label='max', color='red')
        axes[0][1].axvline(section_cut_y, color='blue')
        axes[0][1].axvline(min_center[1], color='green')
        axes[0][1].axvline(max_center[1], color='red')
        axes[0][1].set_xlabel("Y")
        secax = axes[0][1].secondary_yaxis('right', functions=(abs2perc, perc2abs))
        secax.set_ylabel(' (value-mean)/mean [%] ')
        axes[0][1].grid()
        axes[0][1].legend()

        axes[1][0].plot(section(data, section_cut_y, window_size, 0, circle), label='ref', color='blue')
        axes[1][0].plot(section(data, min_center[1], window_size, 0, circle), label='min', color='green')
        axes[1][0].plot(section(data, max_center[1], window_size, 0, circle), label='max', color='red')
        axes[1][0].axvline(section_cut_x, color='blue')
        axes[1][0].axvline(min_center[0], color='green')
        axes[1][0].axvline(max_center[0], color='red')
        axes[1][0].set_xlabel("X")
        secax = axes[1][0].secondary_yaxis('right', functions=(abs2perc, perc2abs))
        secax.set_ylabel(' (value-mean)/mean [%] ')
        axes[1][0].grid()
        axes[1][0].legend()

    if path:
        fig.savefig(path)

    if details:
        return fig, axes
    return fig, ax
