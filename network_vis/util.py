#! /usr/bin/env python
"""
Author: LiangLiang ZHENG
Date:
graph_helper, plotting output of the network
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from termcolor import colored
from scipy.stats import gaussian_kde
import pandas as pd
from copy import deepcopy
from termcolor import colored
import numpy as np

note_str = colored("NOTE: ", 'blue')
warn_str = colored("WARNING: ", 'red')


def selectunits(data, units=32, is_bilstm = False):
    """Select random number of unit in data
    """
    data = [data[0]]
    print(np.shape(data))
    batch_num, max_timesteps, units_all = np.shape(data)
    print(np.shape(data))
    np.random.seed(0)
    res = np.zeros((1, max_timesteps, units))
    if is_bilstm == True:
        # still have bug
        l = list(range(units//2))
        np.random.shuffle(l)
        sub_data1 = data[:, 0:units//2]
        sub_data2 = data[:, units//2:]
        res1 = sub_data1[:, l]
        res2 = sub_data2[:, l]
        res = np.concatenate((res1, res2), axis = 1)
    else:
        l = np.random.choice(units_all, units)
        for i in range(len(data)):
            res[i]  = data[i][:, l]

    return res

def show_scatter_density(data, units):
    """Draw th scatter density of the neuron
    Argument:
        data: the shape should be [max_timesteps, units]
    """
    data = data[0]
    print(np.shape(data))
    fig, ax = plt.subplots()

    for i in range(units):
        z = gaussian_kde(data[:, i])(data[:, i])
        x = [i+1] * data.shape[0]
        a = ax.scatter(x, data[:, i], c=z, s=100, edgecolor='')

    plt.colorbar(a)
    plt.xlabel('Selected units')
    plt.ylabel('Activation')
    plt.show()

def show_box_plot(data, units):
    data = data[0]
    df = pd.DataFrame(data)
    boxplot = df.boxplot()

def show_features_0D(data, marker='o', cmap='bwr', color=None, **kwargs):
    """Plots 0D aligned scatterplots in a standalone graph.

    iter == list/tuple (both work)

    Arguments:
        data: np.ndarray, 2D: (samples, channels).
        marker: str. Pyplot kwarg specifying scatter plot marker shape.
        cmap: str. Pyplot cmap (colormap) kwarg for the heatmap. Overridden
              by `color`!=None.
        color: (float iter) iter / str / str iter / None. Pyplot kwarg,
              specifying marker colors in order of drawing. If str/ float iter,
              draws all curves in one color. Overrides `cmap`. If None,
              automatically colors along equally spaced `cmap` gradient intervals.
              Ex: ['red', 'blue']; [[0., .8, 1.], [.2, .5, 0.]] (RGB)

    kwargs:
        scale_width:   float. Scale width  of resulting plot by a factor.
        scale_height:  float. Scale height of resulting plot by a factor.
        show_borders:  bool.  If True, shows boxes around plot(s).
        title_mode:    bool/str. If True, shows generic supertitle.
              If str in ('grads', 'outputs'), shows supertitle tailored to
              `data` dim (2D/3D). If other str, shows `title_mode` as supertitle.
              If False, no title is shown.
        show_y_zero: bool. If True, draws y=0.
        title_fontsize: int. Title fontsize.
        channel_axis: int. `data` axis holding channels/features. -1 = last axis.
        markersize:   int/int iter. Pyplot kwarg `s` specifying marker size(s).
        markerwidth:  int. Pyplot kwarg `linewidth` specifying marker thickness.
        ylims: str ('auto'); float list/tuple. Plot y-limits; if 'auto',
               sets both lims to max of abs(`data`) (such that y=0 is centered).
    """

    scale_width    = kwargs.get('scale_width',  1)
    scale_height   = kwargs.get('scale_height', 1)
    show_borders   = kwargs.get('show_borders', False)
    title_mode     = kwargs.get('title_mode',   'outputs')
    show_y_zero    = kwargs.get('show_y_zero',  True)
    title_fontsize = kwargs.get('title_fontsize', 14)
    markersize     = kwargs.get('markersize',  15)
    markerwidth    = kwargs.get('markerwidth', 2)
    ylims          = kwargs.get('ylims', 'auto')

    def _catch_unknown_kwargs(kwargs):
        allowed_kwargs = ('scale_width', 'scale_height', 'show_borders',
                          'title_mode', 'show_y_zero', 'title_fontsize',
                          'channel_axis', 'markersize', 'markerwidth', 'ylims')
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise Exception("unknown kwarg `%s`" % kwarg)

    def _get_title(data, title_mode):
        feature = "Context-feature"
        context = "Context-units"

        if title_mode in ['grads', 'outputs']:
            feature = "Gradients" if title_mode=='grads' else "Outputs"
            context = "Timesteps"
        return "(%s vs. %s) vs. Channels" % (feature, context)

    _catch_unknown_kwargs(kwargs)

    if len(data.shape)!=2:
        raise Exception("`data` must be 2D")

    if color is None:
        cmap = cm.get_cmap(cmap)
        cmap_grad = np.linspace(0, 256, len(data[0])).astype('int32')
        color = cmap(cmap_grad)
        color = np.vstack([color] * data.shape[0])
    x = np.ones(data.shape) * np.expand_dims(np.arange(1, len(data) + 1), -1)

    if show_y_zero:
        plt.axhline(0, color='k', linewidth=1)
    plt.scatter(x.flatten(), data.flatten(), marker=marker,
                s=markersize, linewidth=markerwidth, color=color)

    plt.gca().set_xticks(np.arange(1, len(data) + 1), minor=True)
    plt.gca().tick_params(which='minor', length=4)
    if ylims == 'auto':
        ymax = np.max(np.abs(data))
        ymin = -ymax
    else:
        ymin, ymax = ylims
    plt.gca().set_ylim(-ymax, ymax)

    if title_mode:
        title = _get_title(data, title_mode)
        plt.title(title, weight='bold', fontsize=title_fontsize)
    if not show_borders:
        plt.box(None)
    plt.gcf().set_size_inches(12*scale_width, 4*scale_height)
    plt.show()


def show_features_2D(data, n_rows=None, norm=None, cmap='bwr', reflect_half=False,
                     timesteps_xaxis=True, max_timesteps=None, **kwargs):
    """Plots 2D heatmaps in a standalone graph or subplot grid.

    iter == list/tuple (both work)

    Arguments:
        data: np.ndarray, 2D/3D. Data to plot.
              2D -> standalone graph; 3D -> subplot grid.
              3D: (samples, timesteps, channels)
              2D: (timesteps, channels)
        n_rows: int/None. Number of rows in subplot grid. If None,
              determines automatically, closest to n_rows == n_cols.
        norm: float iter. Normalizes colors to range between norm==(vmin, vmax),
              according to `cmap`. Ex: `cmap`='bwr' ('blue white red') -> all
              values <=vmin and >=vmax will be shown as most intense blue and
              red, and those exactly in-between are shown as white.
        cmap: str. Pyplot cmap (colormap) kwarg for the heatmap.
        reflect_half: bool. If True, second half of channels dim will be
              flipped about the timesteps dim.
        timesteps_xaxis: bool. If True, the timesteps dim (`data` dim 1)
              if plotted along the x-axis.
        max_timesteps:  int/None. Max number of timesteps to show per plot.
              If None, keeps original.

    kwargs:
        scale_width:   float. Scale width  of resulting plot by a factor.
        scale_height:  float. Scale height of resulting plot by a factor.
        show_borders:  bool.  If True, shows boxes around plot(s).
        show_xy_ticks: int/bool iter. Slot 0 -> x, Slot 1 -> y.
              Ex: [1, 1] -> show both x- and y-ticks (and their labels).
                  [0, 0] -> hide both.
        show_colorbar: bool. If True, shows one colorbar next to plot(s).
        title_mode:    bool/str. If True, shows generic supertitle.
              If str in ('grads', 'outputs'), shows supertitle tailored to
              `data` dim (2D/3D). If other str, shows `title_mode` as supertitle.
              If False, no title is shown.
        title_fontsize: int. Title fontsize.
        tight: bool. If True, plots compactly by removing subplot padding.
        channel_axis: int, 0 or -1. `data` axis holding channels/features.
              -1 --> (samples,  timesteps, channels)
              0  --> (channels, timesteps, samples)
        borderwidth: float / None. Width of subplot borders.
        dpi: int. Pyplot kwarg, 'dots per inch', specifying plot resolution
    """

    scale_width    = kwargs.get('scale_width',   1)
    scale_height   = kwargs.get('scale_height',  1)
    show_borders   = kwargs.get('show_borders',  True)
    show_xy_ticks  = kwargs.get('show_xy_ticks', [True, True])
    show_colorbar  = kwargs.get('show_colorbar', False)
    title_mode     = kwargs.get('title_mode',    'outputs')
    title_fontsize = kwargs.get('title_fontsize', 14)
    tight          = kwargs.get('tight', False)
    channel_axis   = kwargs.get('channel_axis', -1)
    borderwidth    = kwargs.get('borderwidth', None)
    dpi            = kwargs.get('dpi', 76)

    def _catch_unknown_kwargs(kwargs):
        allowed_kwargs = ('scale_width', 'scale_height', 'show_borders',
                          'show_xy_ticks', 'show_colorbar', 'title_mode',
                          'title_fontsize', 'channel_axis', 'tight',
                          'borderwidth', 'dpi')
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise Exception("unknown kwarg `%s`" % kwarg)

    def _get_title(data, title_mode, timesteps_xaxis, vmin, vmax):
        feature = "Context-feature"
        context = "Context-units"
        context_order = "(%s vs. Channels)" % context
        extra_dim = ""

        if title_mode in ['grads', 'outputs']:
            feature = "Gradients" if title_mode=='grads' else "Outputs"
            context = "Timesteps"
        if timesteps_xaxis:
            context_order = "(Channels vs. %s)" % context
        if len(data.shape)==3:
            extra_dim = ") vs. Samples"
            context_order = "(" + context_order

        norm_txt = "(%s, %s)" % (vmin, vmax) if (vmin is not None) else "auto"
        return "{} vs. {}{} -- norm={}".format(context_order, feature,
                                               extra_dim, norm_txt)

    def _process_data(data, max_timesteps, reflect_half,
                      timesteps_xaxis, channel_axis):
        if max_timesteps is not None:
            data = data[..., :max_timesteps, :]
        if reflect_half:
            data = data.copy()  # prevent passed array from changing
            half_chs = data.shape[-1]//2
            data[..., half_chs:] = np.flip(data[..., half_chs:], axis=0)
        if timesteps_xaxis:
            if len(data.shape) != 3:
                data = np.expand_dims(data, 0)
            data = np.transpose(data, (0, 2, 1))
        return data

    _catch_unknown_kwargs(kwargs)

    if len(data.shape) not in (2, 3):
        raise Exception("`data` must be 2D or 3D")
    data = _process_data(data, max_timesteps, reflect_half,
                         timesteps_xaxis, channel_axis)

    vmin, vmax = norm or (None, None)
    n_subplots = len(data) if len(data.shape)==3 else 1
    n_rows, n_cols = _get_nrows_and_ncols(n_rows, n_subplots)

    fig, axes = plt.subplots(n_rows, n_cols, dpi=dpi, sharex=True, sharey=True)
    axes = np.asarray(axes)

    if title_mode:
        title = _get_title(data, title_mode, timesteps_xaxis, vmin, vmax)
        y = .93 + .12 * tight
        plt.suptitle(title, weight='bold', fontsize=title_fontsize, y=y)

    for ax_idx, ax in enumerate(axes.flat):
        img = ax.imshow(data[ax_idx], cmap=cmap, vmin=vmin, vmax=vmax)
        if not show_xy_ticks[0]:
            ax.set_xticks([])
        if not show_xy_ticks[1]:
            ax.set_yticks([])
        ax.axis('tight')
        if not show_borders:
            ax.set_frame_on(False)

    if show_colorbar:
        fig.colorbar(img, ax=axes.ravel().tolist())

    if tight:
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    if borderwidth is not None:
        for ax in axes.flat:
            [s.set_linewidth(borderwidth) for s in ax.spines.values()]

    plt.gcf().set_size_inches(8*scale_width, 8*scale_height)
    plt.show()


def _get_nrows_and_ncols(n_rows, n_subplots):
    if n_rows is None:
        n_rows = int(np.sqrt(n_subplots))
    n_cols = max(int(n_subplots / n_rows), 1)  # ensure n_cols != 0
    n_rows = int(n_subplots / n_cols)

    while not ((n_subplots / n_cols).is_integer() and
               (n_subplots / n_rows).is_integer()):
        n_cols -= 1
        n_rows = int(n_subplots / n_cols)
    return n_rows, n_cols

