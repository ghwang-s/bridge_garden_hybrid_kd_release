"""
Shared figure style matching the paper manuscript.
Import this module before any plotting.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#  Palette (paper-consistent pastel)
BRIDGE      = '#CC7A62'
BRIDGE_LIGHT= '#F2D6CD'
GARDEN      = '#6E9C8B'
GARDEN_LIGHT= '#D5E6DF'
HYBRID      = '#E6BC6A'
HYBRID_LIGHT= '#F7E5BC'
BG          = '#FFFFFF'
TEXT        = '#2F3440'
GRAY        = '#B9C0C8'
GRAY_DARK   = '#6F7782'
GRID_COLOR  = '#E9EDF2'
SPINE_COLOR = '#D5DBE3'
ANNO_EDGE   = '#DDE3EA'

#  Matplotlib rcParams
_BASE_RCPARAMS = {
    # Font
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11.5,
    'axes.titlesize': 12.5,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'legend.title_fontsize': 9.5,
    # Figure
    'figure.dpi': 150,
    'figure.facecolor': BG,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'savefig.facecolor': BG,
    # Axes
    'axes.facecolor': BG,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'axes.edgecolor': SPINE_COLOR,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'axes.axisbelow': True,
    # Grid
    'grid.color': GRID_COLOR,
    'grid.linewidth': 0.38,
    'grid.alpha': 0.42,
    # Ticks
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'xtick.color': TEXT,
    'ytick.color': TEXT,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    # Text
    'text.color': TEXT,
    'axes.labelcolor': TEXT,
    # Lines & markers
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
    'lines.markeredgewidth': 0.8,
    'lines.markeredgecolor': 'white',
    # Legend
    'legend.framealpha': 0.92,
    'legend.edgecolor': ANNO_EDGE,
    'legend.borderpad': 0.5,
    'legend.handlelength': 1.8,
    'legend.handleheight': 0.8,
    'legend.columnspacing': 1.2,
}


def make_rcparams(scale=1.0):
    rc = dict(_BASE_RCPARAMS)
    scale_keys = [
        'font.size',
        'axes.labelsize',
        'axes.titlesize',
        'xtick.labelsize',
        'ytick.labelsize',
        'legend.fontsize',
        'legend.title_fontsize',
        'lines.linewidth',
        'lines.markersize',
        'lines.markeredgewidth',
    ]
    for key in scale_keys:
        rc[key] = rc[key] * scale
    return rc


RCPARAMS = make_rcparams()


def apply_style(scale=1.0, overrides=None):
    """Apply the paper-consistent style globally."""
    rc = make_rcparams(scale=scale)
    if overrides:
        rc.update(overrides)
    plt.rcParams.update(rc)


def anno_box(alpha=0.72):
    """Standard annotation box style."""
    return dict(
        boxstyle='round,pad=0.26',
        facecolor='white',
        alpha=alpha,
        edgecolor=ANNO_EDGE,
        linewidth=0.45,
    )


def clean_legend(ax, loc='best', **kwargs):
    """Create a clean, paper-quality legend."""
    return ax.legend(
        loc=loc,
        framealpha=0.95,
        facecolor='white',
        edgecolor=ANNO_EDGE,
        borderpad=0.45,
        handletextpad=0.55,
        labelspacing=0.4,
        **kwargs,
    )
