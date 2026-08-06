"""mobility_rt.style."""
import numpy as np
import matplotlib.pyplot as plt
from mobility_rt.config import OKABE_ITO


def _set_pub_style():
    """Publication-quality matplotlib defaults for Nature-family journals.

    Nature guidelines: sans-serif font, 7 pt minimum, no top/right spines,
    figures ≤ 183 mm wide (double-column) or ≤ 89 mm (single-column).
    """
    plt.rcParams.update({
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Helvetica Neue", "Helvetica", "Arial",
                                 "DejaVu Sans"],
        "font.size":            8,
        "axes.titlesize":       8,
        "axes.labelsize":       8,
        "xtick.labelsize":      7,
        "ytick.labelsize":      7,
        "legend.fontsize":      7,
        "legend.frameon":       False,
        "legend.handlelength":  1.5,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       0.7,
        "xtick.major.width":    0.6,
        "ytick.major.width":    0.6,
        "xtick.major.size":     3,
        "ytick.major.size":     3,
        "xtick.minor.size":     1.5,
        "ytick.minor.size":     1.5,
        "lines.linewidth":      0.9,
        "patch.linewidth":      0.6,
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.05,
        "image.cmap":           "viridis",
        "pdf.fonttype":         42,   # embed fonts properly in PDFs
        "ps.fonttype":          42,
    })


def _panel_label(ax, letter, x=-0.14, y=1.04):
    """Add bold panel letter (a, b, c …) in top-left corner of axes."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")


def _panel_label_3d(ax, letter):
    """Add bold panel letter to 3D axes via text2D (transAxes coords)."""
    ax.text2D(-0.06, 1.06, letter, transform=ax.transAxes,
              fontsize=10, fontweight="bold", va="top", ha="left")


def _bar3d_Rkj(ax, R_mat, day_label):
    """3D bar chart of pairwise R_kj matrix. Infectors on x-axis, infectees on y-axis."""
    N = R_mat.shape[0]
    xpos = np.repeat(np.arange(N), N)   # infector k
    ypos = np.tile(np.arange(N), N)     # infectee j
    dz   = np.maximum(R_mat.flatten(), 0.0)
    colors = [OKABE_ITO[k % len(OKABE_ITO)] for k in xpos]
    ax.bar3d(xpos - 0.38, ypos - 0.38, np.zeros(N * N),
             0.75, 0.75, dz, color=colors, alpha=0.85, shade=True, linewidth=0.0)
    # Clean panes: no fill, subtle edges, no grid
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#cccccc")
    ax.xaxis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax.yaxis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax.zaxis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels([f"L{i+1}" for i in range(N)], fontsize=4)
    ax.set_yticklabels([f"L{i+1}" for i in range(N)], fontsize=4)
    ax.set_xlabel("Infector $k$", fontsize=6, labelpad=2)
    ax.set_ylabel("Infectee $j$", fontsize=6, labelpad=2)
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.set_tick_params(labelsize=5, pad=0)
    ax.view_init(elev=28, azim=-55)
    ax.tick_params(axis='z', labelsize=5, pad=0)
    ax.set_title(f"$R_{{kj}}$ — {day_label}", fontsize=7, pad=18, fontweight="bold")


def _bar3d_inc(ax, inc_mat, day_label):
    """3D bar chart of pairwise new infections E_{kj}. Infectors on x-axis, infectees on y-axis."""
    N = inc_mat.shape[0]
    xpos = np.repeat(np.arange(N), N)
    ypos = np.tile(np.arange(N), N)
    dz   = np.maximum(inc_mat.flatten(), 0.0)
    colors = [OKABE_ITO[k % len(OKABE_ITO)] for k in xpos]
    ax.bar3d(xpos - 0.38, ypos - 0.38, np.zeros(N * N),
             0.75, 0.75, dz, color=colors, alpha=0.85, shade=True, linewidth=0.0)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#cccccc")
    ax.xaxis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax.yaxis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax.zaxis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels([f"L{i+1}" for i in range(N)], fontsize=4)
    ax.set_yticklabels([f"L{i+1}" for i in range(N)], fontsize=4)
    ax.set_xlabel("Infector $k$", fontsize=6, labelpad=2)
    ax.set_ylabel("Infectee $j$", fontsize=6, labelpad=2)
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.set_tick_params(labelsize=5, pad=0)
    ax.view_init(elev=28, azim=-55)
    ax.tick_params(axis='z', labelsize=5, pad=0)
    ax.set_title(f"$E_{{kj}}$ — {day_label}", fontsize=7, pad=18, fontweight="bold")
