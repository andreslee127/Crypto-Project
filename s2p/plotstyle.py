"""Shared matplotlib style: one fixed colour per policy (never cycled), thin marks, light grid."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
POLICY_COLOR = {"naive": SERIES[0], "ema": SERIES[1], "gp": SERIES[2], "band": SERIES[3]}
POLICY_LABEL = {"naive": "Naive", "ema": "EMA", "gp": "Garleanu-Pedersen", "band": "No-trade band"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "figure.dpi": 110,
})


def save(fig, path, synthetic=False):
    """Save a figure; anything built from synthetic data is stamped SYNTHETIC."""
    fig.tight_layout()
    if synthetic:
        fig.text(0.995, 0.995, "SYNTHETIC DATA", ha="right", va="top", color="#e34948",
                 fontsize=11, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)
