"""Shared figure style for the manuscript's panel figures.

Targets RSNA / Radiology: Artificial Intelligence submission requirements:
sans-serif type, column-width canvases, uppercase bold panel labels, and
vector PDF plus 600-dpi TIFF output.

Colour
------
Two jobs, two rules, and they are kept apart:

* **Ordinal severity (Normal -> Grade 3) is magnitude**, so it uses a single
  hue, light to dark (``GRADE_CMAP`` / ``grade_colors()``). A categorical
  rainbow over an ordered scale would hide the ordering that quadratic-weighted
  kappa depends on.
* **Model/track identity is categorical**, so it uses a fixed hue order
  (``SERIES``), assigned by entity and never cycled: the cascade is always blue
  and the multiclass baseline always orange, in every panel.

``SERIES`` passes the colorblind-safe checks under all-pairs comparison, with
one pair (magenta/green) in the 6-8 deutan band. That pair is legal only with
secondary encoding, so any panel drawing both also carries hatch texture
(``HATCHES``) and direct labels. Differences use a diverging blue/orange pair
with a neutral grey midpoint, reusing the same two model hues so "blue favours
the cascade" reads consistently across figures.

Print output only -- there is no dark mode, because the destination is paper.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba

log = logging.getLogger(__name__)

# --- canvas sizes (inches) ----------------------------------------------------
# RSNA column widths: 85 mm single, 140 mm 1.5-column, 175 mm double.
WIDTH_SINGLE = 3.35
WIDTH_ONE_HALF = 5.5
WIDTH_DOUBLE = 6.9

# --- colour -------------------------------------------------------------------

#: Fixed categorical hue order. Index 0 is always the cascade, 1 the baseline.
SERIES: tuple[str, ...] = ("#2A6FAF", "#D1700A", "#2F8A5B", "#B5327F")
CASCADE = SERIES[0]
MULTICLASS = SERIES[1]
GEOMETRIC = SERIES[2]
ACCENT = SERIES[3]

#: Secondary encoding for the magenta/green pair and for print-safety generally.
HATCHES: tuple[str, ...] = ("", "///", "...", "\\\\\\")

#: Single-hue sequential ramp for the confusion matrices (magnitude).
GRADE_CMAP = LinearSegmentedColormap.from_list(
    "kkh_blues",
    ["#FFFFFF", "#DCE7F2", "#AFC9E4", "#7BA6D0", "#4A7FB5", "#1F4E79"],
)

#: Diverging ramp for signed differences: orange (baseline better) <-> neutral
#: <-> blue (cascade better). A neutral grey midpoint, never a hue.
DIFF_CMAP = LinearSegmentedColormap.from_list(
    "kkh_diff", [MULTICLASS, "#F2F2F2", CASCADE]
)

INK = "#1A1A1A"
INK_SECONDARY = "#4D4D4D"
INK_MUTED = "#7A7A7A"
GRID = "#D8D8D8"
SURFACE = "#FFFFFF"

#: Reserved status colours. Never reused as a data series.
STATUS_CRITICAL = "#B02418"  # operative-boundary crossings
STATUS_WARNING = "#C8860D"


def grade_colors(n: int = 5) -> list[tuple[float, float, float, float]]:
    """``n`` evenly spaced steps of the ordinal ramp, light to dark.

    Starts at 0.18 rather than 0 so the lightest grade is still visible as a
    fill against a white page.
    """
    return [GRADE_CMAP(v) for v in np.linspace(0.18, 1.0, n)]


# --- rcParams -----------------------------------------------------------------

_RC = {
    "figure.facecolor": SURFACE,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # Helvetica/Arial first; DejaVu Sans is the always-present fallback.
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.titleweight": "regular",
    "axes.labelcolor": INK,
    "axes.edgecolor": INK_SECONDARY,
    "axes.linewidth": 0.6,
    "axes.facecolor": SURFACE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.alpha": 1.0,
    "text.color": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.4,
    "lines.markersize": 3.5,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.columnspacing": 1.0,
    "legend.handletextpad": 0.5,
    "patch.linewidth": 0.5,
    # Keep text as text in vector output so the publisher can reflow it.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def use_paper_style() -> None:
    """Apply the manuscript rcParams. Call once at the top of each figure script."""
    mpl.rcParams.update(_RC)


# --- panel furniture ----------------------------------------------------------


def panel_label(ax, letter: str, dx: float = -0.06, dy: float = 1.06, **kwargs) -> None:
    """Uppercase bold panel label in axes coordinates, per RSNA style."""
    ax.text(
        dx, dy, letter.upper(), transform=ax.transAxes,
        fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK, **kwargs
    )


def subtle_grid(ax, axis: str = "y") -> None:
    """A recessive grid behind the data, on one axis only."""
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5)


def strip_spines(ax, keep: tuple[str, ...] = ("left", "bottom")) -> None:
    for name, spine in ax.spines.items():
        spine.set_visible(name in keep)


def annotate_n(ax, n: int, prefix: str = "n = ", **kwargs) -> None:
    """Put the sample size on the panel, so each one is readable standalone."""
    ax.text(
        0.99, 0.02, f"{prefix}{n}", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=6, color=INK_MUTED, **kwargs
    )


def readable_text_color(rgba) -> str:
    """Ink colour for a label drawn on top of a filled cell.

    Uses relative luminance so labels stay legible at both ends of the ramp.
    """
    r, g, b, _ = to_rgba(rgba)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#FFFFFF" if luminance < 0.55 else INK


# --- output -------------------------------------------------------------------

#: Formats written for every figure: PDF for the submission, TIFF for
#: production, PNG for pasting into slides and drafts.
DEFAULT_FORMATS = ("pdf", "tiff", "png")


def save_figure(
    fig,
    out_dir: Path,
    stem: str,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    demo: bool = False,
) -> list[Path]:
    """Write one figure in each format, and stamp it when the data is synthetic.

    The demo stamp is deliberately loud and drawn on top of the figure: a demo
    render exists to check layout, and must never be mistaken for a result.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if demo:
        fig.text(
            0.5, 0.5, "SYNTHETIC DEMO DATA\nNOT FOR PUBLICATION",
            ha="center", va="center", fontsize=26, color=STATUS_CRITICAL,
            alpha=0.16, fontweight="bold", rotation=28, zorder=1000,
            linespacing=1.4,
        )
        stem = f"{stem}_DEMO"

    written: list[Path] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        kwargs = {}
        if fmt in {"tiff", "tif"}:
            # LZW keeps the file small enough to email without resampling.
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, format=fmt, **kwargs)
        written.append(path)
    plt.close(fig)
    for path in written:
        log.info("wrote %s (%.0f KB)", path, path.stat().st_size / 1024)
    return written
